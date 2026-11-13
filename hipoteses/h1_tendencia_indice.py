"""
H1 — Alocação por tendência no índice (Ibovespa x CDI), decisão mensal.

PRÉ-REGISTRO (escrito antes de ver qualquer resultado)
------------------------------------------------------
Regra principal: no último pregão de cada mês, se o fechamento do Ibovespa estiver acima
da média dos últimos 10 fechamentos mensais, ficar no índice no mês seguinte; senão, CDI.
(10 meses é o parâmetro canônico da literatura; não foi escolhido olhando estes dados.)

Execução: decisão no fechamento do último pregão do mês; a troca vale com 1 pregão de atraso.
Custos: 0,10% por troca; 0,10% a.a. de taxa do ETF enquanto no índice.
Imposto (versão líquida): 15% sobre ganho realizado em cada saída do índice, com compensação
de prejuízo; CDI com 15% sobre o rendimento; buy & hold paga 15% do ganho no final.
Período de avaliação: 2000-01-01 até hoje. Metades: até 2012-12-31 / de 2013-01-01 em diante.

Critério de aprovação (todos precisam valer, na regra principal, líquido de custos e IR):
  1. CAGR > CDI no período inteiro e em cada metade;
  2. CAGR >= buy & hold do índice no período inteiro;
  3. queda máxima <= 2/3 da queda máxima do buy & hold;
  4. perde para o CDI em menos de 40% das janelas móveis de 12 meses.
As variantes (6, 8, 12 meses; momentum de 12 meses; MM200 diária) são só sensibilidade:
não podem ser usadas para trocar a regra principal depois.

Nota: o Ibovespa é um índice de retorno total (reinveste proventos), então é comparável ao BOVA11
antes da taxa. Uso:  python hipoteses/h1_tendencia_indice.py
"""
import io
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

INICIO_AVAL = "2000-01-01"
CORTE = "2013-01-01"
CUSTO_TROCA = 0.001
TAXA_ETF_AA = 0.001
IR = 0.15
RAIZ = Path(__file__).resolve().parents[1]


# ----------------------------- dados -----------------------------

def baixar_ibov() -> pd.Series:
    import requests
    erros = []
    for tentativa in range(3):
        try:
            import yfinance as yf
            df = yf.download("^BVSP", start="1994-07-01", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = df["Close"].dropna()
            if len(s) > 5000:
                return s
            erros.append(f"yfinance devolveu {len(s)} linhas")
        except Exception as e:  # noqa: BLE001
            erros.append(f"yfinance: {e}")
        time.sleep(5)
    try:
        url = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP"
               "?period1=773020800&period2=4102444800&interval=1d")
        j = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).json()["chart"]["result"][0]
        s = pd.Series(j["indicators"]["quote"][0]["close"],
                      index=pd.to_datetime(j["timestamp"], unit="s").normalize()).dropna()
        if len(s) > 5000:
            return s
        erros.append(f"chart API devolveu {len(s)} linhas")
    except Exception as e:  # noqa: BLE001
        erros.append(f"chart API: {e}")
    raise RuntimeError("Não consegui baixar o Ibovespa: " + " | ".join(erros))


def baixar_cdi() -> pd.Series:
    """CDI diário (série 12 do SGS/BCB), em fração ao dia. A API limita a janela a 10 anos."""
    import requests
    partes = []
    for ano in range(1994, date.today().year + 1, 8):
        ini, fim = f"01/01/{ano}", f"31/12/{ano + 7}"
        url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json"
               f"&dataInicial={ini}&dataFinal={fim}")
        for tentativa in range(5):
            try:
                r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 404:      # janela sem dados
                    dados = []
                    break
                r.raise_for_status()
                dados = r.json()
                break
            except Exception as e:  # noqa: BLE001
                if tentativa == 4:
                    raise RuntimeError(f"BCB falhou para {ini}-{fim}: {e}")
                time.sleep(5 * (tentativa + 1))
        if dados:
            d = pd.DataFrame(dados)
            partes.append(pd.Series(d["valor"].astype(float).values / 100,
                                    index=pd.to_datetime(d["data"], dayfirst=True)))
    s = pd.concat(partes).sort_index()
    return s[~s.index.duplicated()]


# ----------------------------- sinais -----------------------------

def sinais(ibov: pd.Series, cdi: pd.Series) -> dict[str, pd.Series]:
    """Devolve, por variante, a posição diária (True = índice) já com o atraso de execução."""
    fim_mes = ibov.groupby(ibov.index.to_period("M")).tail(1)          # último pregão de cada mês
    m = fim_mes.copy()
    out = {}

    def diario(sinal_mensal: pd.Series) -> pd.Series:
        s = sinal_mensal.reindex(ibov.index).ffill().fillna(False).astype(bool)
        return s.shift(2, fill_value=False)   # 1: vale para o retorno do dia seguinte; +1: atraso de execução

    for n in (10, 6, 8, 12):
        out[f"Média de {n} meses"] = diario(m > m.rolling(n).mean())
    cdi_acum = (1 + cdi.reindex(ibov.index).fillna(0)).cumprod()
    cdi_m = cdi_acum.reindex(m.index)
    out["Momentum 12m > CDI 12m"] = diario((m / m.shift(12)) > (cdi_m / cdi_m.shift(12)))
    mm200 = ibov.rolling(200).mean()
    out["MM200 diária (checada no fim do mês)"] = diario((ibov > mm200).reindex(m.index))
    return out


# ----------------------------- simulação -----------------------------

def simular(r_ibov: np.ndarray, r_cdi: np.ndarray, pos: np.ndarray, liquido: bool):
    """Curva de patrimônio. pos[i] = True se o retorno do dia i é do índice."""
    taxa_d = (1 + TAXA_ETF_AA) ** (1 / 252) - 1
    v, base, prejuizo, dentro, trocas = 1.0, 0.0, 0.0, False, 0
    curva = np.empty(len(pos))
    for i in range(len(pos)):
        if pos[i] != dentro:
            trocas += 1
            if dentro:                                   # saindo do índice
                v *= 1 - CUSTO_TROCA
                ganho = v - base
                if liquido:
                    if ganho > 0:
                        tributavel = max(0.0, ganho - prejuizo)
                        prejuizo = max(0.0, prejuizo - ganho)
                        v -= IR * tributavel
                    else:
                        prejuizo += -ganho
            else:                                        # entrando no índice
                v *= 1 - CUSTO_TROCA
                base = v
            dentro = bool(pos[i])
        if dentro:
            v *= 1 + r_ibov[i] - taxa_d
        else:
            v *= 1 + r_cdi[i] * ((1 - IR) if liquido else 1)
        curva[i] = v
    if dentro and liquido:                               # liquida no final para comparar igual
        ganho = v - base
        curva[-1] = v - IR * max(0.0, ganho - prejuizo)
    return curva, trocas


def metricas(curva: pd.Series, cdi_curva: pd.Series) -> dict:
    anos = (curva.index[-1] - curva.index[0]).days / 365.25
    r = curva.pct_change().dropna()
    dd = (curva / curva.cummax() - 1).min()
    jan = curva.pct_change(252).dropna()
    jan_cdi = cdi_curva.pct_change(252).reindex(jan.index)
    return {"CAGR_%": 100 * ((curva.iloc[-1] / curva.iloc[0]) ** (1 / anos) - 1),
            "vol_%": 100 * r.std() * np.sqrt(252),
            "queda_max_%": 100 * dd,
            "pior_12m_%": 100 * jan.min(),
            "perde_do_CDI_12m_%": 100 * (jan < jan_cdi).mean()}


def tabela(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    linhas = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        linhas.append("| " + " | ".join(f"{x:.1f}" if isinstance(x, float) else str(x) for x in row) + " |")
    return "\n".join(linhas)


def rodar(ibov: pd.Series, cdi: pd.Series) -> str:
    ibov = ibov[~ibov.index.duplicated()].sort_index()
    ibov.index = pd.to_datetime(ibov.index).tz_localize(None).normalize()
    cdi = cdi.reindex(ibov.index).fillna(0.0)
    pos_todas = sinais(ibov, cdi)
    r_ibov_all = ibov.pct_change().fillna(0.0)

    def avaliar(ini, fim, liquido):
        idx = ibov.index[(ibov.index >= ini) & (ibov.index < fim)]
        ri, rc = r_ibov_all.reindex(idx).values, cdi.reindex(idx).values
        cdi_curva = pd.Series(simular(ri, rc, np.zeros(len(idx), bool), liquido)[0], index=idx)
        linhas = []
        casos = {"CDI": np.zeros(len(idx), bool), "Buy & hold Ibovespa": np.ones(len(idx), bool)}
        casos.update({k: v.reindex(idx).values for k, v in pos_todas.items()})
        for nome, pos in casos.items():
            curva, trocas = simular(ri, rc, pos, liquido)
            m = metricas(pd.Series(curva, index=idx), cdi_curva)
            linhas.append({"estratégia": nome, **m, "tempo_no_índice_%": 100 * float(pos.mean()), "trocas": trocas})
        return pd.DataFrame(linhas)

    hoje = ibov.index[-1] + pd.Timedelta(days=1)
    ini = pd.Timestamp(INICIO_AVAL)
    corte = pd.Timestamp(CORTE)
    total_liq = avaliar(ini, hoje, True)
    total_bru = avaliar(ini, hoje, False)
    m1 = avaliar(ini, corte, True)
    m2 = avaliar(corte, hoje, True)
    dec = avaliar(pd.Timestamp("2016-01-01"), hoje, True)

    def pega(df, nome, col):
        return float(df.loc[df["estratégia"] == nome, col].iloc[0])
    P, BH, C = "Média de 10 meses", "Buy & hold Ibovespa", "CDI"
    criterios = [
        ("1a. CAGR > CDI no período inteiro", pega(total_liq, P, "CAGR_%") > pega(total_liq, C, "CAGR_%")),
        ("1b. CAGR > CDI na 1ª metade", pega(m1, P, "CAGR_%") > pega(m1, C, "CAGR_%")),
        ("1c. CAGR > CDI na 2ª metade", pega(m2, P, "CAGR_%") > pega(m2, C, "CAGR_%")),
        ("2.  CAGR >= buy & hold no período inteiro", pega(total_liq, P, "CAGR_%") >= pega(total_liq, BH, "CAGR_%")),
        ("3.  Queda máxima <= 2/3 da do buy & hold",
         abs(pega(total_liq, P, "queda_max_%")) <= abs(pega(total_liq, BH, "queda_max_%")) * 2 / 3),
        ("4.  Perde do CDI em < 40% das janelas de 12 meses", pega(total_liq, P, "perde_do_CDI_12m_%") < 40),
    ]
    aprovado = all(ok for _, ok in criterios)

    pos_hoje = {k: bool(v.iloc[-1]) for k, v in pos_todas.items()}
    md = [f"# H1 — Tendência no índice (Ibovespa x CDI)",
          "",
          f"Dados: Ibovespa de {ibov.index[0].date()} a {ibov.index[-1].date()} ({len(ibov)} pregões); CDI do BCB (série 12).",
          f"Avaliação: {ini.date()} a {ibov.index[-1].date()}. Regra principal pré-registrada: **{P}**.",
          "",
          f"## Veredito: {'APROVADA' if aprovado else 'REPROVADA'}",
          ""]
    md += [f"- {'✅' if ok else '❌'} {txt}" for txt, ok in criterios]
    md += ["", "## Período inteiro, líquido de custos e IR", "", tabela(total_liq),
           "", "## Período inteiro, só custos (sem IR)", "", tabela(total_bru),
           "", f"## 1ª metade ({ini.date()} a {corte.date()}), líquido", "", tabela(m1),
           "", f"## 2ª metade ({corte.date()} em diante), líquido", "", tabela(m2),
           "", "## Últimos ~10 anos (2016 em diante), líquido — só informativo", "", tabela(dec),
           "", "## Posição indicada hoje por cada variante", ""]
    md += [f"- {k}: {'ÍNDICE' if v else 'CDI'}" for k, v in pos_hoje.items()]
    md += ["", "Colunas: `perde_do_CDI_12m_%` = fração das janelas móveis de 12 meses em que a estratégia rendeu menos que o CDI.",
           "As variantes além da regra principal são sensibilidade; não servem para escolher parâmetro.", ""]
    return "\n".join(md)


if __name__ == "__main__":
    if "--sintetico" in sys.argv:   # teste de fumaça sem rede
        idx = pd.bdate_range("1994-07-01", "2026-11-01")
        g = np.random.default_rng(0)
        ibov = pd.Series(1000 * np.cumprod(1 + g.normal(0.0005, 0.017, len(idx))), index=idx)
        cdi = pd.Series(0.00045, index=idx)
    else:
        ibov, cdi = baixar_ibov(), baixar_cdi()
    texto = rodar(ibov, cdi)
    print(texto)
    if "--sintetico" not in sys.argv:
        (RAIZ / "relatorios").mkdir(exist_ok=True)
        (RAIZ / "relatorios" / "h1_tendencia_indice.md").write_text(texto, encoding="utf-8")
