"""
Testes de controle da revisão de nov/2026.

Objetivo: verificar com números as afirmações do diagnóstico (docs/01_diagnostico.md),
usando apenas dados que já estavam no repositório (scripts/temp/*.csv).

  A. Quanto da "taxa de acerto" vem só da geometria alvo/stop (entradas sem sinal)?
  B. Mean Reversion LIGHT com execução honesta + teste de permutação.
  C. Efeito de entrar no fechamento do sinal (não executável) vs. abertura seguinte.
  D. Régua: CDI, Ibovespa e buy & hold do universo no mesmo período.
  E. Estudo de evento: retorno futuro em excesso ao universo após cada família de sinal.

Limitações (valem para tudo aqui): 20 papéis escolhidos em 2026 (viés de sobrevivência),
~6 anos de dados, sem custos de aluguel/IR. Serve para descartar, não para aprovar.

Uso:  python controles.py
"""
import numpy as np
import pandas as pd

from painel import carregar_painel

RNG = np.random.default_rng(42)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ---------------------------------------------------------------------------
# Simulação de um trade com alvo/stop/prazo
# ---------------------------------------------------------------------------

def simular(o, h, l, c, i_sinal, alvo, stop, max_dias, entrada="abertura"):
    """
    Sinal calculado no fechamento do dia i_sinal.
      entrada="abertura":   compra na abertura de i_sinal+1 (executável)
      entrada="fechamento": compra no fechamento de i_sinal (não executável na prática)
    Stop tem prioridade se alvo e stop caem no mesmo candle; gap abaixo do stop sai na abertura.
    Retorna (retorno_bruto, resultado, indice_saida) ou None se faltar dado.
    """
    n = len(c)
    if entrada == "abertura":
        i0 = i_sinal + 1
        if i0 >= n:
            return None
        pe = o[i0]
        primeiro = i0          # o próprio dia da entrada já conta
    else:
        pe = c[i_sinal]
        primeiro = i_sinal + 1
    pa, ps = pe * (1 + alvo), pe * (1 - stop)
    for k, j in enumerate(range(primeiro, min(primeiro + max_dias, n)), start=1):
        if l[j] <= ps:
            return (min(ps, o[j]) / pe - 1, "stop", j)
        if h[j] >= pa:
            return (max(pa, o[j]) / pe - 1 if o[j] > pa and j != primeiro else alvo, "alvo", j)
        if k == max_dias:
            return (c[j] / pe - 1, "tempo", j)
    return None


def arrays(df):
    return df["open"].values, df["high"].values, df["low"].values, df["close"].values


def resumo(rets, nome=""):
    r = np.asarray(rets, dtype=float)
    if len(r) == 0:
        return {"caso": nome, "n": 0}
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 and r.std() > 0 else np.nan
    return {"caso": nome, "n": len(r), "acerto_%": round(100 * (r > 0).mean(), 1),
            "ret_medio_%": round(100 * r.mean(), 2), "t": round(t, 2)}


# ---------------------------------------------------------------------------
# A. Geometria alvo/stop sem sinal
# ---------------------------------------------------------------------------

def exp_a(painel):
    print("\n" + "=" * 78)
    print("A. ENTRADAS SEM SINAL (todo dia, todo papel) — o que a geometria alvo/stop entrega sozinha")
    print("=" * 78)
    linhas = []
    for alvo, stop, dias in [(0.03, 0.04, 15), (0.03, 0.03, 15), (0.04, 0.05, 15), (0.05, 0.04, 15),
                             (0.05, 0.04, 20), (0.055, 0.028, 15), (0.02, 0.06, 15)]:
        for filtro in ("todos os dias", "só Ibov > MM200"):
            rets, res = [], []
            for tk, df in painel.items():
                o, h, l, c = arrays(df)
                ok = (df["ibov"] > df["ibov_mm200"]).values
                for i in range(200, len(df) - 1):
                    if filtro != "todos os dias" and not ok[i]:
                        continue
                    s = simular(o, h, l, c, i, alvo, stop, dias)
                    if s:
                        rets.append(s[0]); res.append(s[1])
            r = np.array(rets); res = np.array(res)
            linhas.append({"alvo": f"{alvo:.1%}", "stop": f"{stop:.1%}", "dias": dias, "filtro": filtro,
                           "n": len(r), "acerto_%": round(100 * (r > 0).mean(), 1),
                           "bateu_alvo_%": round(100 * (res == "alvo").mean(), 1),
                           "ret_medio_bruto_%": round(100 * r.mean(), 2)})
    print(pd.DataFrame(linhas).to_string(index=False))
    print("\nLeitura: 'acerto' alto com alvo < stop aparece sem nenhum sinal. Só o retorno médio importa,")
    print("e ele precisa ser comparado com esta linha de base, não com zero.")


# ---------------------------------------------------------------------------
# B/C. Mean Reversion LIGHT
# ---------------------------------------------------------------------------

def sinal_light(df, base="ajustado"):
    if base == "ajustado":
        rsi, ret5, close, mm50 = df["rsi"], df["ret5"], df["close"], df["mm50"]
    else:  # como o pipeline original: indicadores sobre preço sem ajuste de proventos
        rsi, ret5, close, mm50 = df["rsi_bruto"], df["ret5_bruto"], df["close_bruto"], df["mm50_bruto"]
    return ((rsi >= 20) & (rsi <= 40) & (ret5 <= -0.03) & (close > mm50)
            & (df["ibov"] > df["ibov_mm200"]) & (df["vol_rel"] >= 0.5) & (df["adx"] <= 40)).values


def trades_de_sinal(painel, fn_sinal, alvo, stop, dias, entrada="abertura"):
    """Um trade por vez por papel (sem sobreposição), como no motor do e3_refatorado."""
    out = []
    for tk, df in painel.items():
        o, h, l, c = arrays(df)
        sig = fn_sinal(df)
        i = 200
        while i < len(df) - 1:
            if sig[i]:
                if entrada == "abertura" and o[i + 1] > c[i] * 1.05:
                    i += 1
                    continue
                s = simular(o, h, l, c, i, alvo, stop, dias, entrada)
                if s:
                    out.append({"ticker": tk, "data_sinal": df.index[i], "ret": s[0], "resultado": s[1]})
                    i = s[2] + 1
                    continue
            i += 1
    return pd.DataFrame(out)


def exp_bc(painel):
    print("\n" + "=" * 78)
    print("B/C. MEAN REVERSION LIGHT (alvo 3% / stop 4% / 15 dias)")
    print("=" * 78)
    linhas = []
    casos = {}
    for base in ("ajustado", "bruto"):
        for entrada in ("abertura", "fechamento"):
            tr = trades_de_sinal(painel, lambda d, b=base: sinal_light(d, b), 0.03, 0.04, 15, entrada)
            casos[(base, entrada)] = tr
            linhas.append(resumo(tr["ret"], f"indicadores s/ preço {base}, entrada no(a) {entrada}"))
    print(pd.DataFrame(linhas).to_string(index=False))

    tr = casos[("ajustado", "abertura")]
    print("\nPor ano (ajustado, entrada na abertura seguinte, bruto de custos):")
    por_ano = tr.groupby(tr["data_sinal"].dt.year)["ret"].agg(n="count", acerto=lambda x: round(100 * (x > 0).mean(), 1),
                                                           ret_medio=lambda x: round(100 * x.mean(), 2),
                                                           soma=lambda x: round(100 * x.sum(), 1))
    print(por_ano.to_string())
    for custo in (0.002, 0.006):
        print(f"  líquido de {custo:.1%} ida-e-volta: ret médio {100 * (tr['ret'].mean() - custo):+.2f}% por trade")
    datas = tr["data_sinal"].dt.to_period("M").value_counts()
    print(f"  concentração: {len(tr)} trades em {tr['data_sinal'].dt.to_period('M').nunique()} meses distintos; "
          f"os 3 meses mais cheios têm {datas.head(3).sum()} trades")

    # Sinais que só existem por causa de queda 'ex-provento' no preço sem ajuste
    falsos = total = 0
    for tk, df in painel.items():
        sb, sa = sinal_light(df, "bruto"), sinal_light(df, "ajustado")
        total += sb.sum(); falsos += (sb & ~sa).sum()
    print(f"  dias com sinal no preço bruto: {total}; destes, {falsos} ({100 * falsos / max(total, 1):.0f}%) "
          f"somem quando o preço é ajustado por proventos")

    # Teste de permutação: mesmo nº de trades por papel, dias aleatórios no mesmo regime (Ibov > MM200)
    print("\nPermutação (5.000 sorteios): entradas aleatórias, mesmo papel, mesmo regime Ibov > MM200")
    pool = {}
    for tk, df in painel.items():
        o, h, l, c = arrays(df)
        ok = (df["ibov"] > df["ibov_mm200"]).values
        rs = [simular(o, h, l, c, i, 0.03, 0.04, 15) for i in range(200, len(df) - 16) if ok[i]]
        pool[tk] = np.array([s[0] for s in rs if s])
    cont = tr["ticker"].value_counts()
    sims = np.zeros(5000)
    for tk, k in cont.items():
        sims += RNG.choice(pool[tk], size=(5000, k)).sum(axis=1)
    sims /= len(tr)
    obs = tr["ret"].mean()
    print(f"  observado: {100 * obs:+.2f}%  |  aleatório: média {100 * sims.mean():+.2f}%, "
          f"p5 {100 * np.quantile(sims, .05):+.2f}%, p95 {100 * np.quantile(sims, .95):+.2f}%")
    print(f"  p-valor (aleatório >= observado): {(sims >= obs).mean():.3f}")


# ---------------------------------------------------------------------------
# D. Régua
# ---------------------------------------------------------------------------

def exp_d(painel):
    print("\n" + "=" * 78)
    print("D. RÉGUA — o que ficar parado rendeu no mesmo período")
    print("=" * 78)
    df0 = next(iter(painel.values()))
    anos = (df0.index[-1] - df0.index[0]).days / 365.25
    cdi_d = (1 + (df0["selic_meta"] - 0.10) / 100) ** (1 / 252) - 1
    cdi = (1 + cdi_d).cumprod()
    ibov = df0["ibov"] / df0["ibov"].iloc[0]
    rets = pd.DataFrame({tk: d["ret1"] for tk, d in painel.items()}).iloc[1:]
    ew = (1 + rets.mean(axis=1)).cumprod()

    def linha(nome, curva):
        r = curva.pct_change().dropna()
        dd = (curva / curva.cummax() - 1).min()
        return {"ativo": nome, "CAGR_%": round(100 * (curva.iloc[-1] ** (1 / anos) - 1), 1),
                "vol_anual_%": round(100 * r.std() * np.sqrt(252), 1), "max_drawdown_%": round(100 * dd, 1)}
    print(f"Período: {df0.index[0].date()} a {df0.index[-1].date()} ({anos:.1f} anos)")
    print(pd.DataFrame([linha("CDI (aprox. pela meta Selic)", cdi), linha("Ibovespa (sem dividendos)", ibov),
                        linha("Buy&hold 20 papéis, pesos iguais, c/ proventos", ew)]).to_string(index=False))
    print("\nCDI por ano-calendário (aprox.):")
    print((100 * ((1 + cdi_d).groupby(cdi_d.index.year).prod() - 1)).round(1).to_string())
    print("\nNota: o buy&hold do universo tem viés de sobrevivência (lista escolhida em 2026).")


# ---------------------------------------------------------------------------
# E. Estudo de evento
# ---------------------------------------------------------------------------

def t_newey_west(x, lag):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(v / n)


def exp_e(painel):
    print("\n" + "=" * 78)
    print("E. ESTUDO DE EVENTO — retorno da abertura de t+1 ao fechamento de t+h,")
    print("   em excesso à média do universo na mesma janela (remove a direção do mercado)")
    print("=" * 78)
    C = pd.DataFrame({tk: d["close"] for tk, d in painel.items()})
    O = pd.DataFrame({tk: d["open"] for tk, d in painel.items()})

    def col(nome):
        return pd.DataFrame({tk: d[nome] for tk, d in painel.items()})
    rsi, rsi2, ret5, ret20, ret120, adx = col("rsi"), col("rsi2"), col("ret5"), col("ret20"), col("ret120"), col("adx")
    mm20, mm50, mm200, bbs, bbi, volrel = col("mm20"), col("mm50"), col("mm200"), col("bb_sup"), col("bb_inf"), col("vol_rel")
    ibov_ok = col("ibov") > col("ibov_mm200")
    valido = mm200.notna()

    def topo(df, k, maior=True):
        return df.where(valido).rank(axis=1, ascending=not maior) <= k

    mom_12_1 = C.shift(5) / C.shift(125) - 1
    sinais = {
        "REVERSÃO  RSI14 < 30":                        rsi < 30,
        "REVERSÃO  IFR2 < 10 e close > MM200 (E2)":    (rsi2 < 10) & (C > mm200),
        "REVERSÃO  queda 5d < -5%":                    ret5 < -0.05,
        "REVERSÃO  close < Bollinger inferior":        C < bbi,
        "REVERSÃO  LIGHT (E3)":                        (rsi >= 20) & (rsi <= 40) & (ret5 <= -0.03) & (C > mm50) & ibov_ok & (volrel >= 0.5) & (adx <= 40),
        "REVERSÃO  3 piores do universo em 5d":        topo(ret5, 3, maior=False),
        "MOMENTUM  close > Bollinger superior (E2)":   C > bbs,
        "MOMENTUM  ret20 > 5%, ADX > 25, Ibov ok (E3)": (ret20 > 0.05) & (adx > 25) & (rsi < 75) & ibov_ok,
        "MOMENTUM  tendência: >MM200, >MM20, ADX>25":  (C > mm200) & (C > mm20) & (adx > 25) & (mm20.diff(3) > 0),
        "MOMENTUM  3 melhores do universo em 5d":      topo(ret5, 3, maior=True),
        "MOMENTUM  3 melhores em ~6 meses (pula 5d)":  topo(mom_12_1, 3, maior=True),
        "MOMENTUM  3 piores em ~6 meses (pula 5d)":    topo(mom_12_1, 3, maior=False),
    }
    meio = C.index[len(C) // 2]
    linhas = []
    for nome, s in sinais.items():
        s = (s & valido).fillna(False)
        for h in (1, 5, 10):
            fwd = C.shift(-h) / O.shift(-1) - 1
            exc = fwd.sub(fwd.where(valido).mean(axis=1), axis=0)
            por_dia = exc.where(s).mean(axis=1)           # média entre papéis com sinal, por data
            bruto = fwd.where(s).stack().mean()
            linhas.append({"sinal": nome, "h": h, "eventos": int(s.sum().sum()), "dias": int(por_dia.notna().sum()),
                           "ret_bruto_%": round(100 * bruto, 2),
                           "excesso_%": round(100 * por_dia.mean(), 2),
                           "t_NW": round(t_newey_west(por_dia.values, h - 1), 2),
                           "exc_1ª_metade_%": round(100 * por_dia[:meio].mean(), 2),
                           "exc_2ª_metade_%": round(100 * por_dia[meio:].mean(), 2)})
    out = pd.DataFrame(linhas)
    print(out.to_string(index=False))
    print(f"\nMetades: até {meio.date()} / depois. São {len(sinais)} sinais x 3 horizontes = {len(out)} testes:")
    print("com tantos testes, |t| em torno de 2 aparece por acaso; só interessa t alto E sinal igual nas duas metades.")
    return out


if __name__ == "__main__":
    painel = carregar_painel()
    exp_d(painel)
    exp_a(painel)
    exp_bc(painel)
    exp_e(painel).to_csv("estudo_evento.csv", index=False)
