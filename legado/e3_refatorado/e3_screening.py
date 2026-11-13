"""
e3_screening.py
================
Screening e backtest walk-forward no universo B3 líquido.

Carrega estratégias dinamicamente de estrategias/<nome>.py,
conforme listado em e3_estrategias.json.

Comandos:
  python e3_screening.py                          # backtest completo (estratégias ativas)
  python e3_screening.py --estrategias a b        # backtest de estratégias específicas
  python e3_screening.py --modo grade             # testa grade de alvos/stops
  python e3_screening.py --anos 3                 # últimos N anos
  python e3_screening.py --ticker PETR4.SA        # debug em papel único
"""

import os
import sys
import json
import warnings
import argparse
import importlib.util
import traceback
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessamento as pp

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

CFG = {
    "universo_liquido_file":  "e3_universo_b3_liquido.txt",
    "universo_ampliado_file": "e3_universo_b3_ampliado.txt",
    "estrategias_json":       "e3_estrategias.json",
    "alvos_stops_json":       "e3_alvos_stops.json",
    "pasta_resultados":       "resultados_screening",
    "dias_historico":         400,
}

# 27 papéis líquidos (fallback se arquivo não encontrado)
UNIVERSO_PADRAO = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA",
    "WEGE3.SA", "RENT3.SA", "LREN3.SA", "RADL3.SA", "SUZB3.SA",
    "GGBR4.SA", "CSNA3.SA", "EMBR3.SA", "JBSS3.SA", "BRFS3.SA",
    "CSAN3.SA", "MGLU3.SA", "HAPV3.SA", "RAIL3.SA", "EQTL3.SA",
    "SBSP3.SA", "ENEV3.SA", "TAEE11.SA", "CPLE6.SA", "VIVT3.SA",
    "MULT3.SA", "PRIO3.SA",
]


# ---------------------------------------------------------------------------
# Carregamento dinâmico de estratégias
# ---------------------------------------------------------------------------

def carregar_estrategia(nome: str) -> dict:
    """
    Importa dinamicamente estrategias/<nome>.py e retorna ESTRATEGIA.
    """
    pasta = Path(__file__).parent / "estrategias"
    caminho = pasta / f"{nome}.py"

    if not caminho.exists():
        raise FileNotFoundError(
            f"Estratégia '{nome}' não encontrada em {caminho}.\n"
            f"Crie o arquivo seguindo o padrão dos existentes."
        )

    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    if not hasattr(modulo, "ESTRATEGIA"):
        raise AttributeError(
            f"Arquivo {caminho} não exporta a variável ESTRATEGIA."
        )

    return modulo.ESTRATEGIA


def carregar_estrategias_ativas() -> list[dict]:
    """
    Lê e3_estrategias.json e carrega as estratégias listadas em 'estrategias_ativas'.
    """
    cfg_path = Path(CFG["estrategias_json"])

    if not cfg_path.exists():
        print(f"[AVISO] {cfg_path} não encontrado. Usando mean_reversion_light.")
        return [carregar_estrategia("mean_reversion_light")]

    with open(cfg_path) as f:
        cfg = json.load(f)

    nomes = cfg.get("estrategias_ativas", [])
    if not nomes:
        print("[AVISO] Nenhuma estratégia ativa em e3_estrategias.json.")
        return []

    estrategias = []
    for nome in nomes:
        try:
            estrategias.append(carregar_estrategia(nome))
            print(f"  [✓] Estratégia carregada: {nome}")
        except Exception as e:
            print(f"  [✗] Erro ao carregar '{nome}': {e}")

    return estrategias


def carregar_config_alvos_stops() -> dict:
    cfg_path = Path(CFG["alvos_stops_json"])
    if not cfg_path.exists():
        return {"modo": "fixo"}
    with open(cfg_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Universo de papéis
# ---------------------------------------------------------------------------

def carregar_universo() -> list[str]:
    caminho = CFG["universo_ampliado_file"]
    if not os.path.exists(caminho):
        print(f"[AVISO] {caminho} não encontrado. Usando universo padrão ({len(UNIVERSO_PADRAO)} papéis).")
        return UNIVERSO_PADRAO

    with open(caminho) as f:
        linhas = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    vistos = set()
    tickers = []
    for t in linhas:
        if t not in vistos:
            tickers.append(t)
            vistos.add(t)

    print(f"[INFO] {len(tickers)} papéis carregados de {caminho}")
    return tickers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(row: pd.Series, col: str, default: float = np.nan) -> float:
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)


def _retorno_nd(row: pd.Series, n: int) -> float:
    return _val(row, f"ret_{n}d", default=-999)


# ---------------------------------------------------------------------------
# Preparação de dados
# ---------------------------------------------------------------------------

def preparar_df(ticker: str, data_base: date) -> pd.DataFrame:
    df = pp.montar_dataset(ticker, dias_total_historico=CFG["dias_historico"], data_base=data_base)
    df = pp.adicionar_features_sniper(df)

    df["vol_financeiro"] = df["close"] * df["volume"]
    df["adtv_20d"] = df["vol_financeiro"].rolling(20).mean()

    # Retornos
    for n in [1, 3, 5, 10, 20]:
        col = f"ret_{n}d"
        if col not in df.columns:
            df[col] = df["close"].pct_change(n)

    # MMs do Ibovespa
    if "close_ibov" in df.columns:
        df["mm50_ibov"]  = df["close_ibov"].rolling(50).mean()
        df["mm100_ibov"] = df["close_ibov"].rolling(100).mean()
        df["mm200_ibov"] = df["close_ibov"].rolling(200).mean()

    # MMs do papel
    df["mm20"] = df["close"].rolling(20).mean()
    df["mm50"] = df["close"].rolling(50).mean()

    # Máximas de períodos (sem lookahead)
    for n in [10, 15, 20, 30]:
        df[f"max_{n}d"] = df["high"].shift(1).rolling(n).max()

    # ATR
    if "atr" not in df.columns and "ATR_14" in df.columns:
        df["atr"] = df["ATR_14"]
    elif "atr" not in df.columns:
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["prev_close"]).abs(),
            (df["low"]  - df["prev_close"]).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = df["tr"].rolling(14).mean()

    # ATR percentil (para estratégias que usam)
    def _percentil_rank(series, window=60):
        def _pct(x):
            if len(x) < 2 or pd.isna(x.iloc[-1]):
                return 50
            return (x < x.iloc[-1]).sum() / len(x) * 100
        return series.rolling(window).apply(_pct, raw=False)

    df["atr_percentil"] = _percentil_rank(df["atr"], window=60)

    # Swing lows (sem lookahead)
    for janela in [1, 2, 3, 5, 10]:
        min_n_ant = df["low"].shift(1).rolling(janela).min()
        cond = (
            (df["low"] <= min_n_ant) &
            (df["low"] <  df["low"].shift(1)) &
            (df["low"] <= df["low"].shift(2))
        )
        df[f"swing_low_{janela}"]       = cond
        df[f"swing_low_value_{janela}"] = df["low"].where(cond, np.nan).ffill()

    # Mínimas recentes
    for janela in [10, 20, 30]:
        df[f"min_{janela}d"] = df["low"].shift(1).rolling(janela).min()

    df["__ticker__"] = ticker
    return df


# ---------------------------------------------------------------------------
# Motor de backtest
# ---------------------------------------------------------------------------

def backtest_estrategia(ticker: str, df: pd.DataFrame, est: dict,
                        alvo_override: float = None,
                        stop_override: float = None) -> pd.DataFrame:
    """
    Simula a estratégia no DataFrame e retorna lista de trades.

    Parâmetros:
        alvo_override / stop_override: substituem alvo_pct / stop_pct da estratégia
                                       (usados na varredura de grade).
    """
    p = est["parametros"].copy()
    if alvo_override is not None:
        p["alvo_pct"] = alvo_override
    if stop_override is not None:
        p["stop_pct"] = stop_override

    fn_sinal    = est["fn_sinal"]
    TAXAS       = 0.004   # 0.4% round-trip
    SLIPPAGE    = 0.001   
    trades      = []
    em_posicao  = False
    preco_ent   = None
    data_ent    = None
    dias_pos    = 0

    # Reset para iterar com acesso ao próximo dia
    df_exec = df.reset_index()

    for i in range(len(df_exec) - 1):
        row = df_exec.iloc[i]
        next_row = df_exec.iloc[i+1]
        
        if em_posicao:
            dias_pos += 1
            # Saída usa High/Low do dia atual para checar stop/alvo
            alvo_p = preco_ent * (1 + p["alvo_pct"])
            stop_p = preco_ent * (1 - p["stop_pct"])
            
            if row["low"] <= stop_p:
                preco_saida = min(stop_p, row["open"]) # Gap down sai no open
                resultado = "stop"
            elif row["high"] >= alvo_p:
                preco_saida = alvo_p
                resultado = "alvo"
            elif dias_pos >= p["max_dias"]:
                preco_saida = row["close"]
                resultado = "tempo"
            else:
                continue # Mantém posição

            # Execução da Saída
            ret_pct = 100 * ((preco_saida * (1 - SLIPPAGE)) / (preco_ent * (1 + SLIPPAGE)) - (1 + TAXAS) + 1)
            trades.append({
                "ticker": ticker,
                "data_entrada": data_ent,
                "data_saida": row["index"],
                "retorno_%": round(ret_pct, 2),
                "resultado": resultado
            })
            em_posicao = False
            sinal_ontem = False

        else:
            # Verifica sinal no fechamento de T0
            if fn_sinal(row, p):
                # Tenta entrar no Open de T+1
                preco_ent = next_row["open"] * (1 + SLIPPAGE)
                data_ent = next_row["index"]
                
                # Regra de Segurança: Se abrir em gap de alta > 2%, cancela a entrada
                if next_row["open"] > row["close"] * 1.05:
                    continue
                
                em_posicao = True
                dias_pos = 0

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Métricas de resumo
# ---------------------------------------------------------------------------

def calcular_metricas(df_trades: pd.DataFrame, max_dias: int) -> dict:
    """
    Calcula métricas consolidadas a partir de um DataFrame de trades.
    """
    if df_trades.empty:
        return None

    n          = len(df_trades)
    taxa       = (df_trades["retorno_%"] > 0).sum() / n
    ret_medio  = df_trades["retorno_%"].mean()
    ret_std    = df_trades["retorno_%"].std()

    # Retorno anualizado — usa max_dias como proxy do holding period médio
    ret_anual  = ret_medio * (252 / max_dias) if max_dias > 0 else 0
    sharpe     = (ret_medio / ret_std * np.sqrt(252 / max_dias)) if ret_std > 0 else 0

    # Drawdown
    df_s = df_trades.sort_values("data_entrada").reset_index(drop=True)
    df_s["equity"]   = (1 + df_s["retorno_%"] / 100).cumprod()
    df_s["peak"]     = df_s["equity"].cummax()
    df_s["drawdown"] = (df_s["equity"] / df_s["peak"] - 1) * 100
    max_dd = df_s["drawdown"].min()

    return {
        "n_trades":  n,
        "taxa_%":    round(taxa * 100, 1),
        "ret_medio": round(ret_medio, 2),
        "ret_anual": round(ret_anual, 1),
        "sharpe":    round(sharpe, 2),
        "drawdown":  round(max_dd, 1),
    }


# ---------------------------------------------------------------------------
# Screening principal
# ---------------------------------------------------------------------------

def rodar_screening(
    tickers: list[str],
    estrategias: list[dict],
    data_base: date,
    n_anos: int = 5,
    ticker_unico: str = None,
    modo_grade: bool = False,
) -> dict:
    """
    Itera pelos papéis e estratégias, roda walk-forward e gera resumo anual.

    Retorna:
        dict com resultados por estratégia → por ano → lista de trades
    """
    if ticker_unico:
        tickers = [ticker_unico]

    ano_atual = data_base.year
    anos      = list(range(ano_atual - n_anos + 1, ano_atual + 1))

    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)

    cfg_as  = carregar_config_alvos_stops()
    combos  = _montar_combos(cfg_as, estrategias, modo_grade)

    print(f"\n{'='*70}")
    print(f"SCREENING B3  |  {len(tickers)} papéis  |  anos: {anos[0]}-{anos[-1]}")
    print(f"Estratégias ativas: {[e['nome'] for e in estrategias]}")
    if len(combos) > len(estrategias):
        print(f"Grade alvo/stop: {len(combos)//len(estrategias)} combinações por estratégia")
    print(f"{'='*70}\n")

    # Estrutura: {(est_nome, combo_key): {ano: [trades_df]}}
    resultados = {}
    for combo in combos:
        key = _combo_key(combo)
        resultados[key] = {ano: [] for ano in anos}

    for ano in anos:
        print(f"\n{'─'*60}")
        print(f"  ANO {ano}")
        print(f"{'─'*60}")  

        for i, ticker in enumerate(tickers, 1):
            if i % 5 == 0 or len(tickers) <= 10:
                print(f"  [{i:02d}/{len(tickers)}] {ticker}...", end=" ", flush=True)

            try:
                df = preparar_df(ticker, date(ano, 12, 31))
                if df.empty or len(df) < 60:
                    if i % 5 == 0 or len(tickers) <= 10:
                        print("insuf")
                    continue

                for combo in combos:
                    est        = combo["estrategia"]
                    alvo_ovr   = combo.get("alvo_pct")
                    stop_ovr   = combo.get("stop_pct")
                    dias_ovr   = combo.get("max_dias")

                    # Copia parâmetros com overrides
                    est_temp = dict(est)
                    if alvo_ovr or stop_ovr or dias_ovr:
                        est_temp = dict(est)
                        est_temp["parametros"] = dict(est["parametros"])
                        if alvo_ovr:
                            est_temp["parametros"]["alvo_pct"] = alvo_ovr
                        if stop_ovr:
                            est_temp["parametros"]["stop_pct"] = stop_ovr
                        if dias_ovr:
                            est_temp["parametros"]["max_dias"]  = dias_ovr

                    trades = backtest_estrategia(ticker, df, est_temp)

                    if trades.empty:
                        continue

                    trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                    inicio = pd.Timestamp(date(ano, 1, 1))
                    fim    = pd.Timestamp(date(ano, 12, 31))
                    trades_ano = trades[
                        (trades["data_entrada"] >= inicio) &
                        (trades["data_entrada"] <= fim)
                    ]

                    if not trades_ano.empty:
                        key = _combo_key(combo)
                        resultados[key][ano].append(trades_ano)

                if i % 5 == 0 or len(tickers) <= 10:
                    print("✓")

            except Exception:
                if i % 5 == 0 or len(tickers) <= 10:
                    print("ERR")

        print()

    # -----------------------------------------------------------------------
    # Relatório
    # -----------------------------------------------------------------------
    _imprimir_relatorio(resultados, combos, anos)
    _salvar_resultados(resultados, combos, anos, data_base)

    return resultados


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _montar_combos(cfg_as: dict, estrategias: list[dict], modo_grade: bool) -> list[dict]:
    """
    Monta lista de combinações (estratégia × alvo × stop × max_dias).
    Se modo 'fixo', retorna uma combo por estratégia com os parâmetros padrão.
    Se modo 'grade' ou --modo grade, expande a grade.
    """
    modo = cfg_as.get("modo", "fixo")
    if modo_grade:
        modo = "grade"

    combos = []

    for est in estrategias:
        if modo == "grade":
            grade = cfg_as.get("grade", {})
            alvos = grade.get("alvos_pct", [est["parametros"]["alvo_pct"]])
            stops = grade.get("stops_pct", [est["parametros"]["stop_pct"]])
            dias_ = grade.get("max_dias",  [est["parametros"]["max_dias"]])

            for alvo in alvos:
                for stop in stops:
                    for dias in dias_:
                        combos.append({
                            "estrategia": est,
                            "alvo_pct":   alvo,
                            "stop_pct":   stop,
                            "max_dias":   dias,
                        })
        else:
            combos.append({"estrategia": est})

    return combos


def _combo_key(combo: dict) -> str:
    est  = combo["estrategia"]
    nome = est["nome"]
    alvo = combo.get("alvo_pct", est["parametros"]["alvo_pct"])
    stop = combo.get("stop_pct", est["parametros"]["stop_pct"])
    dias = combo.get("max_dias",  est["parametros"]["max_dias"])
    return f"{nome}|A{alvo*100:.1f}|S{stop*100:.1f}|D{dias}"


def _imprimir_relatorio(resultados: dict, combos: list[dict], anos: list[int]):
    print(f"\n{'='*70}")
    print("📊 RESUMO ANUAL POR ESTRATÉGIA")
    print(f"{'='*70}\n")

    for combo in combos:
        key  = _combo_key(combo)
        est  = combo["estrategia"]
        alvo = combo.get("alvo_pct", est["parametros"]["alvo_pct"])
        stop = combo.get("stop_pct", est["parametros"]["stop_pct"])
        dias = combo.get("max_dias",  est["parametros"]["max_dias"])

        print(f"\n{'─'*60}")
        print(f"  {est['nome']}  |  alvo {alvo*100:.1f}%  stop {stop*100:.1f}%  max {dias}d")
        print(f"{'─'*60}")
        print(f"  {'Ano':<6} {'Trades':<8} {'Taxa%':<8} {'Ret/trade':<12} {'Ret Anual':<12} {'Sharpe':<8}")
        print(f"  {'─'*55}")      
        print(f"parametros: {est['parametros']}")

        todos = []

        for ano in anos:
            lista = resultados[key][ano]
            if not lista:
                print(f"  {ano:<6} {'0':<8} {'─':<8} {'─':<12} {'─':<12} {'─':<8}")
                continue

            df_ano = pd.concat(lista, ignore_index=True)
            todos.append(df_ano)
            m = calcular_metricas(df_ano, dias)

            print(
                f"  {ano:<6} {m['n_trades']:<8} {m['taxa_%']:<8.1f} "
                f"{m['ret_medio']:<12.2f} {m['ret_anual']:<12.1f} {m['sharpe']:<8.2f}"
            )

        if todos:
            df_total = pd.concat(todos, ignore_index=True)
            m = calcular_metricas(df_total, dias)
            ops_ano = m["n_trades"] / len(anos)
            print(f"  {'─'*55}")
            print(
                f"  {'TOTAL':<6} {m['n_trades']:<8} {m['taxa_%']:<8.1f} "
                f"{m['ret_medio']:<12.2f} {m['ret_anual']:<12.1f} {m['sharpe']:<8.2f}"
            )
            print(f"  Ops/ano: {ops_ano:.1f}  |  Drawdown: {m['drawdown']:.1f}%")
        else:
            print("  Nenhum trade no período.")


def _salvar_resultados(resultados: dict, combos: list[dict],
                       anos: list[int], data_base: date):
    """
    Salva dois arquivos:
      - trades_detalhados_<data>.csv  (todos os trades)
      - resumo_anual_<data>.csv       (métricas por estratégia × ano)
    """
    pasta = Path(CFG["pasta_resultados"])

    # --- Trades detalhados ---
    todos_trades = []
    for combo in combos:
        key  = _combo_key(combo)
        est  = combo["estrategia"]
        alvo = combo.get("alvo_pct", est["parametros"]["alvo_pct"])
        stop = combo.get("stop_pct", est["parametros"]["stop_pct"])
        for ano in anos:
            for df in resultados[key][ano]:
                df = df.copy()
                df["combo"]    = key
                df["alvo_pct"] = alvo
                df["stop_pct"] = stop
                todos_trades.append(df)

    if todos_trades:
        df_det = pd.concat(todos_trades, ignore_index=True)
        caminho_det = pasta / f"trades_detalhados_{data_base}.csv"
        df_det.to_csv(caminho_det, index=False)
        print(f"\n[✓] Trades detalhados: {caminho_det}")
    else:
        print("\n[!] Nenhum trade gerado.")
        return

    # --- Resumo anual ---
    linhas = []
    for combo in combos:
        key  = _combo_key(combo)
        est  = combo["estrategia"]
        alvo = combo.get("alvo_pct", est["parametros"]["alvo_pct"])
        stop = combo.get("stop_pct", est["parametros"]["stop_pct"])
        dias = combo.get("max_dias",  est["parametros"]["max_dias"])

        for ano in anos:
            lista = resultados[key][ano]
            if not lista:
                linhas.append({
                    "estrategia": est["nome"],
                    "combo":      key,
                    "alvo_pct":   alvo,
                    "stop_pct":   stop,
                    "max_dias":   dias,
                    "ano":        ano,
                    "n_trades":   0,
                    "taxa_%":     None,
                    "ret_medio":  None,
                    "ret_anual":  None,
                    "sharpe":     None,
                    "drawdown":   None,
                })
                continue

            df_ano = pd.concat(lista, ignore_index=True)
            m = calcular_metricas(df_ano, dias)
            linhas.append({
                "estrategia": est["nome"],
                "combo":      key,
                "alvo_pct":   alvo,
                "stop_pct":   stop,
                "max_dias":   dias,
                "ano":        ano,
                **m,
            })

    df_resumo = pd.DataFrame(linhas)
    caminho_res = pasta / f"resumo_anual_{data_base}.csv"
    df_resumo.to_csv(caminho_res, index=False)
    print(f"[✓] Resumo anual:       {caminho_res}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="e3_screening — Backtest walk-forward B3")

    parser.add_argument(
        "--estrategias", nargs="+", default=None,
        help="Nomes das estratégias a testar (sobrepõe e3_estrategias.json)",
    )
    parser.add_argument(
        "--modo", choices=["fixo", "grade"], default=None,
        help="'fixo' usa parâmetros da estratégia; 'grade' testa combinações de alvo/stop",
    )
    parser.add_argument(
        "--anos", type=int, default=5,
        help="Número de anos de walk-forward (padrão: 5)",
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Rodar em um ticker específico (debug)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Data base YYYY-MM-DD (padrão: hoje)",
    )

    args = parser.parse_args()

    data_base = date.fromisoformat(args.data) if args.data else date.today()

    # Carrega estratégias
    if args.estrategias:
        estrategias = []
        for nome in args.estrategias:
            try:
                estrategias.append(carregar_estrategia(nome))
                print(f"  [✓] {nome}")
            except Exception as e:
                print(f"  [✗] {nome}: {e}")
    else:
        print("\nCarregando estratégias de e3_estrategias.json...")
        estrategias = carregar_estrategias_ativas()

    if not estrategias:
        print("[ERRO] Nenhuma estratégia carregada. Abortando.")
        sys.exit(1)

    # Carrega papéis
    tickers = carregar_universo()

    # Roda
    rodar_screening(
        tickers       = tickers,
        estrategias   = estrategias,
        data_base     = data_base,
        n_anos        = args.anos,
        ticker_unico  = args.ticker,
        modo_grade    = (args.modo == "grade"),
    )


if __name__ == "__main__":
    main()
