"""
Monta um painel OHLC ajustado por proventos a partir dos CSVs em legado/temp/
(20 papéis, 2020-01 a 2026-03, gerados pelo pipeline antigo em 2026-03-02).

Ajuste: fator = adj_close / close aplicado a open/high/low/close.
Os indicadores são recalculados aqui, sobre a série ajustada.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PASTA_TEMP = RAIZ / "legado" / "temp"


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    ganho = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    perda = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + ganho / perda)


def adx_wilder(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(pdm, index=h.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * pd.Series(mdm, index=h.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def carregar_painel() -> dict[str, pd.DataFrame]:
    """Retorna {ticker: DataFrame indexado por data} com OHLC ajustado e indicadores."""
    # Os CSVs foram salvos sem a coluna de data. O calendário foi reconstruído casando
    # os fechamentos com o precos.db do backup do E4 (diferença máxima de fechamento ~0).
    calendario = pd.DatetimeIndex(pd.read_csv(Path(__file__).parent / "calendario.csv")["data"])
    painel = {}
    for arq in sorted(PASTA_TEMP.glob("*.csv")):
        ticker = arq.name.split(" - ")[0]
        bruto = pd.read_csv(arq, sep=";")
        assert len(bruto) == len(calendario), f"{ticker}: {len(bruto)} linhas vs calendário {len(calendario)}"
        bruto.index = calendario
        fator = bruto["adj_close"] / bruto["close"]
        df = pd.DataFrame(index=bruto.index)
        for col in ["open", "high", "low", "close"]:
            df[col] = bruto[col] * fator
            df[f"{col}_bruto"] = bruto[col]
        df["volume"] = bruto["volume"]
        df["ibov"] = bruto["close_ibov"].ffill()
        df["selic_meta"] = bruto["selic"].ffill()

        c = df["close"]
        df["ret1"] = c.pct_change()
        for n in (5, 20, 60, 120):
            df[f"ret{n}"] = c.pct_change(n)
        df["rsi"] = rsi_wilder(c)
        df["rsi2"] = rsi_wilder(c, 2)
        df["rsi_bruto"] = rsi_wilder(df["close_bruto"])
        df["ret5_bruto"] = df["close_bruto"].pct_change(5)
        df["adx"] = adx_wilder(df["high"], df["low"], c)
        for n in (20, 50, 200):
            df[f"mm{n}"] = c.rolling(n).mean()
        df["mm50_bruto"] = df["close_bruto"].rolling(50).mean()
        dp = c.rolling(20).std()
        df["bb_sup"] = df["mm20"] + 2 * dp
        df["bb_inf"] = df["mm20"] - 2 * dp
        df["vol_rel"] = df["volume"] / df["volume"].rolling(20).mean()
        df["ibov_mm200"] = df["ibov"].rolling(200).mean()
        df["ibov_mm50"] = df["ibov"].rolling(50).mean()
        painel[ticker] = df
    return painel


if __name__ == "__main__":
    p = carregar_painel()
    print(len(p), "papéis:", ", ".join(p))
    for t, df in p.items():
        print(t, df.index.min().date(), df.index.max().date(), len(df),
              "maior queda diária ajustada: %.1f%%" % (100 * df["ret1"].min()),
              "| bruta: %.1f%%" % (100 * df["close_bruto"].pct_change().min()))
