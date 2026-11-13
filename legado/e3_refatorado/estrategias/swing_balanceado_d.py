"""
Estratégia: Swing Balanceado D
================================
"O Ponto Doce" — Swing low em suporte próximo com volume e MM50.

Status: ⚠️ EM AVALIAÇÃO
  - Meta: 25-40 ops/ano
  - Foco em suporte muito próximo (distância máx 1.5%)

Lógica: Entra no rompimento de um swing low local (janela 2 dias)
quando o papel está bem colado no suporte, com volume de convicção
e acima da MM50.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------

PARAMETROS = {
    # --- Detecção de swing low ---
    "janela_swing":           2,      # swing low de 2 dias
    "max_distancia_suporte":  0.015,  # máx 1.5% acima do suporte

    # --- Filtros de entrada ---
    "vol_relativo_min":  1.2,   # volume acima da média (confirmação)
    "rsi_min":           30,
    "rsi_max":           55,
    "close_acima_mm50":  True,

    # --- Saída ---
    "alvo_pct":  0.055,  # +5.5%
    "stop_pct":  0.028,  # -2.8%
    "max_dias":  15,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(row: pd.Series, col: str, default: float = np.nan) -> float:
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)


def _eh_swing_low(row: pd.Series, janela: int) -> bool:
    return _val(row, f"swing_low_{janela}", False) == True


# ---------------------------------------------------------------------------
# Função de sinal
# ---------------------------------------------------------------------------

def fn_sinal(row: pd.Series, p: dict) -> bool:
    janela = p["janela_swing"]
    swing_val = _val(row, f"swing_low_value_{janela}", 0)

    return (
        _eh_swing_low(row, janela)
        and swing_val > 0
        and _val(row, "close") <= swing_val * (1 + p["max_distancia_suporte"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "close") > _val(row, "mm50", 0)
    )


# ---------------------------------------------------------------------------
# Dicionário exportado
# ---------------------------------------------------------------------------

ESTRATEGIA = {
    "nome":      "swing_balanceado_d",
    "descricao": "O Ponto Doce — Filtro de Volume + Distância Suporte Curta",
    "parametros": PARAMETROS,
    "fn_sinal":   fn_sinal,
}
