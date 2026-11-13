"""
Estratégia: Mean Reversion LIGHT
=================================
Única estratégia VALIDADA no universo B3 líquido.

Status: ✅ APROVADA
  - Taxa de acerto: ~81.7%
  - Sharpe: ~1.38
  - Ops/ano: ~3.5 (universo de 27 papéis)

Lógica: Compra em sobrevenda moderada (RSI < 40) quando o papel está
acima da MM50 e o Ibovespa está em regime de alta (acima da MM200).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------

PARAMETROS = {
    # --- Entrada ---
    "rsi_min":          20,     # RSI > 20 (evita colapsos totais)
    "rsi_max":          40,     # RSI < 40 (sobrevendido moderado)
    "queda_5d_min":     -0.03,  # caiu pelo menos 3% nos últimos 5 dias
    "vol_relativo_min": 0.5,    # volume mínimo (mercado não abandonado)
    "adx_max":          40,     # ADX < 40 (queda não é tendência estabelecida)

    # --- Filtros de tendência ---
    "close_acima_mm50":  True,  # papel acima de MM50 (tendência de curto/médio prazo)
    "ibov_acima_mm200":  True,  # Ibovespa em regime de alta

    # --- Saída ---
    "alvo_pct":  0.03,  # +3%
    "stop_pct":  0.04,  # -4%
    "max_dias":  15,
}


# ---------------------------------------------------------------------------
# Helpers locais (duplicados aqui para que o arquivo seja autocontido)
# ---------------------------------------------------------------------------

def _val(row: pd.Series, col: str, default: float = np.nan) -> float:
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)


def _retorno_nd(row: pd.Series, n: int) -> float:
    return _val(row, f"ret_{n}d", default=-999)


def _parametros() -> dict:
    return PARAMETROS.copy()
# ---------------------------------------------------------------------------
# Função de sinal
# ---------------------------------------------------------------------------

def fn_sinal(row: pd.Series, p: dict) -> bool:
    """
    Retorna True se a linha atual gera sinal de entrada para Mean Reversion LIGHT.

    Condições:
      1. RSI em zona de sobrevenda moderada (entre rsi_min e rsi_max)
      2. Queda acumulada nos últimos 5 dias >= queda_5d_min
      3. Close acima da MM50 (mantém tendência de curto prazo)
      4. Ibovespa acima da MM200 (regime de alta — crítico)
      5. Volume mínimo presente
      6. ADX não muito alto (queda não é tendência de baixa consolidada)
    """
    return (
        _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _retorno_nd(row, 5) <= p["queda_5d_min"]
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "ADX_14") <= p["adx_max"]
    )


# ---------------------------------------------------------------------------
# Dicionário exportado (interface padrão com o screening)
# ---------------------------------------------------------------------------

ESTRATEGIA = {
    "nome":      "mean_reversion_light",
    "descricao": "Reversão à média com critérios relaxados (versão operável) — VALIDADA",
    "parametros": PARAMETROS,
    "fn_sinal":   fn_sinal,
}
