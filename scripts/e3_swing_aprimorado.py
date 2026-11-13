"""
screening_regras.py
===================
Screening diário da B3 com estratégia de Momentum Relativo.

Fluxo de uso:
  1. Rode backtest no universo completo → identifica papéis promissores
  2. Use os papéis promissores no screening diário
  3. Ajuste estratégia conforme necessário

Comandos:
  python screening_regras.py --calibrar       # backtest + gera lista de promissores
  python screening_regras.py                  # screening diário (usa lista de promissores)
  python screening_regras.py --ticker PETR4.SA --data 2025-06-10  # debug
"""

from html import parser
import os
import json
import argparse
import random
import warnings
import traceback
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessamento as pp
import utils

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

CFG = {
    "universo_completo_file": "e3_universo_b3_ampliado.txt",   # universo expandido (~70 papéis)
    "universo_liquido_file": "e3_universo_b3_liquido.txt",     # universo líquido (~21 papéis)
    "papeis_promissores_file": "papeis_promissores.json",  # gerado após calibração
    "testar_universo_reduzido": False,  # se True, só testa uma lista reduzida de papeis
    
    "dias_historico":   400,
    "pasta_resultados": "resultados_screening",
    
    # Critérios para um papel ser considerado "promissor" após backtest
    "min_trades":       5,      # mínimo de trades no backtest para ser estatisticamente relevante
    "min_taxa_acerto":  0.60,   # 60% de acerto
    "min_retorno_medio": 0.015, # 1.5% de retorno médio por trade
    
    # --- GESTÃO DE CAPITAL ---
    "capital_disponivel": 100000,  # R$ 100.000 (ajuste conforme seu capital real)
    "risco_por_trade_pct": 0.02,   # 2% de risco por operação (conservador)
    "max_posicoes_simultaneas": 4, # máximo de posições abertas ao mesmo tempo
    
    # --- EMAIL ---
    "enviar_email": True,  # True = envia email automático | False = só imprime
}


# ---------------------------------------------------------------------------
# ESTRATÉGIA: Momentum Relativo
# ---------------------------------------------------------------------------
# Esta é a única estratégia ATIVA. As outras duas (reversão à média e
# sazonalidade) foram desabilitadas por performance insuficiente no backtest.
#
# Se quiser experimentar outras estratégias no futuro, copie a estrutura
# desta e ajuste os parâmetros.
# ---------------------------------------------------------------------------

def calcular_score_sinal(row: pd.Series, p: dict) -> float:
    """
    Calcula score de 0-15 para um sinal baseado em múltiplos indicadores.
    
    Quanto maior o score, mais "forte" é o sinal.
    Opera apenas se score >= threshold definido.
    """
    score = 0
    
    # --- RSI (0-3 pontos) - quanto menor, melhor ---
    rsi = _val(row, "rsi", 50)
    if rsi < 30:
        score += 3
    elif rsi < 50:
        score += 2
    elif rsi < 70:
        score += 1
    # rsi >= 70: 0 pontos (sobrecomprado)
    
    # --- Momentum de 20 dias (0-3 pontos) - quanto maior, melhor ---
    ret_20d = _retorno_nd(row, 20)
    if ret_20d > 0.15:  # +15%
        score += 3
    elif ret_20d > 0.10:  # +10%
        score += 2
    elif ret_20d > 0.05:  # +5%
        score += 1
    
    # --- Momentum de 5 dias (0-2 pontos) - confirmação recente ---
    ret_5d = _retorno_nd(row, 5)
    if ret_5d > 0.03:  # +3%
        score += 2
    elif ret_5d > 0.01:  # +1%
        score += 1
    
    # --- ADX (0-3 pontos) - força da tendência ---
    adx = _val(row, "ADX_14", 0)
    if adx > 35:
        score += 3
    elif adx > 28:
        score += 2
    elif adx > 22:
        score += 1
    
    # --- Volume relativo (0-2 pontos) - liquidez/interesse ---
    vol_rel = _val(row, "vol_relativo", 1.0)
    if vol_rel > 1.5:
        score += 2
    elif vol_rel > 1.0:
        score += 1
    
    # --- Regime de mercado (0-2 pontos) - CRÍTICO ---
    close_ibov = _val(row, "close_ibov", 0)
    mm200_ibov = _val(row, "mm200_ibov", 0)
    if close_ibov > 0 and mm200_ibov > 0:
        if close_ibov > mm200_ibov * 1.05:  # Ibov 5%+ acima de MM200
            score += 2
        elif close_ibov > mm200_ibov:  # Ibov acima de MM200
            score += 1
    
    return score


ESTRATEGIA_MEAN_REVERSION_SUPER_LIGHT_SEM_REGIME = {
    "nome":      "mean_reversion_super_light_sem_regime",
    "descricao": "Reversão à média relaxada SEM filtro de regime (máxima frequência, maior risco)",
    
    "parametros": {
        # --- Entrada (critérios MUITO RELAXADOS + SEM FILTRO DE REGIME) ---
        "rsi_max":               50,    # RSI < 50
        "queda_5d_min":          -0.02, # caiu pelo menos 2%
        "close_acima_mm20":      True,  # acima de MM20
        "ibov_acima_mm200":      False, # ← REMOVIDO (opera em qualquer regime)
        
        # Filtros básicos
        "vol_relativo_min":      0.3,   # volume muito baixo ok
        "adx_max":               45,    # ADX até 45
        "rsi_min":               25,    # RSI > 25
        
        # --- Saída ---
        "alvo_pct":              0.03,  # +3%
        "stop_pct":              0.04,  # -4%
        "max_dias":              15,
    },
    
    "fn_sinal": lambda row, p: (
        # Sobrevendido
        _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "rsi") >= p["rsi_min"]
        # Caiu nos últimos 5 dias
        and _retorno_nd(row, 5) <= p["queda_5d_min"]
        # Ainda em tendência de alta de curtíssimo prazo
        and _val(row, "close") > _val(row, "mm20", 0)
        # SEM FILTRO DE REGIME (opera sempre)
        # Volume mínimo
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        # ADX
        and _val(row, "ADX_14") <= p["adx_max"]
    ),
}


ESTRATEGIA_MEAN_REVERSION_SUPER_LIGHT = {
    "nome":      "mean_reversion_super_light",
    "descricao": "Reversão à média com critérios muito relaxados (máxima frequência)",
    
    "parametros": {
        # --- Entrada (critérios MUITO RELAXADOS) ---
        "rsi_max":               50,    # RSI < 50 (meio da escala)
        "queda_5d_min":          -0.02, # caiu pelo menos 2%
        "close_acima_mm20":      True,  # acima de MM20 (curtíssimo prazo)
        "ibov_acima_mm200":      True,  # mercado em regime de alta (mantém)
        
        # Filtros básicos
        "vol_relativo_min":      0.3,   # volume muito baixo ok
        "adx_max":               45,    # ADX até 45
        "rsi_min":               25,    # RSI > 25 (evita colapsos extremos)
        
        # --- Saída ---
        "alvo_pct":              0.03,  # +3%
        "stop_pct":              0.04,  # -4%
        "max_dias":              15,
    },
    
    "fn_sinal": lambda row, p: (
        # Sobrevendido (muito relaxado)
        _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "rsi") >= p["rsi_min"]
        # Caiu nos últimos 5 dias (relaxado)
        and _retorno_nd(row, 5) <= p["queda_5d_min"]
        # Ainda em tendência de alta de curtíssimo prazo
        and _val(row, "close") > _val(row, "mm20", 0)
        # Regime de mercado favorável
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
        # Volume mínimo (muito relaxado)
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        # ADX (muito relaxado)
        and _val(row, "ADX_14") <= p["adx_max"]
    ),
}


ESTRATEGIA_MEAN_REVERSION_LIGHT = {
    "nome":      "mean_reversion_light",
    "descricao": "Reversão à média com critérios relaxados (versão operável)",
    
    "parametros": {
        # --- Entrada (critérios RELAXADOS) ---
        "rsi_max":               40,    # RSI < 40 (relaxado de 30)
        "queda_5d_min":          -0.03, # caiu pelo menos 3% (relaxado de 5%)
        "close_acima_mm50":      True,  # acima de MM50 (relaxado de MM200)
        "ibov_acima_mm200":      True,  # mercado em regime de alta (mantém)
        
        # Filtros básicos
        "vol_relativo_min":      0.5,   # volume mínimo relaxado
        "adx_max":               40,    # ADX < 40 (relaxado de 35)
        "rsi_min":               20,    # RSI > 20 (evita colapsos totais)
        
        # --- Saída ---
        "alvo_pct":              0.03,  # +3%
        "stop_pct":              0.04,  # -4%
        "max_dias":              15,
    },
    
    "fn_sinal": lambda row, p: (
        # Sobrevendido (relaxado)
        _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "rsi") >= p["rsi_min"]  # não pega colapsos totais
        # Caiu nos últimos 5 dias (relaxado)
        and _retorno_nd(row, 5) <= p["queda_5d_min"]
        # Ainda em tendência de alta de curto/médio prazo (MM50 em vez de MM200)
        and _val(row, "close") > _val(row, "mm50", 0)
        # Regime de mercado favorável (mantém rigoroso)
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
        # Volume mínimo (relaxado)
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        # ADX não muito alto (relaxado)
        and _val(row, "ADX_14") <= p["adx_max"]
    ),
}


# ============================================================================
# BREAKOUT: DIAGNÓSTICO COM CRITÉRIOS ULTRA-RELAXADOS
# ============================================================================

ESTRATEGIA_BREAKOUT_BASICO_DEBUG = {
    "nome": "breakout_basico_debug",
    "descricao": "DEBUG: mostra valores",
    
    "parametros": {
        "periodo_breakout": 10,
        "alvo_pct": 0.06,
        "stop_pct": 0.04,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        # Debug: imprimir valores
        print(f"  Close: {_val(row, 'close'):.2f} | Max10d: {_val(row, 'max_10d', -999):.2f} | Rompeu: {_val(row, 'close') > _val(row, 'max_10d', 0)}") or
        (_val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0))
    ) if False else (  # Desabilita debug por padrão
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
    ),
}


ESTRATEGIA_BREAKOUT_BASICO = {
    "nome": "breakout_basico",
    "descricao": "Breakout BÁSICO - só rompimento de 10d",
    
    "parametros": {
        "periodo_breakout": 10,         # Período MENOR (10 dias)
        # SEM filtro de volume
        # SEM filtro de ADX
        # SEM filtro de MM
        # SEM filtro de regime
        "alvo_pct": 0.06,
        "stop_pct": 0.04,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        # APENAS: rompeu máxima de 10 dias
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
    ),
}


ESTRATEGIA_BREAKOUT_VOLUME = {
    "nome": "breakout_volume",
    "descricao": "Breakout 10d + Volume (sem ADX, sem regime)",
    
    "parametros": {
        "periodo_breakout": 10,
        "vol_confirmacao_min": 1.0,
        # SEM ADX
        # SEM MM
        # SEM regime
        "alvo_pct": 0.06,
        "stop_pct": 0.04,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
    ),
}


ESTRATEGIA_BREAKOUT_VOLUME_ADX = {
    "nome": "breakout_volume_adx",
    "descricao": "Breakout 10d + Volume + ADX (sem regime)",
    
    "parametros": {
        "periodo_breakout": 10,
        "vol_confirmacao_min": 1.0,
        "adx_min": 10,
        # SEM regime
        "alvo_pct": 0.06,
        "stop_pct": 0.04,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
        and _val(row, "ADX_14") >= p["adx_min"]
    ),
}


ESTRATEGIA_BREAKOUT_COMPLETO_SEM_REGIME = {
    "nome": "breakout_completo_sem_regime",
    "descricao": "Breakout 10d + Volume + ADX + MM50 (sem regime)",
    
    "parametros": {
        "periodo_breakout": 10,
        "vol_confirmacao_min": 1.0,
        "adx_min": 10,
        "close_acima_mm50": True,
        # SEM regime Ibov
        "alvo_pct": 0.06,
        "stop_pct": 0.04,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
        and _val(row, "ADX_14") >= p["adx_min"]
        and _val(row, "close") > _val(row, "mm50", 0)
    ),
}


# ============================================================================
# ESTRATÉGIA EM TESTE: PULLBACK EM TENDÊNCIA
# ============================================================================

ESTRATEGIA_PULLBACK_ULTRA_SIMPLES = {
    "nome": "pullback_ultra_simples",
    "descricao": "TESTE: só MM20",
    
    "parametros": {
        "alvo_pct": 0.04,
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        # SÓ TESTAR: close > MM20
        _val(row, "close") > _val(row, "mm20", 0)
    ),
}

# ======================================================================
# 🚀 BLOCO DE ESTRATÉGIAS REFINADAS - SETEMBRO/2025
# ======================================================================


# ======================================================================
# 🛡️ ESTRATÉGIAS ROBUSTAS - FOCO EM REGIME E CONFIRMAÇÃO
# ======================================================================

ESTRATEGIA_SWING_BALANCEADO_A = {
    "nome": "swing_balanceado_a",
    "descricao": "V1 - Confirmado + Filtro de Regime IBOV",
    "parametros": {
        "janela": 10,
        "vol_relativo_min": 1.0,           # Volume médio (sem exigir spike raro)
        "rsi_max": 50,                     # Filtra papéis já esticados
        "close_acima_mm50": True,          # Papel em tendência de alta
        "alvo_pct": 0.05,                  # Alvo redondo (5%)
        "stop_pct": 0.025,                 # Risco:Retorno 2:1
        "max_dias": 10,                    # Saída por tempo se não andar em 2 semanas
    },
    "fn_sinal": lambda row, p: (
        _val(row, f"swing_confirmado_{p['janela']}") == True
        and _val(row, "close") > _val(row, "mm50", 0)
        # --- FILTRO DE REGIME (O Coração da Robustez) ---
        and _val(row, "close_ibov", 0) > _val(row, "mm50_ibov", 0)
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
    ),
}
ESTRATEGIA_SWING_BALANCEADO_B = {
    "nome": "swing_balanceado_b",
    "descricao": "Capitulação Extrema - Pânico + Volume (V2)",
    "parametros": {
        "janela": 10,
        "rsi_max": 30,
        "vol_relativo_min": 1.3,           # Exige spike de volume
        "alvo_pct": 0.07,                  # Alvo maior para captar o 'dead cat bounce'
        "stop_pct": 0.04,                  # Stop mais folgado para volatilidade
        "max_dias": 10,
    },
    "fn_sinal": lambda row, p: (
        _val(row, f"swing_capitulacao_{p['janela']}") == True
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
    ),
}

ESTRATEGIA_SWING_BALANCEADO_C = {
    "nome": "swing_balanceado_c",
    "descricao": "Sniper Rigoroso - Filtro de ADX e Tendência (V3)",
    "parametros": {
        "janela": 10,
        "adx_max": 35,                     # Evita entrar em quedas livres verticais
        "alvo_pct": 0.05,
        "stop_pct": 0.025,
        "max_dias": 12,
    },
    "fn_sinal": lambda row, p: (
        _val(row, f"swing_rigoroso_{p['janela']}") == True
        and _val(row, "ADX_14", 100) <= p["adx_max"]
    ),
}

ESTRATEGIA_SWING_BALANCEADO_D = {
    "nome": "swing_balanceado_d",
    "descricao": "V2 - Candle Verde + Filtro de Regime IBOV",
    "parametros": {
        "janela": 5,
        "max_distancia_suporte": 0.03,     # Margem de 3% da mínima
        "vol_relativo_min": 1.0,
        "close_acima_mm50": True,          # Papel em tendência
        "alvo_pct": 0.05,                  # Alvo padrão
        "stop_pct": 0.025,                 # Stop padrão
        "max_dias": 10,
    },
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela"])
        and _val(row, "close") > _val(row, "open")   # Confirmação: Candle de Alta
        and _val(row, "close") > _val(row, "mm50", 0)
        # --- FILTRO DE REGIME (Evita o 'moedor' de 2024) ---
        and _val(row, "close_ibov", 0) > _val(row, "mm50_ibov", 0)
        and _val(row, "close") <= _val(row, f"swing_low_value_{p['janela']}", 0) * (1 + p["max_distancia_suporte"])
    ),
}

ESTRATEGIA_SWING_BALANCEADO_E = {
    "nome": "swing_balanceado_e",
    "descricao": "Sniper Hammer - Flexibilizado",
    "parametros": {
        "dist_min_10d_max": 0.05,          # Relaxado de 3.5% para 5% (Aumenta trades)
        "vol_relativo_min": 1.0,           # Relaxado de 1.2 para 1.0
        "close_acima_mm50": True,
        "alvo_pct": 0.07,
        "stop_pct": 0.035,
        "max_dias": 12,
    },
    "fn_sinal": lambda row, p: (
        _val(row, "is_hammer") == True
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "close") <= _val(row, "min_10d", 0) * (1 + p["dist_min_10d_max"])
    ),
}


ESTRATEGIA_SWING_SELETIVO_A = {
    "nome": "swing_seletivo_a",
    "descricao": "Swing 3d + critérios médios - EM VALIDAÇÃO",
    
    "parametros": {
        "janela_swing": 3,
        "max_distancia_suporte": 0.02,     # Mais próximo do suporte (era 0.03)
        "vol_relativo_min": 1.0,           # Volume presente (era 0.5)
        "rsi_min": 25,
        "rsi_max": 50,                     # Mais restrito (era 60)
        "close_acima_mm50": True,          # Adiciona MM50 (não tinha)
        "alvo_pct": 0.06,                  # Alvo maior (era 0.08)
        "stop_pct": 0.03,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela_swing"])
        and _val(row, "close") <= _val(row, f"swing_low_value_{p['janela_swing']}", float('inf')) * (1 + p["max_distancia_suporte"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "close") > _val(row, "mm50", 0)
    ),
}

ESTRATEGIA_SWING_SELETIVO_B = {
    "nome": "swing_seletivo_b",
    "descricao": "Swing 5d + ADX - EM VALIDAÇÃO",
    
    "parametros": {
        "janela_swing": 5,                 # Suporte mais forte
        "max_distancia_suporte": 0.025,    # Meio termo
        "vol_relativo_min": 1.2,           # Volume forte
        "rsi_min": 30,
        "rsi_max": 50,
        "adx_min": 20,                     # Adiciona tendência
        "close_acima_mm50": True,
        "alvo_pct": 0.05,
        "stop_pct": 0.025,
        "max_dias": 18,
    },
    
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela_swing"])
        and _val(row, "close") <= _val(row, f"swing_low_value_{p['janela_swing']}", float('inf')) * (1 + p["max_distancia_suporte"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "ADX_14") >= p["adx_min"]
        and _val(row, "close") > _val(row, "mm50", 0)
    ),
}

ESTRATEGIA_SWING_SELETIVO_C = {
    "nome": "swing_seletivo_c",
    "descricao": "Swing 5d + Regime MM50 - EM VALIDAÇÃO",
    
    "parametros": {
        "janela_swing": 5,
        "max_distancia_suporte": 0.02,
        "vol_relativo_min": 1.0,
        "rsi_min": 30,
        "rsi_max": 50,
        "close_acima_mm50": True,
        "ibov_acima_mm50": True,           # Regime MM50 (não MM200)
        "alvo_pct": 0.05,
        "stop_pct": 0.025,
        "max_dias": 18,
    },
    
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela_swing"])
        and _val(row, "close") <= _val(row, f"swing_low_value_{p['janela_swing']}", float('inf')) * (1 + p["max_distancia_suporte"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "close_ibov") > _val(row, "mm50_ibov", 0)
    ),
}

ESTRATEGIA_SWING_CONSERVADOR = {
    "nome": "swing_conservador",
    "descricao": "Swing em suporte forte + confirmação - EM VALIDAÇÃO",
    
    "parametros": {
        "janela_swing": 5,
        "max_distancia_suporte": 0.02,
        "vol_relativo_min": 1.0,
        "rsi_min": 30,
        "rsi_max": 50,
        "close_acima_mm50": True,
        "ibov_acima_mm200": True,
        "alvo_pct": 0.04,
        "stop_pct": 0.02,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela_swing"])
        and _val(row, "close") <= _val(row, f"swing_low_value_{p['janela_swing']}", float('inf')) * (1 + p["max_distancia_suporte"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
    ),
}

ESTRATEGIA_SWING_AGRESSIVO = {
    "nome": "swing_agressivo",
    "descricao": "Swing em suporte + alvos maiores - EM VALIDAÇÃO",
    
    "parametros": {
        "janela_swing": 3,
        "max_distancia_suporte": 0.03,
        "vol_relativo_min": 0.5,
        "rsi_min": 25,
        "rsi_max": 60,
        "close_acima_mm20": True,
        "alvo_pct": 0.08,
        "stop_pct": 0.03,
        "max_dias": 25,
    },
    
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela_swing"])
        and _val(row, "close") <= _val(row, f"swing_low_value_{p['janela_swing']}", float('inf')) * (1 + p["max_distancia_suporte"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "close") > _val(row, "mm20", 0)
    ),
}

ESTRATEGIA_SWING_SCALPER = {
    "nome": "swing_scalper",
    "descricao": "Swing rápido + stop apertado - EM VALIDAÇÃO",
    
    "parametros": {
        "janela_swing": 2,
        "max_distancia_suporte": 0.015,
        "vol_relativo_min": 0.3,
        "alvo_pct": 0.03,
        "stop_pct": 0.015,
        "max_dias": 8,
    },
    
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela_swing"])
        and _val(row, "close") <= _val(row, f"swing_low_value_{p['janela_swing']}", float('inf')) * (1 + p["max_distancia_suporte"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
    ),
}

ESTRATEGIA_SWING_POSICIONAL = {
    "nome": "swing_posicional",
    "descricao": "Swing de longo prazo + alvos grandes - EM VALIDAÇÃO",
    
    "parametros": {
        "janela_swing": 10,
        "max_distancia_suporte": 0.05,
        "vol_relativo_min": 1.2,
        "rsi_min": 30,
        "rsi_max": 55,
        "adx_min": 20,
        "close_acima_mm50": True,
        "ibov_acima_mm200": True,
        "alvo_pct": 0.12,
        "stop_pct": 0.05,
        "max_dias": 40,
    },
    
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela_swing"])
        and _val(row, "close") <= _val(row, f"swing_low_value_{p['janela_swing']}", float('inf')) * (1 + p["max_distancia_suporte"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "ADX_14") >= p["adx_min"]
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
    ),
}

ESTRATEGIA_PULLBACK_BASICO = {
    "nome": "pullback_basico",
    "descricao": "Pullback BÁSICO - só correção em tendência",
    
    "parametros": {
        # APENAS: close > MM20 + correção
        "close_acima_mm20": True,
        "queda_5d_min": -0.05,          # Caiu >= 5%
        "queda_5d_max": -0.01,          # Caiu <= 1%
        # SEM outros filtros
        "alvo_pct": 0.04,
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, "mm20", 0)
        and _retorno_nd(row, 5) >= p["queda_5d_min"]  # >= -0.05 (não caiu mais que 5%)
        and _retorno_nd(row, 5) <= p["queda_5d_max"]  # <= -0.01 (caiu pelo menos 1%)
    ),
}

ESTRATEGIA_PULLBACK_COM_RSI = {
    "nome": "pullback_com_rsi",
    "descricao": "Pullback + RSI (sem regime)",
    
    "parametros": {
        "close_acima_mm20": True,
        "queda_5d_min": -0.05,
        "queda_5d_max": -0.01,
        "rsi_min": 20,
        "rsi_max": 60,
        # SEM regime Ibov
        "alvo_pct": 0.04,
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, "mm20", 0)
        and _retorno_nd(row, 5) >= p["queda_5d_min"]  # >= -0.05
        and _retorno_nd(row, 5) <= p["queda_5d_max"]  # <= -0.01
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
    ),
}

ESTRATEGIA_PULLBACK_COM_VOLUME = {
    "nome": "pullback_com_volume",
    "descricao": "Pullback + RSI + Volume (sem regime)",
    
    "parametros": {
        "close_acima_mm20": True,
        "queda_5d_min": -0.05,
        "queda_5d_max": -0.01,
        "rsi_min": 20,
        "rsi_max": 60,
        "vol_relativo_min": 0.3,
        # SEM regime Ibov
        "alvo_pct": 0.04,
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, "mm20", 0)
        and _retorno_nd(row, 5) >= p["queda_5d_min"]  # >= -0.05
        and _retorno_nd(row, 5) <= p["queda_5d_max"]  # <= -0.01
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
    ),
}

ESTRATEGIA_PULLBACK_COM_REGIME = {
    "nome": "pullback_com_regime",
    "descricao": "Pullback + RSI + Volume + Regime",
    
    "parametros": {
        "close_acima_mm20": True,
        "queda_5d_min": -0.05,
        "queda_5d_max": -0.01,
        "rsi_min": 20,
        "rsi_max": 60,
        "vol_relativo_min": 0.3,
        "ibov_acima_mm200": True,       # COM regime
        "alvo_pct": 0.04,
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, "mm20", 0)
        and _retorno_nd(row, 5) <= p["queda_5d_min"]
        and _retorno_nd(row, 5) <= p["queda_5d_max"]
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
    ),
}

# Versões antigas (mantidas para referência)
ESTRATEGIA_PULLBACK_STRICT = {
    "nome": "pullback_strict",
    "descricao": "Pullback em tendência forte - EM VALIDAÇÃO",
    
    "parametros": {
        "mm20_acima_mm50": True,
        "mm50_acima_mm200": True,
        "close_acima_mm20": True,
        "queda_5d_min": -0.05,
        "queda_5d_max": -0.02,
        "distancia_maxima_20d": 0.10,
        "rsi_min": 30,
        "rsi_max": 50,
        "vol_relativo_min": 0.8,
        "adx_min": 20,
        "ibov_acima_mm200": True,
        "alvo_pct": 0.05,
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "mm20", 0) > _val(row, "mm50", 0)
        and _val(row, "mm50", 0) > _val(row, "close_norm_mm200", 0) * _val(row, "close", 0)
        and _val(row, "close") > _val(row, "mm20", 0)
        and _retorno_nd(row, 5) <= p["queda_5d_min"]
        and _retorno_nd(row, 5) >= p["queda_5d_max"]
        and _val(row, "close") >= _val(row, "max_20d", 0) * (1 - p["distancia_maxima_20d"])
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "ADX_14") >= p["adx_min"]
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
    ),
}

ESTRATEGIA_PULLBACK_LIGHT = {
    "nome": "pullback_light",
    "descricao": "Pullback balanceado - EM VALIDAÇÃO",
    
    "parametros": {
        "mm20_acima_mm50": True,
        "close_acima_mm20": True,
        "queda_5d_min": -0.04,
        "queda_5d_max": -0.015,
        "distancia_maxima_20d": 0.15,
        "rsi_min": 25,
        "rsi_max": 55,
        "vol_relativo_min": 0.5,
        "adx_min": 15,
        "ibov_acima_mm200": True,
        "alvo_pct": 0.04,
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "mm20", 0) > _val(row, "mm50", 0)
        and _val(row, "close") > _val(row, "mm20", 0)
        and _retorno_nd(row, 5) <= p["queda_5d_min"]
        and _retorno_nd(row, 5) >= p["queda_5d_max"]
        and _val(row, "close") >= _val(row, "max_20d", 0) * (1 - p["distancia_maxima_20d"])
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "ADX_14") >= p["adx_min"]
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
    ),
}

ESTRATEGIA_PULLBACK_SUPER_LIGHT = {
    "nome": "pullback_super_light",
    "descricao": "Pullback mínimo - EM VALIDAÇÃO",
    
    "parametros": {
        "close_acima_mm20": True,
        "queda_5d_min": -0.03,
        "queda_5d_max": -0.01,
        "rsi_min": 20,
        "rsi_max": 60,
        "vol_relativo_min": 0.3,
        "ibov_acima_mm200": True,
        "alvo_pct": 0.04,
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, "mm20", 0)
        and _retorno_nd(row, 5) <= p["queda_5d_min"]
        and _retorno_nd(row, 5) >= p["queda_5d_max"]
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
    ),
}

# ============================================================================
# ESTRATÉGIA EM TESTE: BREAKOUT (4 VERSÕES DE REGIME - ANTIGAS)
# ============================================================================

ESTRATEGIA_BREAKOUT_REGIME_MM200 = {
    "nome": "breakout_regime_mm200",
    "descricao": "Breakout com regime MM200 (longo prazo) - EM VALIDAÇÃO",
    
    "parametros": {
        "periodo_breakout": 20,
        "vol_confirmacao_min": 1.0,
        "adx_min": 10,
        "close_acima_mm50": True,
        "ibov_acima_mm200": True,       # ← Regime MM200
        "alvo_pct": 0.06,
        "stop_pct": 0.04,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
        and _val(row, "ADX_14") >= p["adx_min"]
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)  # MM200
    ),
}


ESTRATEGIA_BREAKOUT_REGIME_MM100 = {
    "nome": "breakout_regime_mm100",
    "descricao": "Breakout com regime MM100 (médio prazo) - EM VALIDAÇÃO",
    
    "parametros": {
        "periodo_breakout": 20,
        "vol_confirmacao_min": 1.0,
        "adx_min": 10,
        "close_acima_mm50": True,
        "ibov_acima_mm100": True,       # ← Regime MM100
        "alvo_pct": 0.06,
        "stop_pct": 0.04,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
        and _val(row, "ADX_14") >= p["adx_min"]
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "close_ibov") > _val(row, "mm100_ibov", 0)  # MM100
    ),
}


ESTRATEGIA_BREAKOUT_REGIME_MM50 = {
    "nome": "breakout_regime_mm50",
    "descricao": "Breakout com regime MM50 (curto prazo) - EM VALIDAÇÃO",
    
    "parametros": {
        "periodo_breakout": 20,
        "vol_confirmacao_min": 1.0,
        "adx_min": 10,
        "close_acima_mm50": True,
        "ibov_acima_mm50": True,        # ← Regime MM50
        "alvo_pct": 0.06,
        "stop_pct": 0.04,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
        and _val(row, "ADX_14") >= p["adx_min"]
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "close_ibov") > _val(row, "mm50_ibov", 0)  # MM50
    ),
}


ESTRATEGIA_BREAKOUT_SEM_REGIME = {
    "nome": "breakout_sem_regime",
    "descricao": "Breakout SEM filtro de regime (opera sempre) - EM VALIDAÇÃO",
    
    "parametros": {
        "periodo_breakout": 20,
        "vol_confirmacao_min": 1.0,
        "adx_min": 10,
        "close_acima_mm50": True,
        # SEM filtro de regime
        "alvo_pct": 0.06,
        "stop_pct": 0.04,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
        and _val(row, "ADX_14") >= p["adx_min"]
        and _val(row, "close") > _val(row, "mm50", 0)
        # SEM filtro Ibov (opera em qualquer regime)
    ),
}


# Versões antigas (manter para referência)
ESTRATEGIA_BREAKOUT_LIGHT = {
    "nome": "breakout_light",
    "descricao": "Breakout com critérios balanceados - EM VALIDAÇÃO",
    
    "parametros": {
        # === ENTRADA (rompimento com confirmação) ===
        "periodo_breakout": 20,         # Rompe máxima de 20 dias
        "vol_confirmacao_min": 1.3,     # Volume > 1.3× média (relaxado de 1.5)
        "atr_percentil_max": 50,        # ATR < p50 (relaxado de 40)
        "adx_min": 15,                  # ADX > 15 (relaxado de 20)
        "close_acima_mm200": True,      # Tendência de longo prazo
        "ibov_acima_mm200": True,       # Regime de alta (CRÍTICO)
        
        # === SAÍDA ===
        "alvo_pct": 0.06,               # +6%
        "stop_pct": 0.04,               # -4%
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        # Rompeu máxima
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        # Volume confirmação (relaxado)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
        # Volatilidade (relaxado - pode estar consolidando ou não)
        and _val(row, "atr_percentil", 50) <= p["atr_percentil_max"]
        # Tendência começando (relaxado)
        and _val(row, "ADX_14") >= p["adx_min"]
        # Tendência de longo prazo
        and _val(row, "close_norm_mm200") > 1.0
        # Regime de mercado
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
    ),
}


ESTRATEGIA_BREAKOUT_SUPER_LIGHT = {
    "nome": "breakout_super_light",
    "descricao": "Breakout com critérios muito relaxados (máxima frequência) - EM VALIDAÇÃO",
    
    "parametros": {
        # === ENTRADA (rompimento básico) ===
        "periodo_breakout": 20,         # Rompe máxima de 20 dias
        "vol_confirmacao_min": 1.0,     # Volume > 1.0× média (só precisa de volume normal)
        "adx_min": 10,                  # ADX > 10 (muito relaxado)
        "close_acima_mm50": True,       # Acima de MM50 (relaxado de MM200)
        "ibov_acima_mm200": True,       # Regime de alta (mantém)
        
        # === SAÍDA ===
        "alvo_pct": 0.06,               # +6%
        "stop_pct": 0.04,               # -4%
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        # Rompeu máxima
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        # Volume normal (muito relaxado)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
        # Tendência mínima
        and _val(row, "ADX_14") >= p["adx_min"]
        # Acima de MM50 (curto prazo)
        and _val(row, "close") > _val(row, "mm50", 0)
        # Regime de mercado
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
    ),
}


# ESTRATÉGIA ORIGINAL (muito restritiva - gerou 0 sinais)
ESTRATEGIA_BREAKOUT = {
    "nome": "breakout_volatilidade",
    "descricao": "Breakout com consolidação prévia - MUITO RESTRITIVO",
    
    "parametros": {
        # === ENTRADA (rompimento de resistência) ===
        "periodo_breakout": 20,         # Rompe máxima de 20 dias
        "vol_confirmacao_min": 1.5,     # Volume > 1.5× média (confirmação)
        "atr_percentil_max": 40,        # ATR < p40 (estava consolidando)
        "atr_janela": 60,               # Janela para calcular percentil
        "adx_min": 20,                  # Tendência se formando
        "close_acima_mm200": True,      # Tendência de longo prazo
        "ibov_acima_mm200": True,       # Regime de alta (CRÍTICO)
        
        # === SAÍDA ===
        "alvo_pct": 0.06,               # +6% (maior que mean reversion)
        "stop_pct": 0.04,               # -4%
        "max_dias": 20,                 # Mais tempo que mean reversion
    },
    
    "fn_sinal": lambda row, p: (
        # Rompeu máxima de N dias
        _val(row, "close") > _val(row, f"max_{p['periodo_breakout']}d", 0)
        # Volume alto (confirmação do rompimento)
        and _val(row, "vol_relativo") >= p["vol_confirmacao_min"]
        # Estava em consolidação (baixa volatilidade antes do rompimento)
        and _val(row, "atr_percentil", 50) <= p["atr_percentil_max"]
        # Tendência começando a se formar
        and _val(row, "ADX_14") >= p["adx_min"]
        # Tendência de longo prazo positiva
        and _val(row, "close_norm_mm200") > 1.0
        # Regime de mercado favorável
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
    ),
}


ESTRATEGIA_MEAN_REVERSION = {
    "nome":      "mean_reversion",
    "descricao": "Compra em sobrevenda exagerada com reversão esperada",
    
    "parametros": {
        # --- Entrada (COMPRA NA QUEDA) ---
        "rsi_max":               30,    # RSI < 30 (sobrevendido)
        "queda_5d_min":          -0.05, # caiu pelo menos 5% nos últimos 5 dias
        "bb_posicao_max":        0.2,   # preço no quintil inferior das Bollinger Bands
        "close_acima_mm200":     True,  # papel ainda em tendência de alta de longo prazo
        "ibov_acima_mm200":      True,  # mercado em regime de alta
        
        # Filtros de qualidade
        "vol_relativo_min":      0.8,   # volume mínimo (não pode estar abandonado)
        "adx_max":               35,    # ADX não muito alto (queda não é tendência estabelecida)
        
        # --- Saída ---
        "alvo_pct":              0.03,  # +3% (recuperação modesta, não ganância)
        "stop_pct":              0.04,  # -4% (proteção se colapsar)
        "max_dias":              15,    # tempo para recuperar
    },
    
    "fn_sinal": lambda row, p: (
        # Sobrevendido
        _val(row, "rsi") <= p["rsi_max"]
        # Caiu nos últimos 5 dias
        and _retorno_nd(row, 5) <= p["queda_5d_min"]
        # Ainda em tendência de alta de longo prazo
        and _val(row, "close_norm_mm200") > 1.0
        # Regime de mercado favorável
        and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
        # Volume mínimo
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        # ADX não muito alto (não é tendência de queda estabelecida)
        and _val(row, "ADX_14") <= p["adx_max"]
        # Posição nas Bollinger Bands (próximo da banda inferior)
        and _calcular_bb_position(row) <= p["bb_posicao_max"]
    ),
}


def _calcular_bb_position(row: pd.Series) -> float:
    """
    Calcula posição do preço nas Bollinger Bands (0 = banda inferior, 1 = banda superior).
    """
    close = _val(row, "close", 0)
    bb_superior = _val(row, "bb_superior", 0)
    bb_inferior = _val(row, "bb_inferior", 0)
    
    if bb_superior == 0 or bb_inferior == 0 or bb_superior == bb_inferior:
        return 0.5  # default (meio das bandas)
    
    # Normaliza: 0 = na banda inferior, 1 = na banda superior
    position = (close - bb_inferior) / (bb_superior - bb_inferior)
    return max(0, min(1, position))  # clamp entre 0 e 1


# Mantém estratégias anteriores
ESTRATEGIA_MOMENTUM_SCORING = {
    "nome":      "momentum_scoring",
    "descricao": "Sistema de pontuação adaptativo - opera apenas sinais com score alto",
    
    "parametros": {
        # --- Score mínimo para operar ---
        "score_minimo":      8,     # será testado: 6, 7, 8, 9, 10
        
        # --- Filtros básicos (eliminam sinais obviamente ruins) ---
        "close_acima_mm200": True,  # papel acima de sua própria MM200
        "rsi_max_absoluto":  80,    # nunca opera com RSI > 80
        "adx_min_absoluto":  15,    # nunca opera com ADX < 15 (sem tendência)
        
        # --- Saída por preço ---
        "alvo_pct":          0.05,  # +5%
        "stop_pct":          0.04,  # -4%
        
        # --- Saída por tempo ---
        "max_dias":          10,
    },
    
    "fn_sinal": lambda row, p: (
        # Filtros eliminatórios (passa/falha)
        _val(row, "close_norm_mm200") > 1.0
        and _val(row, "rsi") <= p["rsi_max_absoluto"]
        and _val(row, "ADX_14") >= p["adx_min_absoluto"]
        # Score suficiente
        and calcular_score_sinal(row, p) >= p["score_minimo"]
    ),
}


ESTRATEGIA_MOMENTUM = {
    "nome":      "momentum_relativo",
    "descricao": "Compra em momentum de alta com tendência forte, mercado favorável e regime de alta",
    
    "parametros": {
        # --- Entrada ---
        "retorno_20d_min":   0.05,  # subiu 5%+ nos últimos 20 dias
        "retorno_5d_min":    0.01,  # ainda subindo nos últimos 5 dias
        "adx_min":           25,    # tendência forte
        "dmp_maior_dmn":     True,  # direcional positivo dominante
        "close_acima_mm200": True,  # tendência de longo prazo positiva
        "rsi_max":           75,    # não entrar em sobrecompra extrema
        
        # --- FILTRO DE REGIME DE MERCADO (CRÍTICO) ---
        "ibov_acima_mm200":  True,  # Ibovespa acima da MM200 (evita bear markets)
        
        # --- Saída por indicador ---
        "rsi_saida":         80,    # sobrecomprado
        "adx_saida_min":     20,    # tendência enfraqueceu
        
        # --- Saída por preço ---
        "alvo_pct":          0.05,  # +5%
        "stop_pct":          0.04,  # -4% (R:R 1.25:1)
        
        # --- Saída por tempo ---
        "max_dias":          10,    # até 10 dias úteis (validado: melhor retorno anualizado)
    },
    
    "fn_sinal": lambda row, p: (
        _retorno_nd(row, 20)      >= p["retorno_20d_min"]
        and _retorno_nd(row, 5)   >= p["retorno_5d_min"]
        and _val(row, "ADX_14")   >= p["adx_min"]
        and _val(row, "rsi")      <= p["rsi_max"]
        and _val(row, "close_norm_mm200") > 1.0
        and (
            not p["dmp_maior_dmn"]
            or _val(row, "DMP_14") > _val(row, "DMN_14")
        )
        # FILTRO DE REGIME: Ibovespa precisa estar acima da MM200
        and (
            not p.get("ibov_acima_mm200", False)
            or _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
        )
    ),
}


# Estratégias Gemini
ESTRATEGIA_MEAN_REVERSION_FAST = {
    "name": "mean_reversion_fast",
    "descricao": "Estiramento curto - Distância MM20 + RSI",
    "parametros": {
        "dist_mm20_min": -0.03,        # Relaxado de -0.04 para -0.03
        "rsi_max": 40,                 # Relaxado de 35 para 40
        "vol_relativo_min": 0.7,       # Relaxado de 0.8 para 0.7
        "close_acima_mm50": True,      
        "alvo_pct": 0.04,              
        "stop_pct": 0.03,              
        "max_dias": 7,                 
    },
    "fn_sinal": lambda row, p: (
        _val(row, "close") < _val(row, "mm20") * (1 + p["dist_mm20_min"])
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "close") > _val(row, "open") 
    ),
}

ESTRATEGIA_THE_STRIKE_ZONE = {
    "name": "the_strike_zone",
    "descricao": "Toque na MM50 após força recente",
    "parametros": {
        "dist_mm50_max": 0.02,         
        "dist_topo_min": 0.03,         # Exige que tenha caído pelo menos 3% do topo recente
        "vol_relativo_min": 0.8,       
        "alvo_pct": 0.06,              
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "close") <= _val(row, "mm50") * (1 + p["dist_mm50_max"])
        # Verifica se o preço caiu em relação à máxima de 20 dias (suporte institucional)
        and _val(row, "close") < _val(row, "max_20d", 0) * (1 - p["dist_topo_min"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
    ),
}

ESTRATEGIA_TURNING_POINT = {
    "name": "turning_point_engolfo",
    "descricao": "Reversão: Close > High Ontem após queda",
    "parametros": {
        "vol_relativo_min": 0.9,
        "alvo_pct": 0.05,
        "stop_pct": 0.03,
        "max_dias": 8,
    },
    "fn_sinal": lambda row, p: (
        # Verifica se vinha de queda (retorno 3 dias < 0)
        _val(row, "ret_3d") < 0
        and _val(row, "close") > _val(row, "high_ontem", 0) # Rompeu máxima de ontem
        and _val(row, "close") > _val(row, "mm50", 0)      # Tendência macro ok
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
    ),
}
# --- VERSÕES CALIBRADAS PARA FREQUÊNCIA E SOBREVIVÊNCIA ---

ESTRATEGIA_VOLATILITY_SQUEEZE_V2 = {
    "nome": "vol_squeeze_v2",
    "descricao": "Breakout 20d (Donchian) + Squeeze Real",
    "parametros": {
        "periodo_breakout": 20,         # Aumentado para 20d (Filtra ruído)
        "atr_percentil_max": 30,       # Um pouco mais de margem
        "vol_relativo_min": 1.1,
        "close_acima_mm50": True,      # Exige tendência de fundo
        "alvo_pct": 0.07,              # Alvo maior para compensar o risco de breakout
        "stop_pct": 0.035,
        "max_dias": 15,
    },
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, "max_20d", 0)
        and _val(row, "atr_percentil") <= p["atr_percentil_max"]
        and _val(row, "close") > _val(row, "mm50", 0)
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
    ),
}

ESTRATEGIA_RS_PULLBACK_V2 = {
    "nome": "rs_pullback_v2",
    "descricao": "Força Relativa + Recuo na MM20",
    "parametros": {
        "dist_mm20_max": 0.02,         # Perto da média de 20
        "alpha_ibov_min": 0.02,        # 2% melhor que o IBOV
        "rsi_max": 55,                 # Relaxado para permitir mais trades
        "alvo_pct": 0.05,
        "stop_pct": 0.025,
        "max_dias": 10,
    },
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, "mm20", 0)
        and _val(row, "close") <= _val(row, "mm20", 0) * (1 + p["dist_mm20_max"])
        and (_val(row, "ret_20d") - _val(row, "ret_20d_ibov", 0)) >= p["alpha_ibov_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "close") > _val(row, "open") # Confirmação de entrada
    ),
}

ESTRATEGIA_TREND_STRENGTH = {
    "nome": "trend_strength_10d",
    "descricao": "Rompimento de 10d COM IBOV confirmando",
    "parametros": {
        "periodo_breakout": 10,
        "vol_relativo_min": 1.2,
        "close_ibov_acima_mm50": True, # SÓ OPERA SE O MERCADO ESTIVER BOM
        "alvo_pct": 0.04,
        "stop_pct": 0.02,
        "max_dias": 7,
    },
    "fn_sinal": lambda row, p: (
        _val(row, "close") > _val(row, "max_10d", 0)
        and _val(row, "close_ibov") > _val(row, "mm50_ibov", 0)
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "ADX_14") > 20   # Exige tendência estabelecida
    ),
}

# ---------------------------------------------------------------------------
# ESTRATÉGIAS DESABILITADAS (mantidas para referência)
# ---------------------------------------------------------------------------
# Você pode reativar essas estratégias copiando o bloco para a lista
# ESTRATEGIAS_ATIVAS se quiser testar novamente no futuro.
# ---------------------------------------------------------------------------

"""
# REVERSÃO À MÉDIA (desabilitada — taxa de acerto 46.7% no backtest)
ESTRATEGIA_REVERSAO = {
    "nome":      "reversao_media",
    "descricao": "Compra em sobrevenda de curto prazo",
    "parametros": {
        "ifr2_entrada":      20,
        "rsi_entrada":       40,
        "close_acima_mm200": True,
        "vol_relativo_max":  1.5,
        "ifr2_saida":        70,
        "rsi_saida":         60,
        "alvo_pct":          0.05,
        "stop_pct":          0.04,
        "max_dias":          10,
    },
    "fn_sinal": lambda row, p: (
        _val(row, "ifr2")         <  p["ifr2_entrada"]
        and _val(row, "rsi")      <  p["rsi_entrada"]
        and _val(row, "vol_relativo") < p["vol_relativo_max"]
        and (not p["close_acima_mm200"] or _val(row, "close_norm_mm200") > 1.0)
    ),
}

# SAZONALIDADE + DIVIDENDOS (desabilitada — taxa de acerto 47.1% após ajustes)
ESTRATEGIA_SAZONALIDADE = {
    "nome":      "sazonalidade_dividendos",
    "descricao": "Compra em valuation atrativo em meses favoráveis",
    "parametros": {
        "meses_favoraveis":      [1, 2, 3, 11, 12],
        "tickers_elegiveis":     ["ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "CSAN3.SA", 
                                  "SUZB3.SA", "GGBR4.SA", "CSNA3.SA", "RENT3.SA",
                                  "BRFS3.SA", "JBSS3.SA", "EMBR3.SA"],
        "close_norm_mm200_max":  1.03,
        "close_norm_mm200_min":  0.85,
        "vol_relativo_max":      1.3,
        "rsi_max":               60,
        "rsi_saida":             70,
        "close_norm_saida":      1.08,
        "alvo_pct":              0.06,
        "stop_pct":              0.04,
        "max_dias":              25,
    },
    "fn_sinal": lambda row, p: (
        _mes_atual(row)                   in p["meses_favoraveis"]
        and _ticker_atual(row)            in p["tickers_elegiveis"]
        and _val(row, "close_norm_mm200") <= p["close_norm_mm200_max"]
        and _val(row, "close_norm_mm200") >= p["close_norm_mm200_min"]
        and _val(row, "vol_relativo")     <= p["vol_relativo_max"]
        and _val(row, "rsi")              <= p["rsi_max"]
    ),
}
"""


# Lista de estratégias ATIVAS (apenas estratégias validadas)
ESTRATEGIAS_ATIVAS = [ESTRATEGIA_MEAN_REVERSION_LIGHT]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eh_swing_low(row, janela: int) -> bool:
    """
    Verifica se a linha atual é um swing low (mínima local).
    Usa coluna pré-calculada 'swing_low_X' do preparar_df.
    """
    swing_col = f"swing_low_{janela}"
    return _val(row, swing_col, False) == True


def _val(row: pd.Series, col: str, default: float = np.nan) -> float:
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)

def _retorno_nd(row: pd.Series, n: int) -> float:
    return _val(row, f"ret_{n}d", default=-999)

def _mes_atual(row: pd.Series) -> int:
    try:
        return pd.to_datetime(row.name).month
    except Exception:
        return -1

def _ticker_atual(row: pd.Series) -> str:
    return row.get("__ticker__", "")


# ---------------------------------------------------------------------------
# Carregar listas de tickers
# ---------------------------------------------------------------------------

def carregar_universo_completo() -> list[str]:
    """Carrega todos os tickers do arquivo universo_b3.txt."""
    caminho = CFG["universo_completo_file"]
    if not os.path.exists(caminho):
        print(f"[AVISO] Arquivo {caminho} não encontrado. Usando lista padrão pequena.")
        return [
            "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA",
            "RENT3.SA", "LREN3.SA", "RADL3.SA", "BBAS3.SA", "SUZB3.SA",
        ]
    
    with open(caminho) as f:
        linhas = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    
    # Remove duplicatas mantendo ordem
    vistos = set()
    tickers = []
    for t in linhas:
        if t not in vistos:
            tickers.append(t)
            vistos.add(t)
    
    return tickers


def carregar_papeis_promissores() -> list[str]:
    """Carrega lista de papéis promissores do arquivo JSON."""
    caminho = CFG["papeis_promissores_file"]
    if not os.path.exists(caminho):
        print(f"[AVISO] {caminho} não encontrado. Use --calibrar primeiro.")
        print(f"        Usando universo completo como fallback.")
        return carregar_universo_completo()
    
    with open(caminho) as f:
        data = json.load(f)
    
    tickers = data.get("tickers", [])
    print(f"[INFO] {len(tickers)} papéis promissores carregados de {caminho}")
    return tickers


def salvar_papeis_promissores(tickers: list[str], metricas: dict):
    """Salva lista de papéis promissores + metadados."""
    caminho = CFG["papeis_promissores_file"]
    data = {
        "data_calibracao": str(date.today()),
        "criterios": {
            "min_trades":       CFG["min_trades"],
            "min_taxa_acerto":  CFG["min_taxa_acerto"],
            "min_retorno_medio": CFG["min_retorno_medio"],
        },
        "tickers": tickers,
        "metricas": metricas,
    }
    with open(caminho, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] {len(tickers)} papéis promissores salvos em {caminho}")


# ---------------------------------------------------------------------------
# Preparação de dados
# ---------------------------------------------------------------------------

def preparar_df(ticker: str, data_base: date) -> pd.DataFrame:
    df = pp.montar_dataset(ticker, dias_total_historico=CFG["dias_historico"], data_base=data_base)
    df = pp.adicionar_features_sniper(df)
    
    for n in [1, 3, 5, 10, 20]:
        col = f"ret_{n}d"
        if col not in df.columns:
            df[col] = df["close"].pct_change(n)
    
    # Adicionar médias móveis do Ibovespa (FILTROS DE REGIME)
    if "close_ibov" in df.columns:
        df["mm50_ibov"] = df["close_ibov"].rolling(window=50).mean()
        df["mm100_ibov"] = df["close_ibov"].rolling(window=100).mean()
        df["mm200_ibov"] = df["close_ibov"].rolling(window=200).mean()
    
    # Adicionar médias móveis do papel (para mean reversion)
    df["mm20"] = df["close"].rolling(window=20).mean()
    df["mm50"] = df["close"].rolling(window=50).mean()
    df["mm200"] = df["close"].rolling(window=200).mean()
    
    df['high_ontem'] = df['high'].shift(1)

    # === FEATURES PARA BREAKOUT ===
    
    # Máximas de múltiplos períodos (EXCLUI dia atual com shift)
    for n in [10, 15, 20, 30]:
        df[f"max_{n}d"] = df["high"].shift(1).rolling(window=n).max()
    
    # ATR (se não existir)
    if "atr" not in df.columns and "ATR_14" in df.columns:
        df["atr"] = df["ATR_14"]
    elif "atr" not in df.columns:
        # Calcula ATR manualmente
        df["tr"] = df[["high", "low", "close"]].apply(
            lambda x: max(
                x["high"] - x["low"],
                abs(x["high"] - x["close"]),
                abs(x["low"] - x["close"])
            ) if pd.notna(x["high"]) and pd.notna(x["low"]) and pd.notna(x["close"]) else 0,
            axis=1
        )
        df["atr"] = df["tr"].rolling(window=14).mean()
    
    # ATR Percentil (posição relativa nos últimos 60 dias)
    def calcular_percentil_rank(series, window=60):
        """Calcula percentil do valor atual na janela móvel"""
        def percentil_atual(x):
            if len(x) < 2:
                return 50
            valor_atual = x.iloc[-1]
            if pd.isna(valor_atual):
                return 50
            return (x < valor_atual).sum() / len(x) * 100
        return series.rolling(window=window).apply(percentil_atual, raw=False)
    
    df["atr_percentil"] = calcular_percentil_rank(df["atr"], window=60)
    
    # === FEATURES PARA SWING HIGHS/LOWS (SEM LOOKAHEAD) ===
    
    # MÉTODO CORRIGIDO: Detecta mínima local SEM usar dados futuros
    # Uma mínima local é detectada quando:
    # 1. Low de hoje < Low dos N dias ANTERIORES
    # 2. Low de hoje < Low de ontem e anteontem (confirmação imediata)
    
    for janela in [2, 3, 5, 10]:
        # Mínima dos últimos N dias (EXCLUINDO hoje)
        min_n_anteriores = df["low"].shift(1).rolling(window=janela).min()
        
        # Condições para ser swing low (SEM lookahead):
        # 1. Low atual é menor ou igual à mínima dos N dias anteriores
        # 2. Low atual é menor que ontem (confirma que é mínima recente)
        # 3. Low atual é menor ou igual a anteontem (confirmação adicional)
        
        condicao_minima_local = (
            (df["low"] <= min_n_anteriores) &  # Menor que N dias atrás
            (df["low"] < df["low"].shift(1)) &  # Menor que ontem
            (df["low"] <= df["low"].shift(2))   # Menor ou igual a anteontem
        )
        
        # Marca como swing low apenas se TODAS condições forem verdadeiras
        df[f"swing_low_{janela}"] = condicao_minima_local
        
        # Valor do último swing low detectado (para cálculo de distância)
        # CORREÇÃO: Nome único por janela para não sobrescrever
        df[f"swing_low_value_{janela}"] = df["low"].where(
            df[f"swing_low_{janela}"], 
            np.nan
        ).ffill()
    
    # === SUPORTES ALTERNATIVOS (Distância da mínima recente) ===
    # Abordagem complementar: mínima dos últimos N dias
    for janela in [10, 20, 30]:
        # Mínima dos últimos N dias (EXCLUINDO hoje)
        df[f"min_{janela}d"] = df["low"].shift(1).rolling(window=janela).min()
        
        # Distância percentual do close até esse suporte
        df[f"dist_suporte_{janela}d"] = (
            (df["close"] - df[f"min_{janela}d"]) / df[f"min_{janela}d"]
        )
    
    # === NOVAS DETECÇÕES DE SWING LOW (APRIMORADAS) ===
    
    # VERSÃO 1: SWING LOW CONFIRMADO (espera recuperação)
    for janela in [5, 10, 20]:
        # Ontem foi a mínima dos últimos N dias
        min_n = df["low"].shift(1).rolling(window=janela).min()
        foi_minima_ontem = (df["low"].shift(1) == min_n)
        
        # Hoje está recuperando
        subindo_hoje = (df["close"] > df["close"].shift(1))
        close_acima_low_ontem = (df["close"] > df["low"].shift(1))
        
        # Volume presente (confirmação)
        vol_ok = (df["volume"] > df["volume"].rolling(20).mean() * 0.8)
        
        df[f"swing_confirmado_{janela}"] = (
            foi_minima_ontem & 
            subindo_hoje & 
            close_acima_low_ontem &
            vol_ok
        )
        
        # Valor do swing confirmado
        df[f"swing_confirmado_value_{janela}"] = df["low"].shift(1).where(
            df[f"swing_confirmado_{janela}"],
            np.nan
        ).ffill()
    
    # VERSÃO 2: SWING LOW COM CAPITULAÇÃO (volume alto + RSI baixo)
    for janela in [5, 10, 20]:
        # É a mínima dos últimos N dias
        min_n = df["low"].shift(1).rolling(window=janela).min()
        eh_minima = (df["low"] <= min_n)
        
        # Volume spike (capitulação)
        vol_spike = (df["volume"] > df["volume"].rolling(20).mean() * 1.3)
        
        # RSI muito baixo (sobrevenda extrema)
        rsi_baixo = (df["rsi"] < 35) if "rsi" in df.columns else True
        
        # Ainda em tendência de alta (acima MM50)
        acima_mm50 = (df["close"] > df["mm50"]) if "mm50" in df.columns else True
        
        df[f"swing_capitulacao_{janela}"] = (
            eh_minima & 
            vol_spike & 
            rsi_baixo &
            acima_mm50
        )
        
        # Valor do swing com capitulação
        df[f"swing_capitulacao_value_{janela}"] = df["low"].where(
            df[f"swing_capitulacao_{janela}"],
            np.nan
        ).ffill()
    
    # VERSÃO 3: SWING LOW RIGOROSO (múltiplas confirmações)
    for janela in [5, 10, 20]:
        # É a mínima dos últimos N dias
        min_n = df["low"].shift(1).rolling(window=janela).min()
        eh_minima = (df["low"] <= min_n)
        
        # Caiu significativamente do topo (pelo menos 5%)
        max_recente = df["high"].shift(1).rolling(window=20).max()
        caiu_do_topo = (df["low"] < max_recente * 0.95)
        
        # RSI oversold
        rsi_oversold = (df["rsi"] < 40) if "rsi" in df.columns else True
        
        # Volume acima da média
        vol_presente = (df["volume"] > df["volume"].rolling(20).mean())
        
        # Tendência de alta (MM50)
        tendencia_alta = (df["close"] > df["mm50"]) if "mm50" in df.columns else True
        
        # Não está em queda livre (ADX não muito alto)
        nao_queda_livre = (df["ADX_14"] < 45) if "ADX_14" in df.columns else True
        
        df[f"swing_rigoroso_{janela}"] = (
            eh_minima &
            caiu_do_topo &
            rsi_oversold &
            vol_presente &
            tendencia_alta &
            nao_queda_livre
        )
        
        # Valor do swing rigoroso
        df[f"swing_rigoroso_value_{janela}"] = df["low"].where(
            df[f"swing_rigoroso_{janela}"],
            np.nan
        ).ffill()

        # === DETECÇÃO DE CANDLESTICK (HAMMER) ===
        corpo = (df["close"] - df["open"]).abs()
        sombra_inferior = df[["open", "close"]].min(axis=1) - df["low"]
        sombra_superior = df["high"] - df[["open", "close"]].max(axis=1)
        amplitude_total = df["high"] - df["low"]

        # Definição de Martelo: Sombra inferior pelo menos 2x o tamanho do corpo
        # E sombra superior muito pequena (menos de 10% da amplitude total)
        df["is_hammer"] = (
            (sombra_inferior > (2 * corpo)) & 
            (sombra_superior < (amplitude_total * 0.15)) &
            (amplitude_total > 0)
        )
    
    df["__ticker__"] = ticker
    return df


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def validar_swing_balanceado(data_base: date = None):
    """
    VALIDAÇÃO: SWING BALANCEADO (meta 25-40 ops/ano)
    
    3 versões relaxadas do SELETIVO A:
    - BALANCEADO A: Distância 2.5%, Vol 0.8×, RSI 25-55
    - BALANCEADO B: Swing 2d, Alvo 5%, Stop 2.5%
    - BALANCEADO C: Swing 2d, Vol 0.7×, MM20, RSI 20-60
    
    Walk-forward correto (dados específicos por ano)
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"VALIDAÇÃO: SWING BALANCEADO (meta 25-40 ops/ano)")
    print(f"Relaxando SELETIVO A para aumentar frequência")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    estrategias = {
        "BALANCEADO A (dist 2.5%, vol 0.8)": ESTRATEGIA_SWING_BALANCEADO_A,
        "BALANCEADO B (swing 2d, alvo 5%)": ESTRATEGIA_SWING_BALANCEADO_B,
        "BALANCEADO C (swing 2d, MM20)": ESTRATEGIA_SWING_BALANCEADO_C,
    }
    
    resultados = {nome: {ano: [] for ano in anos} for nome in estrategias.keys()}
    
    # Walk-forward por ano
    for ano in anos:
        print(f"\n{'='*60}")
        print(f"ANO {ano}")
        print(f"{'='*60}\n")
        
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0:
                print(f"  [{i}/{len(tickers)}]...", end=" ", flush=True)
            
            try:
                df = preparar_df(ticker, date(ano, 12, 31))
                
                if df.empty or len(df) < 50:
                    continue
                
                for nome_estrategia, estrategia in estrategias.items():
                    trades = backtest_estrategia(ticker, df, estrategia)
                    
                    if not trades.empty:
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        
                        inicio = pd.Timestamp(date(ano, 1, 1))
                        fim = pd.Timestamp(date(ano, 12, 31))
                        
                        trades_ano = trades[
                            (trades["data_entrada"] >= inicio) &
                            (trades["data_entrada"] <= fim)
                        ]
                        
                        if not trades_ano.empty:
                            resultados[nome_estrategia][ano].append(trades_ano)
            
            except Exception:
                continue
        
        print()
        for nome in estrategias.keys():
            lista = resultados[nome][ano]
            if lista:
                df_ano = pd.concat(lista, ignore_index=True)
                n = len(df_ano)
                taxa = (df_ano["retorno_%"] > 0).sum() / n
                print(f"  {nome}: {n}t | {taxa:.1%}")
            else:
                print(f"  {nome}: 0t")
    
    # Análise
    print(f"\n{'='*70}")
    print("📊 ANÁLISE COMPLETA")
    print(f"{'='*70}\n")
    
    relatorio = {}
    
    for nome in estrategias.keys():
        print(f"\n{'='*60}")
        print(f"🔬 {nome}")
        print(f"{'='*60}\n")
        
        params = estrategias[nome]["parametros"]
        dias_medio = params["max_dias"]
        
        metricas_por_ano = []
        todos = []
        
        for ano in anos:
            lista = resultados[nome][ano]
            
            if not lista:
                metricas_por_ano.append({
                    "ano": ano,
                    "trades": 0,
                    "taxa": 0,
                    "ret_medio": 0,
                    "ret_anual": 0,
                })
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            todos.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n
            ret_medio = df_ano["retorno_%"].mean()
            ret_anual = ret_medio * (252 / dias_medio)
            
            metricas_por_ano.append({
                "ano": ano,
                "trades": n,
                "taxa": taxa,
                "ret_medio": ret_medio,
                "ret_anual": ret_anual,
            })
        
        # Tabela
        print("📅 ANO A ANO:\n")
        
        df_metricas = pd.DataFrame(metricas_por_ano)
        
        print(f"{'Ano':<6} {'Trades':<8} {'Taxa':<8} {'Ret/trade':<12} {'Ret Anual':<12}")
        print(f"{'-'*60}")
        
        for _, row in df_metricas.iterrows():
            ano_str = str(int(row['ano']))
            trades_str = str(int(row['trades']))
            taxa_str = f"{row['taxa']:.1%}" if row['trades'] > 0 else "-"
            ret_str = f"{row['ret_medio']:.2f}%" if row['trades'] > 0 else "-"
            anual_str = f"{row['ret_anual']:.1f}%" if row['trades'] > 0 else "-"
            
            print(f"{ano_str:<6} {trades_str:<8} {taxa_str:<8} {ret_str:<12} {anual_str:<12}")
        
        if not todos:
            print("\n❌ 0 trades.\n")
            relatorio[nome] = {"aprovado": False}
            continue
        
        # Consolidado
        df_total = pd.concat(todos, ignore_index=True)
        
        n_total = len(df_total)
        ops_ano = n_total / len(anos)
        taxa = (df_total["retorno_%"] > 0).sum() / n_total
        ret_medio = df_total["retorno_%"].mean()
        ret_std = df_total["retorno_%"].std()
        ret_anual = ret_medio * (252 / dias_medio)
        
        sharpe = (ret_medio / ret_std * np.sqrt(252 / dias_medio)) if ret_std > 0 else 0
        
        # Drawdown
        df_total = df_total.sort_values("data_entrada").reset_index(drop=True)
        df_total["equity"] = (1 + df_total["retorno_%"] / 100).cumprod()
        df_total["peak"] = df_total["equity"].cummax()
        df_total["drawdown"] = (df_total["equity"] / df_total["peak"] - 1) * 100
        max_dd = df_total["drawdown"].min()
        
        # Distribuição
        dist = df_total["resultado"].value_counts()
        pct_alvo = dist.get("alvo", 0) / n_total * 100
        pct_stop = dist.get("stop", 0) / n_total * 100
        
        # Anos
        anos_com = df_metricas[df_metricas["trades"] > 0]
        anos_pos = (anos_com["ret_anual"] > 0).sum()
        anos_neg = (anos_com["ret_anual"] < 0).sum()
        pior = anos_com["ret_anual"].min() if len(anos_com) > 0 else 0
        melhor = anos_com["ret_anual"].max() if len(anos_com) > 0 else 0
        
        print(f"\n📈 CONSOLIDADO:\n")
        print(f"  {n_total} trades | {ops_ano:.1f}/ano")
        print(f"  Taxa: {taxa:.1%} | Sharpe: {sharpe:.2f}")
        print(f"  Ret/trade: {ret_medio:.2f}% | Ret anual: {ret_anual:.1f}%")
        print(f"  Drawdown: {max_dd:.1f}%")
        print(f"  Melhor: {melhor:.1f}% | Pior: {pior:.1f}%")
        print(f"  Alvo: {pct_alvo:.1f}% | Stop: {pct_stop:.1f}%")
        print(f"  Anos +/-: {anos_pos}/{anos_neg}")
        
        # Critérios
        print(f"\n✅ CRITÉRIOS:\n")
        
        aprovado = True
        
        # Frequência
        if 25 <= ops_ano <= 40:
            print(f"  ✅ Frequência: {ops_ano:.1f}/ano (meta 25-40) 🎯")
        elif ops_ano > 40:
            print(f"  ⚠️  Frequência: {ops_ano:.1f}/ano (acima da meta)")
        else:
            print(f"  ⚠️  Frequência: {ops_ano:.1f}/ano (abaixo da meta)")
        
        # Taxa
        if taxa >= 0.65:
            print(f"  ✅ Taxa {taxa:.1%} >= 65%")
        else:
            print(f"  ❌ Taxa {taxa:.1%} < 65%")
            aprovado = False
        
        # Sharpe
        if sharpe >= 1.0:
            print(f"  ✅ Sharpe {sharpe:.2f} >= 1.0")
        else:
            print(f"  ❌ Sharpe {sharpe:.2f} < 1.0")
            aprovado = False
        
        # Retorno
        if ret_anual >= 30:
            print(f"  ✅ Ret {ret_anual:.1f}% >= 30%")
        else:
            print(f"  ⚠️  Ret {ret_anual:.1f}% < 30%")
        
        # DD
        if max_dd >= -30:
            print(f"  ✅ DD {max_dd:.1f}% >= -30%")
        else:
            print(f"  ❌ DD {max_dd:.1f}% < -30%")
            aprovado = False
        
        # Pior
        if pior >= -20:
            print(f"  ✅ Pior {pior:.1f}% >= -20%")
        else:
            print(f"  ❌ Pior {pior:.1f}% < -20%")
            aprovado = False
        
        print(f"\n{'='*60}")
        print(f"{'✅ APROVADO' if aprovado else '❌ REPROVADO'}")
        
        relatorio[nome] = {
            "aprovado": aprovado,
            "trades": n_total,
            "ops_ano": ops_ano,
            "taxa": taxa,
            "sharpe": sharpe,
            "ret_anual": ret_anual,
            "drawdown": max_dd,
            "pior": pior,
        }
    
    # Resumo
    print(f"\n{'='*70}")
    print("🏆 RESUMO FINAL")
    print(f"{'='*70}\n")
    
    print("📊 COMPARAÇÃO COM OUTRAS ESTRATÉGIAS:\n")
    print(f"{'Estratégia':<30} {'Ops/ano':<10} {'Taxa':<8} {'Sharpe':<8}")
    print(f"{'-'*70}")
    print(f"{'Mean Reversion LIGHT':<30} {'3.5':<10} {'81.7%':<8} {'1.38':<8}")
    print(f"{'SELETIVO A (original)':<30} {'11.2':<10} {'87.5%':<8} {'5.57':<8}")
    print(f"{'Swing SCALPER':<30} {'72.8':<10} {'74.7%':<8} {'5.23':<8}")
    print()
    
    for nome, m in relatorio.items():
        if m.get("aprovado"):
            ops_str = f"{m['ops_ano']:.1f}"
            taxa_str = f"{m['taxa']:.1%}"
            sharpe_str = f"{m['sharpe']:.2f}"
            status = "🎯" if 25 <= m['ops_ano'] <= 40 else ""
            
            print(f"{'✅ ' + nome:<30} {ops_str:<10} {taxa_str:<8} {sharpe_str:<8} {status}")
        else:
            print(f"{'❌ ' + nome:<30} {'REPROVADO':<10}")
    
    print(f"\n{'='*70}\n")
    
    return relatorio


def validar_swing_seletivo(data_base: date = None):
    """
    VALIDAÇÃO: SWING SELETIVO (meta 20-30 ops/ano)
    
    Walk-forward CORRETO: baixa dados específicos para cada ano
    
    3 versões intermediárias:
    - SELETIVO A: Swing 3d + Vol forte + MM50 + 6%/3%
    - SELETIVO B: Swing 5d + ADX + MM50 + 5%/2.5%
    - SELETIVO C: Swing 5d + Regime MM50 + 5%/2.5%
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"VALIDAÇÃO: SWING SELETIVO (meta 20-30 ops/ano)")
    print(f"Walk-forward CORRETO: dados específicos por ano")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    estrategias = {
        "SELETIVO A (3d, Vol+, MM50, 6%)": ESTRATEGIA_SWING_SELETIVO_A,
        "SELETIVO B (5d, ADX, MM50, 5%)": ESTRATEGIA_SWING_SELETIVO_B,
        "SELETIVO C (5d, Regime MM50, 5%)": ESTRATEGIA_SWING_SELETIVO_C,
    }
    
    # Estrutura: {estrategia: {ano: [trades]}}
    resultados = {nome: {ano: [] for ano in anos} for nome in estrategias.keys()}
    
    # WALK-FORWARD CORRETO: testar ano por ano
    for ano in anos:
        print(f"\n{'='*60}")
        print(f"ANO {ano} (walk-forward: dados até {ano}-12-31)")
        print(f"{'='*60}\n")
        
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0:
                print(f"  [{i}/{len(tickers)}]...", end=" ", flush=True)
            
            try:
                # CRÍTICO: Baixar dados SÓ ATÉ fim do ano testado
                df = preparar_df(ticker, date(ano, 12, 31))
                
                if df.empty or len(df) < 50:
                    continue
                
                # Testar cada estratégia
                for nome_estrategia, estrategia in estrategias.items():
                    trades = backtest_estrategia(ticker, df, estrategia)
                    
                    if not trades.empty:
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        
                        # Filtrar trades do ano
                        inicio = pd.Timestamp(date(ano, 1, 1))
                        fim = pd.Timestamp(date(ano, 12, 31))
                        
                        trades_ano = trades[
                            (trades["data_entrada"] >= inicio) &
                            (trades["data_entrada"] <= fim)
                        ]
                        
                        if not trades_ano.empty:
                            resultados[nome_estrategia][ano].append(trades_ano)
            
            except Exception:
                continue
        
        # Resumo do ano
        print()
        for nome in estrategias.keys():
            lista = resultados[nome][ano]
            if lista:
                df_ano = pd.concat(lista, ignore_index=True)
                n = len(df_ano)
                taxa = (df_ano["retorno_%"] > 0).sum() / n
                print(f"  {nome}: {n} trades | {taxa:.1%}")
            else:
                print(f"  {nome}: 0 trades")
    
    # ===================================================================
    # ANÁLISE DETALHADA
    # ===================================================================
    
    print(f"\n{'='*70}")
    print("📊 ANÁLISE COMPLETA")
    print(f"{'='*70}\n")
    
    relatorio = {}
    
    for nome in estrategias.keys():
        print(f"\n{'='*60}")
        print(f"🔬 {nome}")
        print(f"{'='*60}\n")
        
        params = estrategias[nome]["parametros"]
        dias_medio = params["max_dias"]
        
        metricas_por_ano = []
        todos = []
        
        for ano in anos:
            lista = resultados[nome][ano]
            
            if not lista:
                metricas_por_ano.append({
                    "ano": ano,
                    "trades": 0,
                    "taxa": 0,
                    "ret_medio": 0,
                    "ret_anual": 0,
                })
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            todos.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n
            ret_medio = df_ano["retorno_%"].mean()
            ret_anual = ret_medio * (252 / dias_medio)
            
            metricas_por_ano.append({
                "ano": ano,
                "trades": n,
                "taxa": taxa,
                "ret_medio": ret_medio,
                "ret_anual": ret_anual,
            })
        
        # Tabela
        print("📅 DESEMPENHO ANO A ANO:\n")
        
        df_metricas = pd.DataFrame(metricas_por_ano)
        
        print(f"{'Ano':<6} {'Trades':<8} {'Taxa':<8} {'Ret/trade':<12} {'Ret Anual':<12}")
        print(f"{'-'*60}")
        
        for _, row in df_metricas.iterrows():
            ano_str = str(int(row['ano']))
            trades_str = str(int(row['trades']))
            taxa_str = f"{row['taxa']:.1%}" if row['trades'] > 0 else "-"
            ret_str = f"{row['ret_medio']:.2f}%" if row['trades'] > 0 else "-"
            anual_str = f"{row['ret_anual']:.1f}%" if row['trades'] > 0 else "-"
            
            print(f"{ano_str:<6} {trades_str:<8} {taxa_str:<8} {ret_str:<12} {anual_str:<12}")
        
        if not todos:
            print("\n❌ 0 trades. REPROVADO.\n")
            relatorio[nome] = {"aprovado": False}
            continue
        
        # Consolidado
        df_total = pd.concat(todos, ignore_index=True)
        
        n_total = len(df_total)
        ops_ano = n_total / len(anos)
        taxa = (df_total["retorno_%"] > 0).sum() / n_total
        ret_medio = df_total["retorno_%"].mean()
        ret_std = df_total["retorno_%"].std()
        ret_anual = ret_medio * (252 / dias_medio)
        
        sharpe = (ret_medio / ret_std * np.sqrt(252 / dias_medio)) if ret_std > 0 else 0
        
        # Drawdown
        df_total = df_total.sort_values("data_entrada").reset_index(drop=True)
        df_total["equity"] = (1 + df_total["retorno_%"] / 100).cumprod()
        df_total["peak"] = df_total["equity"].cummax()
        df_total["drawdown"] = (df_total["equity"] / df_total["peak"] - 1) * 100
        max_dd = df_total["drawdown"].min()
        
        # Anos
        anos_com = df_metricas[df_metricas["trades"] > 0]
        anos_pos = (anos_com["ret_anual"] > 0).sum()
        anos_neg = (anos_com["ret_anual"] < 0).sum()
        pior = anos_com["ret_anual"].min() if len(anos_com) > 0 else 0
        melhor = anos_com["ret_anual"].max() if len(anos_com) > 0 else 0
        
        print(f"\n📈 CONSOLIDADO:\n")
        print(f"  {n_total} trades | {ops_ano:.1f}/ano")
        print(f"  Taxa: {taxa:.1%} | Sharpe: {sharpe:.2f}")
        print(f"  Ret/trade: {ret_medio:.2f}% | Ret anual: {ret_anual:.1f}%")
        print(f"  Drawdown: {max_dd:.1f}%")
        print(f"  Melhor: {melhor:.1f}% | Pior: {pior:.1f}%")
        print(f"  Anos +/-: {anos_pos}/{anos_neg}")
        
        # Critérios
        print(f"\n✅ CRITÉRIOS:\n")
        
        aprovado = True
        
        # Frequência
        if 20 <= ops_ano <= 30:
            print(f"  ✅ Frequência: {ops_ano:.1f}/ano (meta 20-30)")
        else:
            print(f"  ⚠️  Frequência: {ops_ano:.1f}/ano (fora da meta 20-30)")
        
        # Taxa
        if taxa >= 0.65:
            print(f"  ✅ Taxa {taxa:.1%} >= 65%")
        else:
            print(f"  ❌ Taxa {taxa:.1%} < 65%")
            aprovado = False
        
        # Sharpe
        if sharpe >= 1.0:
            print(f"  ✅ Sharpe {sharpe:.2f} >= 1.0")
        else:
            print(f"  ❌ Sharpe {sharpe:.2f} < 1.0")
            aprovado = False
        
        # Retorno
        if ret_anual >= 30:
            print(f"  ✅ Ret {ret_anual:.1f}% >= 30%")
        else:
            print(f"  ⚠️  Ret {ret_anual:.1f}% < 30%")
        
        # DD
        if max_dd >= -30:
            print(f"  ✅ DD {max_dd:.1f}% >= -30%")
        else:
            print(f"  ❌ DD {max_dd:.1f}% < -30%")
            aprovado = False
        
        # Pior ano
        if pior >= -20:
            print(f"  ✅ Pior {pior:.1f}% >= -20%")
        else:
            print(f"  ❌ Pior {pior:.1f}% < -20%")
            aprovado = False
        
        # Trades
        if n_total >= 20:
            print(f"  ✅ {n_total} trades >= 20")
        else:
            print(f"  ⚠️  Só {n_total} trades")
        
        print(f"\n{'='*60}")
        print(f"{'✅ APROVADO' if aprovado else '❌ REPROVADO'}")
        
        relatorio[nome] = {
            "aprovado": aprovado,
            "trades": n_total,
            "ops_ano": ops_ano,
            "taxa": taxa,
            "sharpe": sharpe,
            "ret_anual": ret_anual,
            "drawdown": max_dd,
        }
    
    # Resumo
    print(f"\n{'='*70}")
    print("🏆 RESUMO FINAL")
    print(f"{'='*70}\n")
    
    for nome, m in relatorio.items():
        if m.get("aprovado"):
            print(f"✅ {nome}")
            print(f"   {m['trades']}t | {m['ops_ano']:.1f}/ano | {m['taxa']:.1%} | Sharpe {m['sharpe']:.2f}")
        else:
            print(f"❌ {nome}")
        print()
    
    print(f"{'='*70}\n")
    
    return relatorio


def validar_swing_profundamente(data_base: date = None):
    """
    VALIDAÇÃO PROFUNDA: SWING SCALPER & POSICIONAL
    
    Estrutura OTIMIZADA: papel → estratégias → anos
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"VALIDAÇÃO PROFUNDA: SWING SCALPER & POSICIONAL")
    print(f"Walk-forward rigoroso: 2022 → 2026 (otimizado)")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    estrategias = {
        "SCALPER": ESTRATEGIA_SWING_SCALPER,
        "POSICIONAL": ESTRATEGIA_SWING_POSICIONAL,
    }
    
    # Estrutura: {estrategia: {ano: [trades]}}
    resultados = {nome: {ano: [] for ano in anos} for nome in estrategias.keys()}
    
    print("🔬 PROCESSANDO PAPÉIS (otimizado)...\n")
    
    # LOOP OTIMIZADO: POR PAPEL
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        try:
            # Baixa dados 1× (até fim do período)
            df = preparar_df(ticker, date(ano_atual, 12, 31))
            
            if df.empty or len(df) < 50:
                print("insuf")
                continue
            
            # Testa cada estratégia com estes dados
            for nome_estrategia, estrategia in estrategias.items():
                # Backtest completo
                trades = backtest_estrategia(ticker, df, estrategia)
                
                if not trades.empty:
                    trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                    
                    # Separa por ano
                    for ano in anos:
                        inicio = pd.Timestamp(date(ano, 1, 1))
                        fim = pd.Timestamp(date(ano, 12, 31))
                        
                        trades_ano = trades[
                            (trades["data_entrada"] >= inicio) &
                            (trades["data_entrada"] <= fim)
                        ]
                        
                        if not trades_ano.empty:
                            resultados[nome_estrategia][ano].append(trades_ano)
            
            print("✓")
        
        except Exception:
            print("ERR")
    
    # ===================================================================
    # ANÁLISE DETALHADA POR ESTRATÉGIA
    # ===================================================================
    
    print(f"\n{'='*70}")
    print("📊 ANÁLISE DETALHADA")
    print(f"{'='*70}\n")
    
    relatorio_final = {}
    
    for nome_estrategia in estrategias.keys():
        print(f"\n{'='*60}")
        print(f"🔬 {nome_estrategia}")
        print(f"{'='*60}\n")
        
        params = estrategias[nome_estrategia]["parametros"]
        dias_medio = params["max_dias"]
        
        # Consolidar por ano
        metricas_por_ano = []
        todos_anos = []
        
        for ano in anos:
            lista = resultados[nome_estrategia][ano]
            
            if not lista:
                metricas_por_ano.append({
                    "ano": ano,
                    "trades": 0,
                    "taxa": 0,
                    "ret_medio": 0,
                    "ret_anualizado": 0,
                })
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            todos_anos.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n
            ret_medio = df_ano["retorno_%"].mean()
            ret_anualizado = ret_medio * (252 / dias_medio)
            
            metricas_por_ano.append({
                "ano": ano,
                "trades": n,
                "taxa": taxa,
                "ret_medio": ret_medio,
                "ret_anualizado": ret_anualizado,
            })
        
        # Tabela ano a ano
        print("📅 DESEMPENHO ANO A ANO:\n")
        
        df_metricas = pd.DataFrame(metricas_por_ano)
        
        print(f"{'Ano':<6} {'Trades':<8} {'Taxa':<8} {'Ret/trade':<12} {'Ret Anual':<12}")
        print(f"{'-'*60}")
        
        for _, row in df_metricas.iterrows():
            ano_str = str(int(row['ano']))
            trades_str = str(int(row['trades']))
            taxa_str = f"{row['taxa']:.1%}" if row['trades'] > 0 else "-"
            ret_trade_str = f"{row['ret_medio']:.2f}%" if row['trades'] > 0 else "-"
            ret_anual_str = f"{row['ret_anualizado']:.1f}%" if row['trades'] > 0 else "-"
            
            print(f"{ano_str:<6} {trades_str:<8} {taxa_str:<8} {ret_trade_str:<12} {ret_anual_str:<12}")
        
        if not todos_anos:
            print("\n❌ Nenhum trade. REPROVADO.\n")
            relatorio_final[nome_estrategia] = {"aprovado": False, "motivo": "0 trades"}
            continue
        
        # Consolidar tudo
        df_total = pd.concat(todos_anos, ignore_index=True)
        
        # === MÉTRICAS CONSOLIDADAS ===
        n_total = len(df_total)
        ops_ano = n_total / len(anos)
        taxa_total = (df_total["retorno_%"] > 0).sum() / n_total
        ret_medio_total = df_total["retorno_%"].mean()
        ret_std = df_total["retorno_%"].std()
        ret_anualizado_total = ret_medio_total * (252 / dias_medio)
        
        sharpe = (ret_medio_total / ret_std * np.sqrt(252 / dias_medio)) if ret_std > 0 else 0
        
        # Drawdown
        df_total = df_total.sort_values("data_entrada").reset_index(drop=True)
        df_total["equity"] = (1 + df_total["retorno_%"] / 100).cumprod()
        df_total["peak"] = df_total["equity"].cummax()
        df_total["drawdown"] = (df_total["equity"] / df_total["peak"] - 1) * 100
        
        max_drawdown = df_total["drawdown"].min()
        
        # Distribuição
        dist_resultado = df_total["resultado"].value_counts()
        pct_alvo = dist_resultado.get("alvo", 0) / n_total * 100
        pct_stop = dist_resultado.get("stop", 0) / n_total * 100
        pct_tempo = dist_resultado.get("tempo", 0) / n_total * 100
        
        # Anos
        anos_com_trades = df_metricas[df_metricas["trades"] > 0]
        anos_positivos = (anos_com_trades["ret_anualizado"] > 0).sum()
        anos_negativos = (anos_com_trades["ret_anualizado"] < 0).sum()
        pior_ano = anos_com_trades["ret_anualizado"].min() if len(anos_com_trades) > 0 else 0
        melhor_ano = anos_com_trades["ret_anualizado"].max() if len(anos_com_trades) > 0 else 0
        
        print(f"\n{'='*60}")
        print("📈 MÉTRICAS CONSOLIDADAS:\n")
        print(f"  Total: {n_total} trades | {ops_ano:.1f}/ano")
        print(f"  Taxa: {taxa_total:.1%}")
        print(f"  Ret/trade: {ret_medio_total:.2f}%")
        print(f"  Ret anual: {ret_anualizado_total:.1f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Drawdown: {max_drawdown:.1f}%")
        print(f"  Melhor ano: {melhor_ano:.1f}%")
        print(f"  Pior ano: {pior_ano:.1f}%")
        print(f"  Anos +/-: {anos_positivos}/{anos_negativos}")
        
        print(f"\n📊 DISTRIBUIÇÃO DE SAÍDAS:\n")
        print(f"  Alvo: {pct_alvo:.1f}% | Stop: {pct_stop:.1f}% | Tempo: {pct_tempo:.1f}%")
        
        # Aprovação
        print(f"\n{'='*60}")
        print("✅ CRITÉRIOS:\n")
        
        aprovado = True
        alertas = []
        
        # Taxa >= 65%
        if taxa_total >= 0.65:
            print(f"  ✅ Taxa {taxa_total:.1%} >= 65%")
        else:
            print(f"  ❌ Taxa {taxa_total:.1%} < 65%")
            alertas.append(f"Taxa {taxa_total:.1%}")
            aprovado = False
        
        # Sharpe >= 1.0
        if sharpe >= 1.0:
            print(f"  ✅ Sharpe {sharpe:.2f} >= 1.0")
        else:
            print(f"  ❌ Sharpe {sharpe:.2f} < 1.0")
            alertas.append(f"Sharpe {sharpe:.2f}")
            aprovado = False
        
        # Ret >= 30%
        if ret_anualizado_total >= 30:
            print(f"  ✅ Ret {ret_anualizado_total:.1f}% >= 30%")
        else:
            print(f"  ⚠️  Ret {ret_anualizado_total:.1f}% < 30%")
        
        # DD >= -30%
        if max_drawdown >= -30:
            print(f"  ✅ DD {max_drawdown:.1f}% >= -30%")
        else:
            print(f"  ❌ DD {max_drawdown:.1f}% < -30%")
            alertas.append(f"DD {max_drawdown:.1f}%")
            aprovado = False
        
        # Pior ano >= -20%
        if pior_ano >= -20:
            print(f"  ✅ Pior {pior_ano:.1f}% >= -20%")
        else:
            print(f"  ❌ Pior {pior_ano:.1f}% < -20%")
            alertas.append(f"Pior {pior_ano:.1f}%")
            aprovado = False
        
        # Anos negativos
        if anos_negativos == 0:
            print(f"  ✅ 0 anos negativos")
        else:
            print(f"  ⚠️  {anos_negativos} ano(s) negativo(s)")
        
        # Trades
        if n_total >= 20:
            print(f"  ✅ {n_total} trades >= 20")
        else:
            print(f"  ⚠️  Só {n_total} trades")
            alertas.append(f"{n_total} trades")
        
        print(f"\n{'='*60}")
        
        if aprovado:
            print(f"✅ {nome_estrategia} APROVADO!")
        else:
            print(f"❌ {nome_estrategia} REPROVADO")
            for a in alertas:
                print(f"   - {a}")
        
        relatorio_final[nome_estrategia] = {
            "aprovado": aprovado,
            "trades": n_total,
            "ops_ano": ops_ano,
            "taxa": taxa_total,
            "ret_anualizado": ret_anualizado_total,
            "sharpe": sharpe,
            "drawdown": max_drawdown,
            "pior_ano": pior_ano,
        }
    
    # Resumo
    print(f"\n{'='*70}")
    print("🏆 RESUMO FINAL")
    print(f"{'='*70}\n")
    
    for nome, m in relatorio_final.items():
        if m.get("aprovado"):
            print(f"✅ {nome}: APROVADO")
            print(f"   {m['trades']}t | {m['ops_ano']:.1f}/ano | {m['taxa']:.1%} | Sharpe {m['sharpe']:.2f}")
        else:
            print(f"❌ {nome}: REPROVADO")
            if "motivo" in m:
                print(f"   {m['motivo']}")
        print()
    
    print(f"{'='*70}\n")
    
    # Salvar
    caminho = f"{CFG['pasta_resultados']}/validacao_swing_{date.today()}.csv"
    pd.DataFrame(relatorio_final).T.to_csv(caminho)
    print(f"[✓] Salvo: {caminho}\n")
    
    return relatorio_final


def testar_swing_completo(data_base: date = None):
    """
    Teste completo de SWING HIGHS/LOWS.
    
    4 Perfis:
    - CONSERVADOR: Swing 5d + confirmações + 4%/2% + 15 dias
    - AGRESSIVO: Swing 3d + alvos maiores 8%/3% + 25 dias
    - SCALPER: Swing 2d + rápido 3%/1.5% + 8 dias
    - POSICIONAL: Swing 10d + longo prazo 12%/5% + 40 dias
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"TESTE COMPLETO: SWING HIGHS/LOWS")
    print(f"4 perfis: Conservador | Agressivo | Scalper | Posicional")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    perfis = {
        "CONSERVADOR (5d, 4%/2%, 15d)": ESTRATEGIA_SWING_CONSERVADOR,
        "AGRESSIVO (3d, 8%/3%, 25d)": ESTRATEGIA_SWING_AGRESSIVO,
        "SCALPER (2d, 3%/1.5%, 8d)": ESTRATEGIA_SWING_SCALPER,
        "POSICIONAL (10d, 12%/5%, 40d)": ESTRATEGIA_SWING_POSICIONAL,
    }
    
    resultados = {nome: {ano: [] for ano in anos} for nome in perfis.keys()}
    
    print("🔬 PROCESSANDO PAPÉIS...\n")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        try:
            df = preparar_df(ticker, date(ano_atual, 12, 31))
            
            if df.empty or len(df) < 50:
                print("insuf")
                continue
            
            for nome_perfil, estrategia in perfis.items():
                trades = backtest_estrategia(ticker, df, estrategia)
                
                if not trades.empty:
                    trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                    
                    for ano in anos:
                        inicio = pd.Timestamp(date(ano, 1, 1))
                        fim = pd.Timestamp(date(ano, 12, 31))
                        
                        trades_ano = trades[
                            (trades["data_entrada"] >= inicio) &
                            (trades["data_entrada"] <= fim)
                        ]
                        
                        if not trades_ano.empty:
                            resultados[nome_perfil][ano].append(trades_ano)
            
            print("✓")
        except Exception as e:
            print(f"ERR")
    
    # Análise detalhada
    print(f"\n{'='*70}")
    print("📊 RESULTADOS DETALHADOS (ano a ano)")
    print(f"{'='*70}\n")
    
    metricas_finais = {}
    
    for nome in perfis.keys():
        print(f"\n{'='*60}")
        print(f"{nome}")
        print(f"{'='*60}")
        
        todos = []
        
        for ano in anos:
            lista = resultados[nome][ano]
            
            if not lista:
                print(f"  Ano {ano}: 0 trades")
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            todos.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n if n > 0 else 0
            ret = df_ano["retorno_%"].mean()
            
            print(f"  Ano {ano}: {n} trades | Taxa {taxa:.1%} | Ret médio {ret:.2f}%")
        
        if todos:
            df_total = pd.concat(todos, ignore_index=True)
            n_total = len(df_total)
            taxa_media = (df_total["retorno_%"] > 0).sum() / n_total
            ret_medio = df_total["retorno_%"].mean()
            ret_std = df_total["retorno_%"].std()
            
            # Ajustar Sharpe por dias médios
            params = perfis[nome]["parametros"]
            dias_medio = params["max_dias"]
            sharpe = (ret_medio / ret_std * np.sqrt(252/dias_medio)) if ret_std > 0 else 0
            
            metricas_finais[nome] = {
                "total": n_total,
                "ops_ano": round(n_total / len(anos), 1),
                "taxa": taxa_media,
                "retorno": ret_medio,
                "sharpe": sharpe,
            }
            
            print(f"  {'='*55}")
            print(f"  CONSOLIDADO: {n_total} trades | {metricas_finais[nome]['ops_ano']}/ano | Taxa {taxa_media:.1%} | Sharpe {sharpe:.2f}")
        else:
            metricas_finais[nome] = {"total": 0, "ops_ano": 0, "taxa": 0, "retorno": 0, "sharpe": 0}
            print(f"  Nenhum trade")
    
    # Resumo
    print(f"\n{'='*70}")
    print("🏆 COMPARAÇÃO FINAL")
    print(f"{'='*70}\n")
    
    for nome, m in metricas_finais.items():
        print(f"{nome}")
        print(f"  {m['total']}t | Taxa {m['taxa']:.1%} | Ret {m['retorno']:.2f}% | Sharpe {m['sharpe']:.2f}")
        print()
    
    # Veredicto
    melhor_nome = max(metricas_finais.keys(), key=lambda k: metricas_finais[k]["sharpe"])
    melhor = metricas_finais[melhor_nome]
    
    print(f"{'='*70}")
    
    if melhor['sharpe'] >= 1.0 and melhor['taxa'] >= 0.65:
        print("✅ SWING APROVADO!")
        print(f"   Usar: {melhor_nome}")
        print(f"   {melhor['total']} trades | Taxa {melhor['taxa']:.1%} | Sharpe {melhor['sharpe']:.2f}")
    else:
        print("❌ SWING REPROVADO")
        print(f"   Melhor: {melhor_nome}")
        print(f"   Sharpe {melhor['sharpe']:.2f} (precisa ≥1.0)")
        print(f"   Taxa {melhor['taxa']:.1%} (precisa ≥65%)")
        print()
        print("   💡 Estratégias técnicas testadas:")
        print("      ❌ Momentum")
        print("      ❌ Breakout")
        print("      ❌ Pullback")
        print("      ❌ Swing")
        print("      ✅ Mean Reversion LIGHT (única aprovada)")
    
    print(f"{'='*70}\n")
    
    return metricas_finais


def testar_pullback_perfis_finais(data_base: date = None):
    """
    TESTE FINAL: 3 perfis diferentes de Pullback
    
    PERFIL A: SEM REGIME + Alvos maiores (6%, 8%, 10%)
    PERFIL B: REGIME MM50 (mais flexível) + Alvo 5%
    PERFIL C: QUALIDADE (ADX forte + Volume alto) + Alvo 7%
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"TESTE FINAL: 3 PERFIS DE PULLBACK")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # === PERFIL A: SEM REGIME, ALVOS VARIADOS ===
    print("📊 PERFIL A: SEM REGIME + Alvos maiores\n")
    
    perfil_a_configs = []
    
    for alvo_pct in [0.06, 0.08, 0.10]:
        stop_pct = alvo_pct / 2  # Ratio 2:1
        
        estrategia_a = {
            "nome": f"pullback_a_{int(alvo_pct*100)}",
            "parametros": {
                "close_acima_mm20": True,
                "queda_5d_min": -0.05,
                "queda_5d_max": -0.01,
                "rsi_min": 20,
                "rsi_max": 60,
                "vol_relativo_min": 0.3,
                # SEM regime
                "alvo_pct": alvo_pct,
                "stop_pct": stop_pct,
                "max_dias": 20,
            },
            "fn_sinal": lambda row, p: (
                _val(row, "close") > _val(row, "mm20", 0)
                and _retorno_nd(row, 5) >= p["queda_5d_min"]
                and _retorno_nd(row, 5) <= p["queda_5d_max"]
                and _val(row, "rsi") >= p["rsi_min"]
                and _val(row, "rsi") <= p["rsi_max"]
                and _val(row, "vol_relativo") >= p["vol_relativo_min"]
            ),
        }
        
        resultados_ano = {ano: [] for ano in anos}
        
        for ticker in tickers:
            try:
                df = preparar_df(ticker, date(ano_atual, 12, 31))
                if df.empty:
                    continue
                
                trades = backtest_estrategia(ticker, df, estrategia_a)
                if trades.empty:
                    continue
                
                trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                
                for ano in anos:
                    inicio = pd.Timestamp(date(ano, 1, 1))
                    fim = pd.Timestamp(date(ano, 12, 31))
                    
                    trades_ano = trades[
                        (trades["data_entrada"] >= inicio) &
                        (trades["data_entrada"] <= fim)
                    ]
                    
                    if not trades_ano.empty:
                        resultados_ano[ano].append(trades_ano)
            except:
                continue
        
        # Consolidar
        todos = []
        for ano in anos:
            if resultados_ano[ano]:
                todos.append(pd.concat(resultados_ano[ano], ignore_index=True))
        
        if todos:
            df_total = pd.concat(todos, ignore_index=True)
            n = len(df_total)
            taxa = (df_total["retorno_%"] > 0).sum() / n
            ret = df_total["retorno_%"].mean()
            std = df_total["retorno_%"].std()
            sharpe = (ret / std * np.sqrt(252/20)) if std > 0 else 0
            
            perfil_a_configs.append({
                "config": f"Alvo {int(alvo_pct*100)}% / Stop {int(stop_pct*100)}%",
                "trades": n,
                "taxa": taxa,
                "ret": ret,
                "sharpe": sharpe,
            })
            
            print(f"  {int(alvo_pct*100)}%/{int(stop_pct*100)}%: {n}t | {taxa:.1%} | Ret {ret:.2f}% | Sharpe {sharpe:.2f}")
        else:
            print(f"  {int(alvo_pct*100)}%/{int(stop_pct*100)}%: 0 trades")
    
    # === PERFIL B: REGIME MM50 ===
    print(f"\n📊 PERFIL B: REGIME MM50 (mais flexível)\n")
    
    estrategia_b = {
        "nome": "pullback_b_mm50",
        "parametros": {
            "close_acima_mm20": True,
            "queda_5d_min": -0.05,
            "queda_5d_max": -0.01,
            "rsi_min": 20,
            "rsi_max": 60,
            "vol_relativo_min": 0.3,
            "ibov_acima_mm50": True,  # MM50 em vez de MM200
            "alvo_pct": 0.05,
            "stop_pct": 0.03,
            "max_dias": 20,
        },
        "fn_sinal": lambda row, p: (
            _val(row, "close") > _val(row, "mm20", 0)
            and _retorno_nd(row, 5) >= p["queda_5d_min"]
            and _retorno_nd(row, 5) <= p["queda_5d_max"]
            and _val(row, "rsi") >= p["rsi_min"]
            and _val(row, "rsi") <= p["rsi_max"]
            and _val(row, "vol_relativo") >= p["vol_relativo_min"]
            and _val(row, "close_ibov") > _val(row, "mm50_ibov", 0)  # MM50
        ),
    }
    
    resultados_b = {ano: [] for ano in anos}
    
    for ticker in tickers:
        try:
            df = preparar_df(ticker, date(ano_atual, 12, 31))
            if df.empty:
                continue
            
            trades = backtest_estrategia(ticker, df, estrategia_b)
            if trades.empty:
                continue
            
            trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
            
            for ano in anos:
                inicio = pd.Timestamp(date(ano, 1, 1))
                fim = pd.Timestamp(date(ano, 12, 31))
                
                trades_ano = trades[
                    (trades["data_entrada"] >= inicio) &
                    (trades["data_entrada"] <= fim)
                ]
                
                if not trades_ano.empty:
                    resultados_b[ano].append(trades_ano)
        except:
            continue
    
    # Consolidar B
    todos_b = []
    for ano in anos:
        if resultados_b[ano]:
            df_ano = pd.concat(resultados_b[ano], ignore_index=True)
            todos_b.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n
            ret = df_ano["retorno_%"].mean()
            print(f"  Ano {ano}: {n}t | {taxa:.1%} | {ret:.2f}%")
    
    if todos_b:
        df_total_b = pd.concat(todos_b, ignore_index=True)
        n_b = len(df_total_b)
        taxa_b = (df_total_b["retorno_%"] > 0).sum() / n_b
        ret_b = df_total_b["retorno_%"].mean()
        std_b = df_total_b["retorno_%"].std()
        sharpe_b = (ret_b / std_b * np.sqrt(252/20)) if std_b > 0 else 0
        
        print(f"  CONSOLIDADO: {n_b}t | {taxa_b:.1%} | Ret {ret_b:.2f}% | Sharpe {sharpe_b:.2f}")
    else:
        n_b, taxa_b, ret_b, sharpe_b = 0, 0, 0, 0
        print(f"  0 trades")
    
    # === PERFIL C: QUALIDADE (ADX + VOLUME) ===
    print(f"\n📊 PERFIL C: QUALIDADE (ADX>25 + Volume>1.5×)\n")
    
    estrategia_c = {
        "nome": "pullback_c_qualidade",
        "parametros": {
            "close_acima_mm20": True,
            "queda_5d_min": -0.05,
            "queda_5d_max": -0.01,
            "rsi_min": 25,
            "rsi_max": 45,  # Mais restritivo (oversold real)
            "vol_relativo_min": 1.5,  # Volume forte
            "adx_min": 25,  # Tendência forte
            # SEM regime
            "alvo_pct": 0.07,
            "stop_pct": 0.04,
            "max_dias": 20,
        },
        "fn_sinal": lambda row, p: (
            _val(row, "close") > _val(row, "mm20", 0)
            and _retorno_nd(row, 5) >= p["queda_5d_min"]
            and _retorno_nd(row, 5) <= p["queda_5d_max"]
            and _val(row, "rsi") >= p["rsi_min"]
            and _val(row, "rsi") <= p["rsi_max"]
            and _val(row, "vol_relativo") >= p["vol_relativo_min"]
            and _val(row, "ADX_14") >= p["adx_min"]
        ),
    }
    
    resultados_c = {ano: [] for ano in anos}
    
    for ticker in tickers:
        try:
            df = preparar_df(ticker, date(ano_atual, 12, 31))
            if df.empty:
                continue
            
            trades = backtest_estrategia(ticker, df, estrategia_c)
            if trades.empty:
                continue
            
            trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
            
            for ano in anos:
                inicio = pd.Timestamp(date(ano, 1, 1))
                fim = pd.Timestamp(date(ano, 12, 31))
                
                trades_ano = trades[
                    (trades["data_entrada"] >= inicio) &
                    (trades["data_entrada"] <= fim)
                ]
                
                if not trades_ano.empty:
                    resultados_c[ano].append(trades_ano)
        except:
            continue
    
    # Consolidar C
    todos_c = []
    for ano in anos:
        if resultados_c[ano]:
            df_ano = pd.concat(resultados_c[ano], ignore_index=True)
            todos_c.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n
            ret = df_ano["retorno_%"].mean()
            print(f"  Ano {ano}: {n}t | {taxa:.1%} | {ret:.2f}%")
    
    if todos_c:
        df_total_c = pd.concat(todos_c, ignore_index=True)
        n_c = len(df_total_c)
        taxa_c = (df_total_c["retorno_%"] > 0).sum() / n_c
        ret_c = df_total_c["retorno_%"].mean()
        std_c = df_total_c["retorno_%"].std()
        sharpe_c = (ret_c / std_c * np.sqrt(252/20)) if std_c > 0 else 0
        
        print(f"  CONSOLIDADO: {n_c}t | {taxa_c:.1%} | Ret {ret_c:.2f}% | Sharpe {sharpe_c:.2f}")
    else:
        n_c, taxa_c, ret_c, sharpe_c = 0, 0, 0, 0
        print(f"  0 trades")
    
    # === RESUMO FINAL ===
    print(f"\n{'='*70}")
    print("🏆 COMPARAÇÃO FINAL")
    print(f"{'='*70}\n")
    
    if perfil_a_configs:
        melhor_a = max(perfil_a_configs, key=lambda x: x["sharpe"])
        print(f"PERFIL A (melhor): {melhor_a['config']}")
        print(f"  {melhor_a['trades']}t | {melhor_a['taxa']:.1%} | Ret {melhor_a['ret']:.2f}% | Sharpe {melhor_a['sharpe']:.2f}")
    else:
        print(f"PERFIL A: 0 trades")
        melhor_a = {"sharpe": 0, "taxa": 0}
    
    print()
    print(f"PERFIL B (MM50): {n_b}t | {taxa_b:.1%} | Ret {ret_b:.2f}% | Sharpe {sharpe_b:.2f}")
    print()
    print(f"PERFIL C (Qualidade): {n_c}t | {taxa_c:.1%} | Ret {ret_c:.2f}% | Sharpe {sharpe_c:.2f}")
    
    print(f"\n{'='*70}")
    
    # Veredicto
    melhor_sharpe = max(melhor_a["sharpe"], sharpe_b, sharpe_c)
    melhor_taxa = max(melhor_a["taxa"], taxa_b, taxa_c)
    
    if melhor_sharpe >= 1.0 and melhor_taxa >= 0.65:
        print("✅ PULLBACK APROVADO!")
        if melhor_sharpe == melhor_a["sharpe"]:
            print(f"   Usar: PERFIL A - {melhor_a['config']}")
        elif melhor_sharpe == sharpe_b:
            print(f"   Usar: PERFIL B - Regime MM50")
        else:
            print(f"   Usar: PERFIL C - Qualidade")
    else:
        print("❌ PULLBACK REPROVADO")
        print(f"   Melhor Sharpe: {melhor_sharpe:.2f} (precisa ≥1.0)")
        print(f"   Melhor Taxa: {melhor_taxa:.1%} (precisa ≥65%)")
        print()
        print("   💡 CONCLUSÃO: Pullback não funciona na B3")
        print("      Partir para próxima estratégia ou aceitar Mean Reversion")
    
    print(f"{'='*70}\n")
    
    return {
        "perfil_a": perfil_a_configs,
        "perfil_b": {"trades": n_b, "taxa": taxa_b, "sharpe": sharpe_b},
        "perfil_c": {"trades": n_c, "taxa": taxa_c, "sharpe": sharpe_c},
    }


def testar_pullback_completo(data_base: date = None):
    """
    Teste DIAGNÓSTICO de Pullback:
    Adiciona filtros incrementalmente para ver qual bloqueia sinais.
    
    Versões:
    1. BÁSICO - só close > MM20 + correção
    2. + RSI - adiciona RSI
    3. + VOLUME - adiciona volume
    4. + REGIME - adiciona filtro Ibov > MM200
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO: PULLBACK EM TENDÊNCIA")
    print(f"Adicionando filtros incrementalmente")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Versões em ordem crescente de complexidade
    versoes = {
        "0. ULTRA SIMPLES (só MM20)": ESTRATEGIA_PULLBACK_ULTRA_SIMPLES,
        "1. BÁSICO (MM20 + correção)": ESTRATEGIA_PULLBACK_BASICO,
        "2. + RSI": ESTRATEGIA_PULLBACK_COM_RSI,
        "3. + VOLUME": ESTRATEGIA_PULLBACK_COM_VOLUME,
        "4. + REGIME": ESTRATEGIA_PULLBACK_COM_REGIME,
    }
    
    resultados = {nome: {ano: [] for ano in anos} for nome in versoes.keys()}
    
    print("🔬 PROCESSANDO PAPÉIS...\n")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        try:
            df = preparar_df(ticker, date(ano_atual, 12, 31))
            
            if df.empty or len(df) < 30:
                print("insuf")
                continue
            
            for nome_versao, estrategia in versoes.items():
                trades = backtest_estrategia(ticker, df, estrategia)
                
                if not trades.empty:
                    trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                    
                    for ano in anos:
                        inicio = pd.Timestamp(date(ano, 1, 1))
                        fim = pd.Timestamp(date(ano, 12, 31))
                        
                        trades_ano = trades[
                            (trades["data_entrada"] >= inicio) &
                            (trades["data_entrada"] <= fim)
                        ]
                        
                        if not trades_ano.empty:
                            resultados[nome_versao][ano].append(trades_ano)
            
            print("✓")
        except Exception as e:
            print(f"ERR")
    
    # Análise detalhada
    print(f"\n{'='*70}")
    print("📊 RESULTADOS DETALHADOS (ano a ano)")
    print(f"{'='*70}\n")
    
    metricas_finais = {}
    
    for nome in versoes.keys():
        print(f"\n{'='*60}")
        print(f"{nome}")
        print(f"{'='*60}")
        
        todos = []
        
        for ano in anos:
            lista = resultados[nome][ano]
            
            if not lista:
                print(f"  Ano {ano}: 0 trades")
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            todos.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n if n > 0 else 0
            ret = df_ano["retorno_%"].mean()
            
            print(f"  Ano {ano}: {n} trades | Taxa {taxa:.1%} | Ret médio {ret:.2f}%")
        
        if todos:
            df_total = pd.concat(todos, ignore_index=True)
            n_total = len(df_total)
            taxa_media = (df_total["retorno_%"] > 0).sum() / n_total
            ret_medio = df_total["retorno_%"].mean()
            ret_std = df_total["retorno_%"].std()
            sharpe = (ret_medio / ret_std * np.sqrt(252/15)) if ret_std > 0 else 0
            
            metricas_finais[nome] = {
                "total": n_total,
                "ops_ano": round(n_total / len(anos), 1),
                "taxa": taxa_media,
                "retorno": ret_medio,
                "sharpe": sharpe,
            }
            
            print(f"  {'='*55}")
            print(f"  CONSOLIDADO: {n_total} trades | {metricas_finais[nome]['ops_ano']}/ano | Taxa {taxa_media:.1%} | Sharpe {sharpe:.2f}")
        else:
            metricas_finais[nome] = {"total": 0, "ops_ano": 0, "taxa": 0, "retorno": 0, "sharpe": 0}
            print(f"  Nenhum trade em nenhum ano")
    
    # Resumo
    print(f"\n{'='*70}")
    print("📊 RESUMO")
    print(f"{'='*70}\n")
    
    ultra = metricas_finais.get("0. ULTRA SIMPLES (só MM20)", {}).get("total", 0)
    basico = metricas_finais.get("1. BÁSICO (MM20 + correção)", {}).get("total", 0)
    rsi = metricas_finais.get("2. + RSI", {}).get("total", 0)
    volume = metricas_finais.get("3. + VOLUME", {}).get("total", 0)
    regime = metricas_finais.get("4. + REGIME", {}).get("total", 0)
    
    if ultra == 0:
        print("❌ PROBLEMA CRÍTICO:")
        print("   Nem close > MM20 gera sinais!")
        print("   Verificar preparar_df ou backtest_estrategia.")
    
    elif basico == 0:
        print(f"✓ ULTRA SIMPLES funciona: {ultra} trades")
        print(f"❌ Adicionar correção zerou os sinais")
        print()
        print("   BUG na condição de correção (_retorno_nd).")
    
    elif rsi == 0:
        print(f"✓ BÁSICO funciona: {basico} trades")
        print(f"❌ Adicionar RSI zerou os sinais")
        print()
        print("   Filtro RSI é muito restritivo.")
    
    elif volume == 0:
        print(f"✓ BÁSICO + RSI funciona: {rsi} trades")
        print(f"❌ Adicionar VOLUME zerou os sinais")
        print()
        print("   Filtro de volume é muito restritivo.")
    
    elif regime == 0:
        print(f"✓ BÁSICO + RSI + VOLUME funciona: {volume} trades")
        print(f"❌ Adicionar REGIME zerou os sinais")
        print()
        print("   Filtro Ibov > MM200 é muito restritivo.")
    
    else:
        print(f"✓ TODAS as versões geraram sinais!")
        print()
        print("   Impacto de cada filtro:")
        print(f"   BÁSICO: {basico} trades")
        if basico > 0:
            print(f"   + RSI: {rsi} trades (-{basico - rsi}, {(rsi/basico)*100:.0f}% restante)")
        if rsi > 0:
            print(f"   + VOLUME: {volume} trades (-{rsi - volume}, {(volume/rsi)*100:.0f}% restante)")
        if volume > 0:
            print(f"   + REGIME: {regime} trades (-{volume - regime}, {(regime/volume)*100:.0f}% restante)")
        print()
        
        # Avaliar melhor versão
        melhor_nome = max(metricas_finais.keys(), key=lambda k: metricas_finais[k]["sharpe"])
        melhor = metricas_finais[melhor_nome]
        
        print(f"✅ MELHOR: {melhor_nome}")
        print(f"   {melhor['total']} trades | Taxa {melhor['taxa']:.1%} | Sharpe {melhor['sharpe']:.2f}")
        print()
        
        if melhor['taxa'] >= 0.65 and melhor['sharpe'] >= 1.0:
            print("✅ PULLBACK APROVADO!")
        else:
            print("❌ PULLBACK REPROVADO")
            if melhor['taxa'] < 0.65:
                print(f"   Taxa {melhor['taxa']:.1%} < 65%")
            if melhor['sharpe'] < 1.0:
                print(f"   Sharpe {melhor['sharpe']:.2f} < 1.0")
    
    print(f"\n{'='*70}\n")
    
    return metricas_finais


def diagnosticar_breakout_incremental(data_base: date = None):
    """
    Diagnóstico INCREMENTAL com estrutura OTIMIZADA.
    
    Estrutura correta:
    - Loop por PAPEL primeiro (baixa 1× por papel)
    - Para cada papel: testa todas estratégias
    - Separa trades por ano no final
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    estrategias = {
        "1. BÁSICO (só rompimento)": ESTRATEGIA_BREAKOUT_BASICO,
        "2. + VOLUME": ESTRATEGIA_BREAKOUT_VOLUME,
        "3. + VOLUME + ADX": ESTRATEGIA_BREAKOUT_VOLUME_ADX,
        "4. + VOLUME + ADX + MM50": ESTRATEGIA_BREAKOUT_COMPLETO_SEM_REGIME,
    }
    
    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO INCREMENTAL: BREAKOUT")
    print(f"Estrutura otimizada: papel → estratégias → anos")
    print(f"{len(tickers)} papéis × {len(estrategias)} estratégias × {len(anos)} anos")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    todos_resultados = {versao: {ano: [] for ano in anos} for versao in estrategias.keys()}
    
    print("🔬 PROCESSANDO PAPÉIS...\n")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        try:
            df = preparar_df(ticker, date(ano_atual, 12, 31))
            
            if df.empty or len(df) < 30:
                print("insuficiente")
                continue
            
            for versao, estrategia in estrategias.items():
                trades_todos = backtest_estrategia(ticker, df, estrategia)
                
                if not trades_todos.empty:
                    trades_todos["data_entrada"] = pd.to_datetime(trades_todos["data_entrada"])
                    
                    for ano in anos:
                        inicio = pd.Timestamp(date(ano, 1, 1))
                        fim = pd.Timestamp(date(ano, 12, 31))
                        
                        trades_ano = trades_todos[
                            (trades_todos["data_entrada"] >= inicio) &
                            (trades_todos["data_entrada"] <= fim)
                        ]
                        
                        if not trades_ano.empty:
                            todos_resultados[versao][ano].append(trades_ano)
            
            print("✓")
        
        except Exception as e:
            print(f"ERRO: {e}")
    
    print(f"\n{'='*70}")
    print("📊 RESULTADOS")
    print(f"{'='*70}\n")
    
    resultados_por_versao = {}
    
    for versao in estrategias.keys():
        print(f"\n{'='*60}")
        print(f"{versao}")
        print(f"{'='*60}")
        
        trades_completa = []
        
        for ano in anos:
            lista = todos_resultados[versao][ano]
            
            if not lista:
                print(f"  Ano {ano}: 0 trades")
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            trades_completa.append(df_ano)
            
            n = len(df_ano)
            ganhos = (df_ano["retorno_%"] > 0).sum()
            taxa = ganhos / n if n > 0 else 0
            
            print(f"  Ano {ano}: {n} trades | {taxa:.1%}")
        
        if trades_completa:
            df_total = pd.concat(trades_completa, ignore_index=True)
            n_total = len(df_total)
            ganhos_total = (df_total["retorno_%"] > 0).sum()
            taxa_media = ganhos_total / n_total if n_total > 0 else 0
            ret_medio = df_total["retorno_%"].mean()
            
            resultados_por_versao[versao] = {
                "total_trades": n_total,
                "taxa_acerto": taxa_media,
                "retorno_medio": ret_medio,
                "ops_ano": round(n_total / len(anos), 1),
            }
        else:
            resultados_por_versao[versao] = {
                "total_trades": 0,
                "taxa_acerto": 0,
                "retorno_medio": 0,
                "ops_ano": 0,
            }
    
    print(f"\n{'='*70}")
    print("📊 RESUMO")
    print(f"{'='*70}\n")
    
    for versao, m in resultados_por_versao.items():
        print(f"{versao}")
        print(f"  Total: {m['total_trades']} | Ops/ano: {m['ops_ano']} | Taxa: {m['taxa_acerto']:.1%} | Ret: {m['retorno_medio']:.2f}%")
        print()
    
    basico = resultados_por_versao.get("1. BÁSICO (só rompimento)", {}).get("total_trades", 0)
    
    if basico == 0:
        print("❌ BREAKOUT não funciona na B3")
    else:
        print(f"✓ BREAKOUT gera {basico} trades")
    
    print(f"\n{'='*70}\n")
    
    return resultados_por_versao



def comparar_breakout_regimes(data_base: date = None):
    """
    Compara 4 filtros de regime para Breakout:
    - MM200 (longo prazo, ~1 ano)
    - MM100 (médio prazo, ~5 meses)
    - MM50 (curto prazo, ~2 meses)
    - SEM REGIME (opera sempre)
    
    Versão otimizada com cache de dados.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    # Estratégias a comparar
    estrategias = {
        "MM200": ESTRATEGIA_BREAKOUT_REGIME_MM200,
        "MM100": ESTRATEGIA_BREAKOUT_REGIME_MM100,
        "MM50": ESTRATEGIA_BREAKOUT_REGIME_MM50,
        "SEM REGIME": ESTRATEGIA_BREAKOUT_SEM_REGIME,
    }
    
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO: BREAKOUT - TESTE DE FILTROS DE REGIME")
    print(f"Testando {len(tickers)} papéis em {len(anos)} anos")
    print(f"Versões: MM200 | MM100 | MM50 | SEM REGIME")
    print(f"⚡ OTIMIZAÇÃO: Baixa cada papel 1× e reutiliza")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # === FASE 1: BAIXAR DADOS ===
    print("📥 FASE 1/2: Baixando dados (1× por papel)...\n")
    
    cache_dados = {}
    data_final = date(ano_atual, 12, 31)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            df = preparar_df(ticker, data_final)
            cache_dados[ticker] = df
            print("✓")
        except Exception as e:
            print(f"ERRO: {e}")
            continue
    
    print(f"\n✓ {len(cache_dados)} papéis carregados em cache\n")
    
    # === FASE 2: TESTAR ESTRATÉGIAS ===
    print("🧪 FASE 2/2: Testando filtros de regime...\n")
    
    resultados_por_versao = {v: {} for v in estrategias.keys()}
    
    for versao, estrategia in estrategias.items():
        print(f"\n{'='*60}")
        print(f"TESTANDO: {versao}")
        print(f"{'='*60}")
        
        for ano in anos:
            print(f"  Ano {ano}:", end=" ", flush=True)
            
            try:
                inicio_ano = date(ano, 1, 1)
                fim_ano = date(ano, 12, 31)
                
                todos_trades = []
                
                for ticker, df_completo in cache_dados.items():
                    try:
                        df_ate_ano = df_completo[df_completo.index <= pd.Timestamp(fim_ano)]
                        
                        if df_ate_ano.empty:
                            continue
                        
                        trades = backtest_estrategia(ticker, df_ate_ano, estrategia)
                        
                        if trades.empty:
                            continue
                        
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        trades_ano = trades[
                            (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                            (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                        ]
                        
                        if not trades_ano.empty:
                            todos_trades.append(trades_ano)
                    except Exception:
                        continue
                
                if not todos_trades:
                    print("sem trades")
                    continue
                
                df_todos = pd.concat(todos_trades, ignore_index=True)
                
                # Métricas
                n_total = len(df_todos)
                n_ganhos = (df_todos["retorno_%"] > 0).sum()
                taxa_acerto = (n_ganhos / n_total) if n_total > 0 else 0
                retorno_med = df_todos["retorno_%"].mean()
                
                dist = df_todos["resultado"].value_counts()
                pct_alvo = dist.get("alvo", 0) / n_total if n_total > 0 else 0
                pct_stop = dist.get("stop", 0) / n_total if n_total > 0 else 0
                
                dias_med = (pct_alvo * 12) + (pct_stop * 7) + ((1 - pct_alvo - pct_stop) * 20)
                ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
                
                resultados_por_versao[versao][ano] = {
                    "n_trades": n_total,
                    "taxa_acerto": round(taxa_acerto, 3),
                    "retorno_med_%": round(retorno_med, 2),
                    "ret_anualizado_%": round(ret_anualizado, 1),
                }
                
                print(f"{n_total}t | {taxa_acerto:.1%} | {ret_anualizado:.0f}%/ano")
            
            except Exception as e:
                print(f"ERRO: {e}")
    
    # === COMPARAÇÃO CONSOLIDADA ===
    print(f"\n{'='*70}")
    print("📊 COMPARAÇÃO CONSOLIDADA")
    print(f"{'='*70}\n")
    
    resumo = []
    
    for versao in estrategias.keys():
        anos_data = resultados_por_versao[versao]
        
        if not anos_data:
            # Versão sem trades
            resumo.append({
                "regime": versao,
                "ops_ano": 0,
                "taxa_%": 0,
                "ret_ano_%": 0,
                "sharpe": 0,
                "melhor_%": 0,
                "pior_%": 0,
                "anos_pos/neg": "0/0",
            })
            continue
        
        taxas = [anos_data[a]["taxa_acerto"] for a in anos if a in anos_data]
        rets = [anos_data[a]["ret_anualizado_%"] for a in anos if a in anos_data]
        n_trades_total = sum([anos_data[a]["n_trades"] for a in anos if a in anos_data])
        
        if not taxas:
            continue
        
        taxa_media = np.mean(taxas)
        ret_medio = np.mean(rets)
        ret_std = np.std(rets)
        pior_ano = min(rets)
        melhor_ano = max(rets)
        sharpe = ret_medio / ret_std if ret_std > 0 else 0
        anos_positivos = sum([1 for r in rets if r > 0])
        anos_negativos = sum([1 for r in rets if r < 0])
        
        resumo.append({
            "regime": versao,
            "ops_ano": round(n_trades_total / len(taxas), 1),
            "taxa_%": round(taxa_media * 100, 1),
            "ret_ano_%": round(ret_medio, 1),
            "sharpe": round(sharpe, 2),
            "melhor_%": round(melhor_ano, 1),
            "pior_%": round(pior_ano, 1),
            "anos_pos/neg": f"{anos_positivos}/{anos_negativos}",
        })
    
    df_resumo = pd.DataFrame(resumo)
    df_resumo = df_resumo.sort_values("sharpe", ascending=False)
    
    print(df_resumo.to_string(index=False))
    
    # === ANÁLISE ===
    print(f"\n{'='*70}")
    print("💡 ANÁLISE DO TRADE-OFF")
    print(f"{'='*70}\n")
    
    # Encontrar versões com trades
    versoes_com_trades = df_resumo[df_resumo["ops_ano"] > 0]
    
    if len(versoes_com_trades) == 0:
        print("❌ NENHUMA VERSÃO GEROU TRADES")
        print("\n🔍 DIAGNÓSTICO:")
        print("   Mesmo relaxando o filtro de regime (MM100, MM50, SEM),")
        print("   Breakout não gerou sinais suficientes.")
        print()
        print("   POSSÍVEL CAUSA:")
        print("   - Condição 'rompe máxima de 20d' é muito específica")
        print("   - Volume > 1.0× também pode estar filtrando demais")
        print("   - ADX > 10 pode ser restritivo")
        print()
        print("   💡 PRÓXIMOS PASSOS:")
        print("   1. Testar com período menor (máxima de 10 dias)")
        print("   2. Remover filtro de volume")
        print("   3. Ou aceitar que Breakout não funciona na B3")
        return None
    
    melhor = versoes_com_trades.iloc[0]
    
    print(f"🏆 MELHOR FILTRO: {melhor['regime']}")
    print(f"   Ops/ano: {melhor['ops_ano']:.1f}")
    print(f"   Taxa: {melhor['taxa_%']:.1f}%")
    print(f"   Retorno: {melhor['ret_ano_%']:.1f}%/ano")
    print(f"   Sharpe: {melhor['sharpe']:.2f}")
    print(f"   Pior ano: {melhor['pior_%']:.1f}%")
    print(f"   Anos pos/neg: {melhor['anos_pos/neg']}")
    print()
    
    # Critérios
    aprovado = True
    alertas = []
    
    if melhor['taxa_%'] < 65:
        alertas.append(f"Taxa {melhor['taxa_%']:.1f}% < 65%")
        aprovado = False
    
    if melhor['pior_%'] < -20:
        alertas.append(f"Pior ano {melhor['pior_%']:.1f}% < -20%")
        aprovado = False
    
    if melhor['ret_ano_%'] < 30:
        alertas.append(f"Retorno {melhor['ret_ano_%']:.1f}% < 30%")
        aprovado = False
    
    if melhor['sharpe'] < 1.0:
        alertas.append(f"Sharpe {melhor['sharpe']:.2f} < 1.0")
        aprovado = False
    
    if aprovado:
        print("✅ BREAKOUT APROVADO!")
        print(f"   Use filtro {melhor['regime']} no portfólio.")
    else:
        print("❌ BREAKOUT REPROVADO")
        print("   Motivos:")
        for alerta in alertas:
            print(f"     - {alerta}")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar
    caminho = f"{CFG['pasta_resultados']}/breakout_regimes_{data_base}.csv"
    df_resumo.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_resumo


def comparar_breakout_light_vs_super(data_base: date = None):
    """
    Compara BREAKOUT LIGHT vs SUPER-LIGHT lado a lado.
    
    Versão otimizada: baixa dados 1× e reutiliza.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    # Estratégias a comparar
    estrategias = {
        "LIGHT": ESTRATEGIA_BREAKOUT_LIGHT,
        "SUPER-LIGHT": ESTRATEGIA_BREAKOUT_SUPER_LIGHT,
    }
    
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO: BREAKOUT LIGHT vs SUPER-LIGHT")
    print(f"Testando {len(tickers)} papéis em {len(anos)} anos")
    print(f"⚡ OTIMIZAÇÃO: Baixa cada papel 1× e reutiliza")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # === FASE 1: BAIXAR DADOS ===
    print("📥 FASE 1/2: Baixando dados (1× por papel)...\n")
    
    cache_dados = {}
    data_final = date(ano_atual, 12, 31)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            df = preparar_df(ticker, data_final)
            cache_dados[ticker] = df
            print("✓")
        except Exception as e:
            print(f"ERRO: {e}")
            continue
    
    print(f"\n✓ {len(cache_dados)} papéis carregados em cache\n")
    
    # === FASE 2: TESTAR ESTRATÉGIAS ===
    print("🧪 FASE 2/2: Testando LIGHT vs SUPER-LIGHT...\n")
    
    resultados_por_nivel = {nivel: {} for nivel in estrategias.keys()}
    
    for nivel, estrategia in estrategias.items():
        print(f"\n{'='*60}")
        print(f"TESTANDO: {nivel}")
        print(f"{'='*60}")
        
        for ano in anos:
            print(f"  Ano {ano}:", end=" ", flush=True)
            
            try:
                inicio_ano = date(ano, 1, 1)
                fim_ano = date(ano, 12, 31)
                
                todos_trades = []
                
                for ticker, df_completo in cache_dados.items():
                    try:
                        df_ate_ano = df_completo[df_completo.index <= pd.Timestamp(fim_ano)]
                        
                        if df_ate_ano.empty:
                            continue
                        
                        trades = backtest_estrategia(ticker, df_ate_ano, estrategia)
                        
                        if trades.empty:
                            continue
                        
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        trades_ano = trades[
                            (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                            (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                        ]
                        
                        if not trades_ano.empty:
                            todos_trades.append(trades_ano)
                    except Exception:
                        continue
                
                if not todos_trades:
                    print("sem trades")
                    continue
                
                df_todos = pd.concat(todos_trades, ignore_index=True)
                
                # Métricas
                n_total = len(df_todos)
                n_ganhos = (df_todos["retorno_%"] > 0).sum()
                taxa_acerto = (n_ganhos / n_total) if n_total > 0 else 0
                retorno_med = df_todos["retorno_%"].mean()
                
                dist = df_todos["resultado"].value_counts()
                pct_alvo = dist.get("alvo", 0) / n_total if n_total > 0 else 0
                pct_stop = dist.get("stop", 0) / n_total if n_total > 0 else 0
                
                dias_med = (pct_alvo * 12) + (pct_stop * 7) + ((1 - pct_alvo - pct_stop) * 20)
                ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
                
                resultados_por_nivel[nivel][ano] = {
                    "n_trades": n_total,
                    "taxa_acerto": round(taxa_acerto, 3),
                    "retorno_med_%": round(retorno_med, 2),
                    "ret_anualizado_%": round(ret_anualizado, 1),
                }
                
                print(f"{n_total}t | {taxa_acerto:.1%} | {ret_anualizado:.0f}%/ano")
            
            except Exception as e:
                print(f"ERRO: {e}")
    
    # === COMPARAÇÃO CONSOLIDADA ===
    print(f"\n{'='*70}")
    print("📊 COMPARAÇÃO CONSOLIDADA")
    print(f"{'='*70}\n")
    
    resumo = []
    
    for nivel in estrategias.keys():
        anos_data = resultados_por_nivel[nivel]
        
        if not anos_data:
            continue
        
        taxas = [anos_data[a]["taxa_acerto"] for a in anos if a in anos_data]
        rets = [anos_data[a]["ret_anualizado_%"] for a in anos if a in anos_data]
        n_trades_total = sum([anos_data[a]["n_trades"] for a in anos if a in anos_data])
        
        if not taxas:
            continue
        
        taxa_media = np.mean(taxas)
        ret_medio = np.mean(rets)
        ret_std = np.std(rets)
        pior_ano = min(rets)
        melhor_ano = max(rets)
        sharpe = ret_medio / ret_std if ret_std > 0 else 0
        anos_positivos = sum([1 for r in rets if r > 0])
        anos_negativos = sum([1 for r in rets if r < 0])
        
        resumo.append({
            "nivel": nivel,
            "ops_ano": round(n_trades_total / len(taxas), 1),
            "taxa_%": round(taxa_media * 100, 1),
            "ret_ano_%": round(ret_medio, 1),
            "sharpe": round(sharpe, 2),
            "melhor_%": round(melhor_ano, 1),
            "pior_%": round(pior_ano, 1),
            "anos_pos/neg": f"{anos_positivos}/{anos_negativos}",
        })
    
    if not resumo:
        print("❌ Nenhum resultado válido.")
        print("\n⚠️  BREAKOUT também não gerou sinais suficientes.")
        print("    Mesmo com critérios relaxados, rompimentos são raros na B3.")
        return None
    
    df_resumo = pd.DataFrame(resumo)
    
    print(df_resumo.to_string(index=False))
    
    # === DECISÃO ===
    print(f"\n{'='*70}")
    print("✅ RESULTADO FINAL")
    print(f"{'='*70}\n")
    
    melhor = df_resumo.iloc[0] if len(df_resumo) > 0 else None
    
    if melhor is None:
        print("❌ Nenhuma versão gerou resultados.")
        return None
    
    print(f"🏆 MELHOR: {melhor['nivel']}")
    print(f"   Ops/ano: {melhor['ops_ano']:.1f}")
    print(f"   Taxa: {melhor['taxa_%']:.1f}%")
    print(f"   Retorno: {melhor['ret_ano_%']:.1f}%/ano")
    print(f"   Sharpe: {melhor['sharpe']:.2f}")
    print(f"   Pior ano: {melhor['pior_%']:.1f}%")
    print(f"   Anos pos/neg: {melhor['anos_pos/neg']}")
    print()
    
    # Critérios
    aprovado = True
    alertas = []
    
    if melhor['taxa_%'] < 65:
        alertas.append(f"Taxa {melhor['taxa_%']:.1f}% < 65%")
        aprovado = False
    
    if melhor['pior_%'] < -20:
        alertas.append(f"Pior ano {melhor['pior_%']:.1f}% < -20%")
        aprovado = False
    
    if melhor['ret_ano_%'] < 30:
        alertas.append(f"Retorno {melhor['ret_ano_%']:.1f}% < 30%")
        aprovado = False
    
    if melhor['sharpe'] < 1.0:
        alertas.append(f"Sharpe {melhor['sharpe']:.2f} < 1.0")
        aprovado = False
    
    if aprovado:
        print("✅ BREAKOUT APROVADO!")
        print(f"   Use {melhor['nivel']} no portfólio.")
    else:
        print("❌ BREAKOUT REPROVADO")
        print("   Motivos:")
        for alerta in alertas:
            print(f"     - {alerta}")
        print()
        print("   💀 Breakout também não funciona na B3.")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar
    caminho = f"{CFG['pasta_resultados']}/breakout_comparacao_{data_base}.csv"
    df_resumo.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_resumo


def testar_breakout_walkforward_otimizado(data_base: date = None):
    """
    Versão OTIMIZADA: baixa cada papel 1× e reutiliza.
    
    Reduz de 4,140 downloads → 69 downloads (60× mais rápido).
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    # Configurações a testar
    periodos = [10, 15, 20, 30]
    alvos_stops = [(0.05, 0.04), (0.06, 0.04), (0.07, 0.05)]
    
    print(f"\n{'='*70}")
    print(f"TESTE: BREAKOUT DE VOLATILIDADE (OTIMIZADO)")
    print(f"Testando {len(tickers)} papéis × {len(periodos)} períodos × {len(alvos_stops)} alvos/stops")
    print(f"Anos: {anos}")
    print(f"⚡ OTIMIZAÇÃO: Baixa cada papel 1× e reutiliza")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # === FASE 1: BAIXAR TODOS OS DADOS (1× por papel) ===
    print("📥 FASE 1/2: Baixando dados (1× por papel)...\n")
    
    cache_dados = {}
    data_final = date(ano_atual, 12, 31)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            df = preparar_df(ticker, data_final)
            cache_dados[ticker] = df
            print("✓")
        except Exception as e:
            print(f"ERRO: {e}")
            continue
    
    print(f"\n✓ {len(cache_dados)} papéis carregados em cache\n")
    
    # === FASE 2: TESTAR TODAS AS CONFIGURAÇÕES (usando cache) ===
    print("🧪 FASE 2/2: Testando configurações (reutilizando dados)...\n")
    
    todos_resultados = []
    
    for periodo in periodos:
        for alvo_pct, stop_pct in alvos_stops:
            config_key = f"P{periodo}_A{int(alvo_pct*100)}_S{int(stop_pct*100)}"
            print(f"\n{'='*60}")
            print(f"CONFIG: Período={periodo}d | Alvo={alvo_pct*100:.0f}% | Stop={stop_pct*100:.0f}%")
            print(f"{'='*60}")
            
            for ano in anos:
                print(f"  Ano {ano}:", end=" ", flush=True)
                
                try:
                    inicio_ano = date(ano, 1, 1)
                    fim_ano = date(ano, 12, 31)
                    
                    # Cria estratégia temporária
                    est_temp = ESTRATEGIA_BREAKOUT.copy()
                    est_temp["parametros"] = est_temp["parametros"].copy()
                    est_temp["parametros"]["periodo_breakout"] = periodo
                    est_temp["parametros"]["alvo_pct"] = alvo_pct
                    est_temp["parametros"]["stop_pct"] = stop_pct
                    
                    todos_trades = []
                    
                    # REUTILIZA DADOS DO CACHE (não baixa de novo)
                    for ticker, df_completo in cache_dados.items():
                        try:
                            # Filtra até fim do ano
                            df_ate_ano = df_completo[df_completo.index <= pd.Timestamp(fim_ano)]
                            
                            if df_ate_ano.empty:
                                continue
                            
                            trades = backtest_estrategia(ticker, df_ate_ano, est_temp)
                            
                            if trades.empty:
                                continue
                            
                            trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                            trades_ano = trades[
                                (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                                (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                            ]
                            
                            if not trades_ano.empty:
                                todos_trades.append(trades_ano)
                        except Exception:
                            continue
                    
                    if not todos_trades:
                        print("sem trades")
                        continue
                    
                    df_todos = pd.concat(todos_trades, ignore_index=True)
                    
                    # Métricas
                    n_total = len(df_todos)
                    n_ganhos = (df_todos["retorno_%"] > 0).sum()
                    taxa_acerto = (n_ganhos / n_total) if n_total > 0 else 0
                    retorno_med = df_todos["retorno_%"].mean()
                    
                    dist = df_todos["resultado"].value_counts()
                    pct_alvo = dist.get("alvo", 0) / n_total if n_total > 0 else 0
                    pct_stop = dist.get("stop", 0) / n_total if n_total > 0 else 0
                    
                    dias_med = (pct_alvo * 12) + (pct_stop * 7) + ((1 - pct_alvo - pct_stop) * 20)
                    ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
                    
                    todos_resultados.append({
                        "periodo": periodo,
                        "alvo_%": int(alvo_pct * 100),
                        "stop_%": int(stop_pct * 100),
                        "ano": ano,
                        "n_trades": n_total,
                        "taxa_acerto": round(taxa_acerto, 3),
                        "retorno_med_%": round(retorno_med, 2),
                        "ret_anualizado_%": round(ret_anualizado, 1),
                    })
                    
                    print(f"{n_total}t | {taxa_acerto:.1%} | {ret_anualizado:.0f}%/ano")
                
                except Exception as e:
                    print(f"ERRO: {e}")
    
    # Análise consolidada (mesmo código de antes)
    print(f"\n{'='*70}")
    print("📊 RESUMO POR CONFIGURAÇÃO")
    print(f"{'='*70}\n")
    
    if not todos_resultados:
        print("❌ Nenhum resultado válido.")
        return None
    
    df_resultados = pd.DataFrame(todos_resultados)
    
    # Agrupa por configuração
    resumo = []
    for (periodo, alvo, stop), group in df_resultados.groupby(["periodo", "alvo_%", "stop_%"]):
        taxas = group["taxa_acerto"].values
        rets = group["ret_anualizado_%"].values
        n_trades_total = group["n_trades"].sum()
        
        taxa_media = np.mean(taxas)
        ret_medio = np.mean(rets)
        ret_std = np.std(rets)
        pior_ano = min(rets)
        sharpe = ret_medio / ret_std if ret_std > 0 else 0
        anos_positivos = sum([1 for r in rets if r > 0])
        
        resumo.append({
            "periodo": periodo,
            "alvo/stop": f"{alvo}%/{stop}%",
            "ops_ano": round(n_trades_total / len(taxas), 1),
            "taxa_%": round(taxa_media * 100, 1),
            "ret_ano_%": round(ret_medio, 1),
            "sharpe": round(sharpe, 2),
            "pior_%": round(pior_ano, 1),
            "anos_pos/neg": f"{anos_positivos}/{len(taxas)-anos_positivos}",
        })
    
    df_resumo = pd.DataFrame(resumo)
    df_resumo = df_resumo.sort_values("sharpe", ascending=False)
    
    print(df_resumo.to_string(index=False))
    
    # Melhor configuração
    print(f"\n{'='*70}")
    print("✅ RESULTADO FINAL")
    print(f"{'='*70}\n")
    
    melhor = df_resumo.iloc[0]
    
    print(f"🏆 MELHOR CONFIGURAÇÃO:")
    print(f"   Período: {melhor['periodo']} dias")
    print(f"   Alvo/Stop: {melhor['alvo/stop']}")
    print(f"   Taxa de acerto: {melhor['taxa_%']:.1f}%")
    print(f"   Retorno anual: {melhor['ret_ano_%']:.1f}%")
    print(f"   Sharpe: {melhor['sharpe']:.2f}")
    print(f"   Pior ano: {melhor['pior_%']:.1f}%")
    print(f"   Anos pos/neg: {melhor['anos_pos/neg']}")
    print(f"   Ops/ano: {melhor['ops_ano']:.1f}")
    print()
    
    # Critérios de aprovação
    aprovado = True
    alertas = []
    
    if melhor['taxa_%'] < 65:
        alertas.append(f"Taxa {melhor['taxa_%']:.1f}% < 65%")
        aprovado = False
    
    if melhor['pior_%'] < -20:
        alertas.append(f"Pior ano {melhor['pior_%']:.1f}% < -20%")
        aprovado = False
    
    if melhor['ret_ano_%'] < 30:
        alertas.append(f"Retorno {melhor['ret_ano_%']:.1f}% < 30%")
        aprovado = False
    
    if melhor['sharpe'] < 1.0:
        alertas.append(f"Sharpe {melhor['sharpe']:.2f} < 1.0")
        aprovado = False
    
    if aprovado:
        print("✅ BREAKOUT APROVADO!")
        print("   Atende todos os critérios rigorosos.")
        print(f"   Configure: período={melhor['periodo']}d, alvo/stop={melhor['alvo/stop']}")
    else:
        print("❌ BREAKOUT REPROVADO")
        print("   Motivos:")
        for alerta in alertas:
            print(f"     - {alerta}")
        print()
        print("   Breakout não atinge critérios mínimos para operação.")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar CSV
    caminho = f"{CFG['pasta_resultados']}/breakout_walkforward_{data_base}.csv"
    df_resultados.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_resumo


# Mantém função antiga para compatibilidade
def testar_breakout_walkforward(data_base: date = None):
    """
    Valida estratégia de Breakout através de walk-forward anual.
    
    Testa múltiplos períodos de breakout (10, 15, 20, 30 dias) e
    múltiplas combinações de alvo/stop.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    # Configurações a testar
    periodos = [10, 15, 20, 30]
    alvos_stops = [(0.05, 0.04), (0.06, 0.04), (0.07, 0.05)]
    
    print(f"\n{'='*70}")
    print(f"TESTE: BREAKOUT DE VOLATILIDADE (WALK-FORWARD)")
    print(f"Testando {len(tickers)} papéis × {len(periodos)} períodos × {len(alvos_stops)} alvos/stops")
    print(f"Anos: {anos}")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Resultados: {config: {ano: métricas}}
    todos_resultados = []
    
    for periodo in periodos:
        for alvo_pct, stop_pct in alvos_stops:
            config_key = f"P{periodo}_A{int(alvo_pct*100)}_S{int(stop_pct*100)}"
            print(f"\n{'='*60}")
            print(f"CONFIG: Período={periodo}d | Alvo={alvo_pct*100:.0f}% | Stop={stop_pct*100:.0f}%")
            print(f"{'='*60}")
            
            for ano in anos:
                print(f"  Ano {ano}:", end=" ", flush=True)
                
                try:
                    inicio_ano = date(ano, 1, 1)
                    fim_ano = date(ano, 12, 31)
                    
                    # Cria estratégia temporária
                    est_temp = ESTRATEGIA_BREAKOUT.copy()
                    est_temp["parametros"] = est_temp["parametros"].copy()
                    est_temp["parametros"]["periodo_breakout"] = periodo
                    est_temp["parametros"]["alvo_pct"] = alvo_pct
                    est_temp["parametros"]["stop_pct"] = stop_pct
                    
                    todos_trades = []
                    
                    for ticker in tickers:
                        try:
                            df = preparar_df(ticker, fim_ano)
                            trades = backtest_estrategia(ticker, df, est_temp)
                            
                            if trades.empty:
                                continue
                            
                            trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                            trades_ano = trades[
                                (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                                (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                            ]
                            
                            if not trades_ano.empty:
                                todos_trades.append(trades_ano)
                        except Exception:
                            continue
                    
                    if not todos_trades:
                        print("sem trades")
                        continue
                    
                    df_todos = pd.concat(todos_trades, ignore_index=True)
                    
                    # Métricas
                    n_total = len(df_todos)
                    n_ganhos = (df_todos["retorno_%"] > 0).sum()
                    taxa_acerto = (n_ganhos / n_total) if n_total > 0 else 0
                    retorno_med = df_todos["retorno_%"].mean()
                    
                    dist = df_todos["resultado"].value_counts()
                    pct_alvo = dist.get("alvo", 0) / n_total if n_total > 0 else 0
                    pct_stop = dist.get("stop", 0) / n_total if n_total > 0 else 0
                    
                    dias_med = (pct_alvo * 12) + (pct_stop * 7) + ((1 - pct_alvo - pct_stop) * 20)
                    ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
                    
                    todos_resultados.append({
                        "periodo": periodo,
                        "alvo_%": int(alvo_pct * 100),
                        "stop_%": int(stop_pct * 100),
                        "ano": ano,
                        "n_trades": n_total,
                        "taxa_acerto": round(taxa_acerto, 3),
                        "retorno_med_%": round(retorno_med, 2),
                        "ret_anualizado_%": round(ret_anualizado, 1),
                    })
                    
                    print(f"{n_total}t | {taxa_acerto:.1%} | {ret_anualizado:.0f}%/ano")
                
                except Exception as e:
                    print(f"ERRO: {e}")
    
    # Análise consolidada por configuração
    print(f"\n{'='*70}")
    print("📊 RESUMO POR CONFIGURAÇÃO")
    print(f"{'='*70}\n")
    
    if not todos_resultados:
        print("❌ Nenhum resultado válido.")
        return None
    
    df_resultados = pd.DataFrame(todos_resultados)
    
    # Agrupa por configuração
    resumo = []
    for (periodo, alvo, stop), group in df_resultados.groupby(["periodo", "alvo_%", "stop_%"]):
        taxas = group["taxa_acerto"].values
        rets = group["ret_anualizado_%"].values
        n_trades_total = group["n_trades"].sum()
        
        taxa_media = np.mean(taxas)
        ret_medio = np.mean(rets)
        ret_std = np.std(rets)
        pior_ano = min(rets)
        sharpe = ret_medio / ret_std if ret_std > 0 else 0
        anos_positivos = sum([1 for r in rets if r > 0])
        
        resumo.append({
            "periodo": periodo,
            "alvo/stop": f"{alvo}%/{stop}%",
            "ops_ano": round(n_trades_total / len(taxas), 1),
            "taxa_%": round(taxa_media * 100, 1),
            "ret_ano_%": round(ret_medio, 1),
            "sharpe": round(sharpe, 2),
            "pior_%": round(pior_ano, 1),
            "anos_pos/neg": f"{anos_positivos}/{len(taxas)-anos_positivos}",
        })
    
    df_resumo = pd.DataFrame(resumo)
    df_resumo = df_resumo.sort_values("sharpe", ascending=False)
    
    print(df_resumo.to_string(index=False))
    
    # Melhor configuração
    print(f"\n{'='*70}")
    print("✅ RESULTADO FINAL")
    print(f"{'='*70}\n")
    
    melhor = df_resumo.iloc[0]
    
    print(f"🏆 MELHOR CONFIGURAÇÃO:")
    print(f"   Período: {melhor['periodo']} dias")
    print(f"   Alvo/Stop: {melhor['alvo/stop']}")
    print(f"   Taxa de acerto: {melhor['taxa_%']:.1f}%")
    print(f"   Retorno anual: {melhor['ret_ano_%']:.1f}%")
    print(f"   Sharpe: {melhor['sharpe']:.2f}")
    print(f"   Pior ano: {melhor['pior_%']:.1f}%")
    print(f"   Anos pos/neg: {melhor['anos_pos/neg']}")
    print(f"   Ops/ano: {melhor['ops_ano']:.1f}")
    print()
    
    # Critérios de aprovação
    aprovado = True
    alertas = []
    
    if melhor['taxa_%'] < 65:
        alertas.append(f"Taxa {melhor['taxa_%']:.1f}% < 65%")
        aprovado = False
    
    if melhor['pior_%'] < -20:
        alertas.append(f"Pior ano {melhor['pior_%']:.1f}% < -20%")
        aprovado = False
    
    if melhor['ret_ano_%'] < 30:
        alertas.append(f"Retorno {melhor['ret_ano_%']:.1f}% < 30%")
        aprovado = False
    
    if melhor['sharpe'] < 1.0:
        alertas.append(f"Sharpe {melhor['sharpe']:.2f} < 1.0")
        aprovado = False
    
    if aprovado:
        print("✅ BREAKOUT APROVADO!")
        print("   Atende todos os critérios rigorosos.")
        print(f"   Configure: período={melhor['periodo']}d, alvo/stop={melhor['alvo/stop']}")
    else:
        print("❌ BREAKOUT REPROVADO")
        print("   Motivos:")
        for alerta in alertas:
            print(f"     - {alerta}")
        print()
        print("   Breakout não atinge critérios mínimos para operação.")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar CSV
    caminho = f"{CFG['pasta_resultados']}/breakout_walkforward_{data_base}.csv"
    df_resultados.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_resumo


def otimizar_alvos_stops_light_completo(data_base: date = None):
    """
    OTIMIZAÇÃO FINAL COMPLETA: testa fixos + ATR e recomenda o melhor.
    
    Executa:
    1. Teste de alvos/stops fixos (%)
    2. Teste de alvos/stops ATR
    3. Comparação lado a lado
    4. Recomendação final definitiva
    """
    if data_base is None:
        data_base = date.today()
    
    print(f"\n{'='*70}")
    print(f"🎯 OTIMIZAÇÃO COMPLETA: ALVOS E STOPS - MEAN REVERSION LIGHT")
    print(f"Data base: {data_base}")
    print(f"{'='*70}")
    print()
    print("Este teste vai rodar:")
    print("  1. Alvos/stops FIXOS (%) - 7 combinações")
    print("  2. Alvos/stops ATR - 5 combinações")
    print("  3. Comparação e recomendação final")
    print()
    print("⏱️  Tempo estimado: 40-50 minutos")
    print()
    print(f"{'='*70}\n")
    
    # --- PARTE 1: FIXOS ---
    print("\n" + "="*70)
    print("PARTE 1/2: TESTANDO ALVOS/STOPS FIXOS (%)")
    print("="*70 + "\n")
    
    df_fixos = testar_alvos_stops_light_fixos(data_base)
    
    # --- PARTE 2: ATR ---
    print("\n" + "="*70)
    print("PARTE 2/2: TESTANDO ALVOS/STOPS ATR")
    print("="*70 + "\n")
    
    df_atr = testar_alvos_stops_light_atr(data_base)
    
    # --- COMPARAÇÃO FINAL ---
    print("\n" + "="*70)
    print("🏆 COMPARAÇÃO FINAL: FIXOS vs ATR")
    print("="*70 + "\n")
    
    if df_fixos is None or df_atr is None:
        print("❌ Erro: não foi possível completar ambos os testes.")
        return None
    
    # Melhor de cada categoria
    melhor_fixo = df_fixos.iloc[0]
    melhor_atr = df_atr.iloc[0]
    atual_fixo = df_fixos[df_fixos["alvo/stop"] == "3.0%/4.0%"].iloc[0] if "3.0%/4.0%" in df_fixos["alvo/stop"].values else None
    
    print("📊 RESUMO EXECUTIVO:\n")
    
    # Tabela comparativa
    print("┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Métrica             │ ATUAL (3%/4%)│ MELHOR FIXO  │ MELHOR ATR   │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    
    if atual_fixo is not None:
        print(f"│ Configuração        │ 3.0%/4.0%    │ {melhor_fixo['alvo/stop']:12s} │ {melhor_atr['stop/alvo']:12s} │")
        print(f"│ R:R                 │ {atual_fixo['r:r']:12.2f} │ {melhor_fixo['r:r']:12.2f} │ {melhor_atr['r:r']:12.2f} │")
        print(f"│ Ops/ano             │ {atual_fixo['ops_ano']:12.1f} │ {melhor_fixo['ops_ano']:12.1f} │ {melhor_atr['ops_ano']:12.1f} │")
        print(f"│ Taxa acerto         │ {atual_fixo['taxa_%']:11.1f}% │ {melhor_fixo['taxa_%']:11.1f}% │ {melhor_atr['taxa_%']:11.1f}% │")
        print(f"│ Retorno anual       │ {atual_fixo['ret_ano_%']:11.1f}% │ {melhor_fixo['ret_ano_%']:11.1f}% │ {melhor_atr['ret_ano_%']:11.1f}% │")
        print(f"│ Sharpe              │ {atual_fixo['sharpe']:12.2f} │ {melhor_fixo['sharpe']:12.2f} │ {melhor_atr['sharpe']:12.2f} │")
        print(f"│ Pior ano            │ {atual_fixo['pior_%']:11.1f}% │ {melhor_fixo['pior_%']:11.1f}% │ {melhor_atr['pior_%']:11.1f}% │")
    else:
        print(f"│ Configuração        │ N/A          │ {melhor_fixo['alvo/stop']:12s} │ {melhor_atr['stop/alvo']:12s} │")
        print(f"│ R:R                 │ N/A          │ {melhor_fixo['r:r']:12.2f} │ {melhor_atr['r:r']:12.2f} │")
        print(f"│ Taxa acerto         │ N/A          │ {melhor_fixo['taxa_%']:11.1f}% │ {melhor_atr['taxa_%']:11.1f}% │")
        print(f"│ Retorno anual       │ N/A          │ {melhor_fixo['ret_ano_%']:11.1f}% │ {melhor_atr['ret_ano_%']:11.1f}% │")
        print(f"│ Sharpe              │ N/A          │ {melhor_fixo['sharpe']:12.2f} │ {melhor_atr['sharpe']:12.2f} │")
    
    print("└─────────────────────┴──────────────┴──────────────┴──────────────┘")
    print()
    
    # --- DECISÃO FINAL ---
    print("="*70)
    print("✅ RECOMENDAÇÃO FINAL")
    print("="*70 + "\n")
    
    # Determina o melhor absoluto
    if melhor_fixo['sharpe'] > melhor_atr['sharpe']:
        vencedor = "FIXO"
        config_vencedora = melhor_fixo['alvo/stop']
        sharpe_vencedor = melhor_fixo['sharpe']
        taxa_vencedora = melhor_fixo['taxa_%']
        ret_vencedor = melhor_fixo['ret_ano_%']
    else:
        vencedor = "ATR"
        config_vencedora = melhor_atr['stop/alvo']
        sharpe_vencedor = melhor_atr['sharpe']
        taxa_vencedora = melhor_atr['taxa_%']
        ret_vencedor = melhor_atr['ret_ano_%']
    
    print(f"🏆 VENCEDOR: {vencedor}")
    print(f"   Configuração: {config_vencedora}")
    print(f"   Sharpe: {sharpe_vencedor:.2f}")
    print(f"   Taxa: {taxa_vencedora:.1f}% | Retorno: {ret_vencedor:.1f}%/ano")
    print()
    
    # Compara com atual
    if atual_fixo is not None:
        diff_sharpe = sharpe_vencedor - atual_fixo['sharpe']
        diff_ret = ret_vencedor - atual_fixo['ret_ano_%']
        
        print(f"📊 COMPARAÇÃO COM CONFIGURAÇÃO ATUAL (3%/4%):")
        print(f"   Sharpe: {atual_fixo['sharpe']:.2f} → {sharpe_vencedor:.2f} ({diff_sharpe:+.2f})")
        print(f"   Retorno: {atual_fixo['ret_ano_%']:.1f}% → {ret_vencedor:.1f}% ({diff_ret:+.1f}pp)")
        print()
        
        if diff_sharpe > 0.20:
            print("✅ TROQUE IMEDIATAMENTE")
            print(f"   Ganho de Sharpe {diff_sharpe:+.2f} é MUITO significativo.")
            print(f"   Configure: {config_vencedora}")
        elif diff_sharpe > 0.10:
            print("⚖️  RECOMENDO TROCAR")
            print(f"   Ganho de Sharpe {diff_sharpe:+.2f} é significativo.")
            print(f"   Configure: {config_vencedora}")
        elif diff_sharpe > 0.05:
            print("🤔 CONSIDERE TROCAR")
            print(f"   Ganho de Sharpe {diff_sharpe:+.2f} é moderado.")
            print(f"   Se quiser otimizar ao máximo: {config_vencedora}")
        else:
            print("⚠️  MANTENHA 3%/4%")
            print(f"   Diferença de Sharpe {diff_sharpe:+.2f} é marginal.")
            print("   Não vale o esforço de mudar.")
        
        if vencedor == "ATR":
            print()
            print("⚠️  ATENÇÃO: ATR é adaptativo - stops/alvos variam por papel:")
            print("   Exemplo: MGLU3 (volátil) terá stop/alvo maiores que ITUB4 (estável)")
            print("   Você precisará calcular stop/alvo específico para cada entrada.")
    
    else:
        print("💡 USE A CONFIGURAÇÃO VENCEDORA:")
        print(f"   {config_vencedora}")
        print(f"   Sharpe: {sharpe_vencedor:.2f}")
        print(f"   Taxa: {taxa_vencedora:.1f}% | Retorno: {ret_vencedor:.1f}%/ano")
    
    print()
    print("="*70)
    print()
    print("🎯 OTIMIZAÇÃO CONCLUÍDA!")
    print()
    print("Próximos passos:")
    print("  1. Aplicar a configuração recomendada em ESTRATEGIA_MEAN_REVERSION_LIGHT")
    print("  2. Rodar screening diário: python screening_regras.py")
    print("  3. Executar 3-4 operações/ano com 80%+ de acerto esperado")
    print()
    print("="*70 + "\n")
    
    # Salvar comparação final
    resultado_final = {
        "vencedor": vencedor,
        "config": config_vencedora,
        "sharpe": sharpe_vencedor,
        "taxa_%": taxa_vencedora,
        "ret_ano_%": ret_vencedor,
        "melhor_fixo": melhor_fixo.to_dict(),
        "melhor_atr": melhor_atr.to_dict(),
        "atual": atual_fixo.to_dict() if atual_fixo is not None else None,
    }
    
    caminho = f"{CFG['pasta_resultados']}/otimizacao_final_{data_base}.json"
    with open(caminho, 'w') as f:
        json.dump(resultado_final, f, indent=2, default=str)
    print(f"[✓] Resultado final salvo: {caminho}\n")
    
    return resultado_final


def testar_alvos_stops_light_atr(data_base: date = None):
    """
    Testa alvos/stops baseados em ATR (Average True Range).
    
    Stop = Preço - (multiplicador × ATR)
    Alvo = Preço + (multiplicador × ATR)
    
    Mais adaptativo: papéis voláteis têm stops/alvos maiores.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    # Multiplicadores de ATR a testar
    # (stop_mult, alvo_mult)
    combinacoes_atr = [
        (1.5, 2.0),   # R:R 1.33
        (2.0, 2.5),   # R:R 1.25
        (2.0, 3.0),   # R:R 1.50
        (2.5, 3.0),   # R:R 1.20
        (2.5, 3.5),   # R:R 1.40
    ]
    
    print(f"\n{'='*70}")
    print(f"TESTE: ALVOS E STOPS BASEADOS EM ATR - LIGHT")
    print(f"Testando {len(tickers)} papéis × {len(combinacoes_atr)} combinações × {len(anos)} anos")
    print(f"ATR = Average True Range (volatilidade do papel)")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Resultados
    resultados = {}
    
    for stop_mult, alvo_mult in combinacoes_atr:
        combo_key = f"{stop_mult:.1f}×ATR/{alvo_mult:.1f}×ATR"
        print(f"\n{'='*60}")
        print(f"TESTANDO: Stop {stop_mult}×ATR / Alvo {alvo_mult}×ATR")
        print(f"{'='*60}")
        
        resultados[combo_key] = {}
        
        for ano in anos:
            print(f"  Ano {ano}:", end=" ", flush=True)
            
            try:
                inicio_ano = date(ano, 1, 1)
                fim_ano    = date(ano, 12, 31)
                
                todos_trades = []
                
                for ticker in tickers:
                    try:
                        df = preparar_df(ticker, fim_ano)
                        
                        # Backtest com ATR
                        trades = _backtest_com_atr(ticker, df, ESTRATEGIA_MEAN_REVERSION_LIGHT, 
                                                   stop_mult, alvo_mult)
                        
                        if trades.empty:
                            continue
                        
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        trades_ano = trades[
                            (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                            (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                        ]
                        
                        if not trades_ano.empty:
                            todos_trades.append(trades_ano)
                    except Exception:
                        continue
                
                if not todos_trades:
                    print("sem trades")
                    continue
                
                df_todos = pd.concat(todos_trades, ignore_index=True)
                
                # Métricas
                n_total = len(df_todos)
                n_ganhos = (df_todos["retorno_%"] > 0).sum()
                taxa_acerto = (n_ganhos / n_total) if n_total > 0 else 0
                retorno_med = df_todos["retorno_%"].mean()
                
                dist = df_todos["resultado"].value_counts()
                pct_alvo = dist.get("alvo", 0) / n_total if n_total > 0 else 0
                pct_stop = dist.get("stop", 0) / n_total if n_total > 0 else 0
                
                dias_med = (pct_alvo * 9) + (pct_stop * 6) + ((1 - pct_alvo - pct_stop) * 15)
                ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
                
                resultados[combo_key][ano] = {
                    "n_trades": n_total,
                    "taxa_acerto": round(taxa_acerto, 3),
                    "retorno_med_%": round(retorno_med, 2),
                    "ret_anualizado_%": round(ret_anualizado, 1),
                }
                
                print(f"{n_total}t | {taxa_acerto:.1%} | {ret_anualizado:.0f}%/ano")
            
            except Exception as e:
                print(f"ERRO: {e}")
    
    # --- ANÁLISE CONSOLIDADA ---
    print(f"\n{'='*70}")
    print("📊 COMPARAÇÃO CONSOLIDADA (ATR)")
    print(f"{'='*70}\n")
    
    resumo = []
    
    for combo_key, anos_data in resultados.items():
        if not anos_data:
            continue
        
        taxas = [anos_data[a]["taxa_acerto"] for a in anos if a in anos_data]
        rets = [anos_data[a]["ret_anualizado_%"] for a in anos if a in anos_data]
        n_trades_total = sum([anos_data[a]["n_trades"] for a in anos if a in anos_data])
        
        if not taxas:
            continue
        
        taxa_media = np.mean(taxas)
        ret_medio = np.mean(rets)
        ret_std = np.std(rets)
        pior_ano = min(rets)
        sharpe = ret_medio / ret_std if ret_std > 0 else 0
        anos_positivos = sum([1 for r in rets if r > 0])
        
        stop_str, alvo_str = combo_key.split('/')
        stop_val = float(stop_str.replace('×ATR',''))
        alvo_val = float(alvo_str.replace('×ATR',''))
        rr = alvo_val / stop_val
        
        resumo.append({
            "stop/alvo": combo_key,
            "r:r": round(rr, 2),
            "ops_ano": round(n_trades_total / len(taxas), 1),
            "taxa_%": round(taxa_media * 100, 1),
            "ret_ano_%": round(ret_medio, 1),
            "sharpe": round(sharpe, 2),
            "pior_%": round(pior_ano, 1),
            "anos_pos/neg": f"{anos_positivos}/{len(taxas)-anos_positivos}",
        })
    
    if not resumo:
        print("❌ Nenhum resultado válido.")
        return None
    
    df_resumo = pd.DataFrame(resumo)
    df_resumo = df_resumo.sort_values("sharpe", ascending=False)
    
    print(df_resumo.to_string(index=False))
    
    # --- RECOMENDAÇÃO ---
    print(f"\n{'='*70}")
    print("💡 RECOMENDAÇÃO")
    print(f"{'='*70}\n")
    
    melhor = df_resumo.iloc[0]
    
    print(f"🏆 MELHOR COMBINAÇÃO ATR (maior Sharpe):")
    print(f"   {melhor['stop/alvo']} | R:R 1:{melhor['r:r']:.2f}")
    print(f"   Taxa: {melhor['taxa_%']:.1f}% | Ret: {melhor['ret_ano_%']:.1f}% | Sharpe: {melhor['sharpe']:.2f}")
    print(f"   Pior ano: {melhor['pior_%']:.1f}% | {melhor['anos_pos/neg']}")
    print()
    print("⚠️  Lembre-se: ATR varia por papel. Stops/alvos serão diferentes para cada ação.")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar
    caminho = f"{CFG['pasta_resultados']}/teste_alvos_stops_atr_{data_base}.csv"
    df_resumo.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_resumo


def _backtest_com_atr(ticker: str, df: pd.DataFrame, estrategia: dict, 
                      stop_mult: float, alvo_mult: float) -> pd.DataFrame:
    """
    Backtest usando stops/alvos baseados em ATR em vez de % fixo.
    """
    p = estrategia["parametros"]
    fn_sinal = estrategia["fn_sinal"]
    
    # Calcula ATR se não existir
    if "atr" not in df.columns and "ATR_14" in df.columns:
        df["atr"] = df["ATR_14"]
    elif "atr" not in df.columns:
        # Calcula ATR manualmente
        df["tr"] = df[["high", "low", "close"]].apply(
            lambda x: max(x["high"] - x["low"], 
                         abs(x["high"] - x["close"]), 
                         abs(x["low"] - x["close"])), axis=1
        )
        df["atr"] = df["tr"].rolling(window=14).mean()
    
    trades = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        if not fn_sinal(row, p):
            continue
        
        # Sinal detectado
        data_entrada = row.name
        preco_entrada = row["close"]
        atr_entrada = row.get("atr", preco_entrada * 0.02)  # fallback: 2% do preço
        
        # Calcula stop e alvo baseados em ATR
        stop_preco = preco_entrada - (stop_mult * atr_entrada)
        alvo_preco = preco_entrada + (alvo_mult * atr_entrada)
        
        # Simula operação
        for j in range(i + 1, min(i + 1 + p["max_dias"], len(df))):
            row_saida = df.iloc[j]
            
            # Verifica alvo
            if row_saida["high"] >= alvo_preco:
                retorno_pct = ((alvo_preco / preco_entrada) - 1) * 100
                trades.append({
                    "ticker": ticker,
                    "data_entrada": data_entrada,
                    "data_saida": row_saida.name,
                    "preco_entrada": preco_entrada,
                    "preco_saida": alvo_preco,
                    "retorno_%": retorno_pct,
                    "resultado": "alvo",
                })
                break
            
            # Verifica stop
            if row_saida["low"] <= stop_preco:
                retorno_pct = ((stop_preco / preco_entrada) - 1) * 100
                trades.append({
                    "ticker": ticker,
                    "data_entrada": data_entrada,
                    "data_saida": row_saida.name,
                    "preco_entrada": preco_entrada,
                    "preco_saida": stop_preco,
                    "retorno_%": retorno_pct,
                    "resultado": "stop",
                })
                break
        else:
            # Saiu por tempo
            preco_saida = df.iloc[min(i + p["max_dias"], len(df) - 1)]["close"]
            retorno_pct = ((preco_saida / preco_entrada) - 1) * 100
            trades.append({
                "ticker": ticker,
                "data_entrada": data_entrada,
                "data_saida": df.iloc[min(i + p["max_dias"], len(df) - 1)].name,
                "preco_entrada": preco_entrada,
                "preco_saida": preco_saida,
                "retorno_%": retorno_pct,
                "resultado": "tempo",
            })
    
    return pd.DataFrame(trades)


def testar_alvos_stops_light_fixos(data_base: date = None):
    """
    Testa múltiplas combinações de alvo/stop FIXOS (%) para LIGHT.
    
    Identifica a melhor combinação baseada em:
    - Taxa de acerto
    - Retorno anualizado
    - Sharpe
    - Estabilidade entre anos
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    # Combinações a testar
    combinacoes = [
        (0.025, 0.03),  # 2.5% / 3%
        (0.03, 0.03),   # 3% / 3%
        (0.03, 0.04),   # 3% / 4% (atual)
        (0.04, 0.04),   # 4% / 4%
        (0.04, 0.05),   # 4% / 5%
        (0.05, 0.04),   # 5% / 4%
        (0.05, 0.05),   # 5% / 5%
    ]
    
    print(f"\n{'='*70}")
    print(f"TESTE: ALVOS E STOPS FIXOS (%) - LIGHT")
    print(f"Testando {len(tickers)} papéis × {len(combinacoes)} combinações × {len(anos)} anos")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Resultados: {(alvo, stop): {ano: métricas}}
    resultados = {}
    
    for alvo_pct, stop_pct in combinacoes:
        combo_key = f"{alvo_pct*100:.1f}%/{stop_pct*100:.1f}%"
        print(f"\n{'='*60}")
        print(f"TESTANDO: Alvo {alvo_pct*100:.1f}% / Stop {stop_pct*100:.1f}%")
        print(f"{'='*60}")
        
        resultados[combo_key] = {}
        
        for ano in anos:
            print(f"  Ano {ano}:", end=" ", flush=True)
            
            try:
                inicio_ano = date(ano, 1, 1)
                fim_ano    = date(ano, 12, 31)
                
                # Cria estratégia temporária
                est_temp = ESTRATEGIA_MEAN_REVERSION_LIGHT.copy()
                est_temp["parametros"] = est_temp["parametros"].copy()
                est_temp["parametros"]["alvo_pct"] = alvo_pct
                est_temp["parametros"]["stop_pct"] = stop_pct
                
                todos_trades = []
                
                for ticker in tickers:
                    try:
                        df = preparar_df(ticker, fim_ano)
                        trades = backtest_estrategia(ticker, df, est_temp)
                        
                        if trades.empty:
                            continue
                        
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        trades_ano = trades[
                            (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                            (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                        ]
                        
                        if not trades_ano.empty:
                            todos_trades.append(trades_ano)
                    except Exception:
                        continue
                
                if not todos_trades:
                    print("sem trades")
                    continue
                
                df_todos = pd.concat(todos_trades, ignore_index=True)
                
                # Métricas
                n_total = len(df_todos)
                n_ganhos = (df_todos["retorno_%"] > 0).sum()
                taxa_acerto = (n_ganhos / n_total) if n_total > 0 else 0
                retorno_med = df_todos["retorno_%"].mean()
                
                dist = df_todos["resultado"].value_counts()
                pct_alvo = dist.get("alvo", 0) / n_total if n_total > 0 else 0
                pct_stop = dist.get("stop", 0) / n_total if n_total > 0 else 0
                
                dias_med = (pct_alvo * 9) + (pct_stop * 6) + ((1 - pct_alvo - pct_stop) * 15)
                ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
                
                resultados[combo_key][ano] = {
                    "n_trades": n_total,
                    "taxa_acerto": round(taxa_acerto, 3),
                    "retorno_med_%": round(retorno_med, 2),
                    "ret_anualizado_%": round(ret_anualizado, 1),
                }
                
                print(f"{n_total}t | {taxa_acerto:.1%} | {ret_anualizado:.0f}%/ano")
            
            except Exception as e:
                print(f"ERRO: {e}")
    
    # --- ANÁLISE CONSOLIDADA ---
    print(f"\n{'='*70}")
    print("📊 COMPARAÇÃO CONSOLIDADA")
    print(f"{'='*70}\n")
    
    resumo = []
    
    for combo_key, anos_data in resultados.items():
        if not anos_data:
            continue
        
        taxas = [anos_data[a]["taxa_acerto"] for a in anos if a in anos_data]
        rets = [anos_data[a]["ret_anualizado_%"] for a in anos if a in anos_data]
        n_trades_total = sum([anos_data[a]["n_trades"] for a in anos if a in anos_data])
        
        if not taxas:
            continue
        
        taxa_media = np.mean(taxas)
        ret_medio = np.mean(rets)
        ret_std = np.std(rets)
        pior_ano = min(rets)
        sharpe = ret_medio / ret_std if ret_std > 0 else 0
        anos_positivos = sum([1 for r in rets if r > 0])
        
        alvo_str, stop_str = combo_key.split('/')
        rr = float(alvo_str.replace('%','')) / float(stop_str.replace('%',''))
        
        resumo.append({
            "alvo/stop": combo_key,
            "r:r": round(rr, 2),
            "ops_ano": round(n_trades_total / len(taxas), 1),
            "taxa_%": round(taxa_media * 100, 1),
            "ret_ano_%": round(ret_medio, 1),
            "sharpe": round(sharpe, 2),
            "pior_%": round(pior_ano, 1),
            "anos_pos/neg": f"{anos_positivos}/{len(taxas)-anos_positivos}",
        })
    
    if not resumo:
        print("❌ Nenhum resultado válido.")
        return None
    
    df_resumo = pd.DataFrame(resumo)
    df_resumo = df_resumo.sort_values("sharpe", ascending=False)
    
    print(df_resumo.to_string(index=False))
    
    # --- RECOMENDAÇÃO ---
    print(f"\n{'='*70}")
    print("💡 RECOMENDAÇÃO")
    print(f"{'='*70}\n")
    
    melhor = df_resumo.iloc[0]
    atual = df_resumo[df_resumo["alvo/stop"] == "3.0%/4.0%"].iloc[0] if "3.0%/4.0%" in df_resumo["alvo/stop"].values else None
    
    print(f"🏆 MELHOR COMBINAÇÃO (maior Sharpe):")
    print(f"   {melhor['alvo/stop']} | R:R 1:{melhor['r:r']:.2f}")
    print(f"   Taxa: {melhor['taxa_%']:.1f}% | Ret: {melhor['ret_ano_%']:.1f}% | Sharpe: {melhor['sharpe']:.2f}")
    print(f"   Pior ano: {melhor['pior_%']:.1f}% | {melhor['anos_pos/neg']}")
    print()
    
    if atual is not None and melhor['alvo/stop'] != "3.0%/4.0%":
        diff_sharpe = melhor['sharpe'] - atual['sharpe']
        diff_ret = melhor['ret_ano_%'] - atual['ret_ano_%']
        
        print(f"📊 vs CONFIGURAÇÃO ATUAL (3%/4%):")
        print(f"   Sharpe: {atual['sharpe']:.2f} → {melhor['sharpe']:.2f} ({diff_sharpe:+.2f})")
        print(f"   Retorno: {atual['ret_ano_%']:.1f}% → {melhor['ret_ano_%']:.1f}% ({diff_ret:+.1f}pp)")
        print()
        
        if diff_sharpe > 0.15:
            print(f"✅ TROQUE para {melhor['alvo/stop']} (ganho significativo)")
        elif diff_sharpe > 0.05:
            print(f"⚖️  CONSIDERE trocar para {melhor['alvo/stop']} (ganho moderado)")
        else:
            print(f"⚠️  MANTENHA 3%/4% (diferença marginal)")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar
    caminho = f"{CFG['pasta_resultados']}/teste_alvos_stops_fixos_{data_base}.csv"
    df_resumo.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_resumo


def comparar_niveis_mean_reversion(data_base: date = None):
    """
    Compara 3 níveis de mean reversion lado a lado:
    - LIGHT: critérios balanceados + regime
    - SUPER-LIGHT: critérios relaxados + regime
    - SUPER-LIGHT SEM REGIME: critérios relaxados, opera sempre
    
    Objetivo: trade-off entre frequência e qualidade.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    # Estratégias a comparar
    estrategias = {
        "LIGHT": ESTRATEGIA_MEAN_REVERSION_LIGHT,
        "SUPER-LIGHT": ESTRATEGIA_MEAN_REVERSION_SUPER_LIGHT,
        "SUPER-LIGHT SEM REGIME": ESTRATEGIA_MEAN_REVERSION_SUPER_LIGHT_SEM_REGIME,
    }
    
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO: NÍVEIS DE MEAN REVERSION")
    print(f"Testando {len(tickers)} papéis em {len(anos)} anos")
    print(f"Níveis: LIGHT | SUPER-LIGHT | SUPER-LIGHT SEM REGIME")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Resultados: {nivel: {ano: métricas}}
    resultados_por_nivel = {nivel: {} for nivel in estrategias.keys()}
    
    for nivel, estrategia in estrategias.items():
        print(f"\n{'='*60}")
        print(f"TESTANDO: {nivel}")
        print(f"{'='*60}")
        
        for ano in anos:
            print(f"\n  Ano {ano}:", end=" ", flush=True)
            
            try:
                inicio_ano = date(ano, 1, 1)
                fim_ano    = date(ano, 12, 31)
                
                todos_trades = []
                
                for ticker in tickers:
                    try:
                        df = preparar_df(ticker, fim_ano)
                        trades = backtest_estrategia(ticker, df, estrategia)
                        
                        if trades.empty:
                            continue
                        
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        trades_ano = trades[
                            (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                            (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                        ]
                        
                        if not trades_ano.empty:
                            todos_trades.append(trades_ano)
                    except Exception:
                        continue
                
                if not todos_trades:
                    print("sem trades")
                    continue
                
                df_todos = pd.concat(todos_trades, ignore_index=True)
                
                # Métricas
                n_total     = len(df_todos)
                n_ganhos    = (df_todos["retorno_%"] > 0).sum()
                taxa_acerto = (n_ganhos / n_total) if n_total > 0 else 0
                retorno_med = df_todos["retorno_%"].mean()
                
                dist = df_todos["resultado"].value_counts()
                pct_alvo = dist.get("alvo", 0) / n_total if n_total > 0 else 0
                pct_stop = dist.get("stop", 0) / n_total if n_total > 0 else 0
                
                dias_med = (pct_alvo * 9) + (pct_stop * 6) + ((1 - pct_alvo - pct_stop) * 15)
                ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
                
                resultados_por_nivel[nivel][ano] = {
                    "n_trades":         n_total,
                    "taxa_acerto":      round(taxa_acerto, 3),
                    "retorno_med_%":    round(retorno_med, 2),
                    "ret_anualizado_%": round(ret_anualizado, 1),
                }
                
                print(f"{n_total}t | {taxa_acerto:.1%} | {ret_anualizado:.0f}%/ano")
            
            except Exception as e:
                print(f"ERRO: {e}")
    
    # --- COMPARAÇÃO CONSOLIDADA ---
    print(f"\n{'='*70}")
    print("📊 COMPARAÇÃO CONSOLIDADA")
    print(f"{'='*70}\n")
    
    resumo = []
    
    for nivel in estrategias.keys():
        anos_data = resultados_por_nivel[nivel]
        
        if not anos_data:
            continue
        
        taxas = [anos_data[a]["taxa_acerto"] for a in anos if a in anos_data]
        rets  = [anos_data[a]["ret_anualizado_%"] for a in anos if a in anos_data]
        n_trades_total = sum([anos_data[a]["n_trades"] for a in anos if a in anos_data])
        
        if not taxas:
            continue
        
        taxa_media = np.mean(taxas)
        ret_medio  = np.mean(rets)
        ret_std    = np.std(rets)
        pior_ano   = min(rets)
        melhor_ano = max(rets)
        anos_positivos = sum([1 for r in rets if r > 0])
        anos_negativos = sum([1 for r in rets if r < 0])
        
        # Sharpe simplificado (ret / volatilidade)
        sharpe = ret_medio / ret_std if ret_std > 0 else 0
        
        resumo.append({
            "nivel":            nivel,
            "ops_ano":          round(n_trades_total / len(taxas), 1),
            "taxa_%":           round(taxa_media * 100, 1),
            "ret_ano_%":        round(ret_medio, 1),
            "desvio_%":         round(ret_std, 1),
            "sharpe":           round(sharpe, 2),
            "melhor_%":         round(melhor_ano, 1),
            "pior_%":           round(pior_ano, 1),
            "anos_pos/neg":     f"{anos_positivos}/{anos_negativos}",
        })
    
    if not resumo:
        print("❌ Nenhum resultado válido.")
        return None
    
    df_resumo = pd.DataFrame(resumo)
    
    print(df_resumo.to_string(index=False))
    
    # --- ANÁLISE E RECOMENDAÇÃO ---
    print(f"\n{'='*70}")
    print("💡 ANÁLISE DO TRADE-OFF")
    print(f"{'='*70}\n")
    
    light = df_resumo[df_resumo["nivel"] == "LIGHT"].iloc[0] if "LIGHT" in df_resumo["nivel"].values else None
    sem_regime = df_resumo[df_resumo["nivel"] == "SUPER-LIGHT SEM REGIME"].iloc[0] if "SUPER-LIGHT SEM REGIME" in df_resumo["nivel"].values else None
    
    if light is not None and sem_regime is not None:
        print(f"📊 LIGHT (com regime) vs SEM REGIME:")
        print(f"   Operações/ano:  {light['ops_ano']:.1f} → {sem_regime['ops_ano']:.1f} ({sem_regime['ops_ano'] / light['ops_ano']:.1f}×)")
        print(f"   Taxa acerto:    {light['taxa_%']:.1f}% → {sem_regime['taxa_%']:.1f}% ({sem_regime['taxa_%'] - light['taxa_%']:+.1f}pp)")
        print(f"   Ret anualizado: {light['ret_ano_%']:.1f}% → {sem_regime['ret_ano_%']:.1f}% ({sem_regime['ret_ano_%'] - light['ret_ano_%']:+.1f}pp)")
        print(f"   Sharpe:         {light['sharpe']:.2f} → {sem_regime['sharpe']:.2f} ({sem_regime['sharpe'] - light['sharpe']:+.2f})")
        print(f"   Pior ano:       {light['pior_%']:.1f}% → {sem_regime['pior_%']:.1f}% ({sem_regime['pior_%'] - light['pior_%']:+.1f}pp)")
        print(f"   Anos pos/neg:   {light['anos_pos/neg']} → {sem_regime['anos_pos/neg']}")
        print()
        
        # Decide
        if sem_regime['anos_pos/neg'].split('/')[1] != '0':
            print("⚠️  SEM REGIME tem anos negativos.")
            print(f"   Pior ano: {sem_regime['pior_%']:.1f}%")
            print()
            if sem_regime['pior_%'] < -20:
                print("❌ REJEITA SEM REGIME: pior ano < -20% (critério de segurança)")
                print(f"   Mantenha LIGHT ({light['ops_ano']:.0f} ops/ano, {light['taxa_%']:.0f}% acerto)")
            else:
                print("⚖️  SEM REGIME é operável, mas com risco:")
                print(f"   + Frequência {sem_regime['ops_ano'] / light['ops_ano']:.1f}× maior")
                print(f"   - Taxa {sem_regime['taxa_%'] - light['taxa_%']:.0f}pp pior")
                print(f"   - Anos negativos possíveis")
                print()
                print("   💡 Escolha baseada em perfil:")
                print(f"      CONSERVADOR → LIGHT ({light['taxa_%']:.0f}% acerto, sem anos negativos)")
                print(f"      AGRESSIVO → SEM REGIME ({sem_regime['ops_ano']:.0f} ops/ano, aceita volatilidade)")
        else:
            if sem_regime['sharpe'] > light['sharpe']:
                print("✅ RECOMENDAÇÃO: SEM REGIME")
                print(f"   Sharpe {sem_regime['sharpe']:.2f} > {light['sharpe']:.2f}")
                print(f"   Frequência {sem_regime['ops_ano'] / light['ops_ano']:.1f}× maior")
                print(f"   Sem anos negativos!")
            else:
                print("⚖️  TRADE-OFF EQUILIBRADO:")
                print(f"   LIGHT: {light['ops_ano']:.0f} ops, {light['taxa_%']:.0f}% acerto, sharpe {light['sharpe']:.2f}")
                print(f"   SEM REGIME: {sem_regime['ops_ano']:.0f} ops, {sem_regime['taxa_%']:.0f}% acerto, sharpe {sem_regime['sharpe']:.2f}")
                print()
                print("   💡 Considere usar AMBOS com 50/50 de capital")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar CSV
    caminho = f"{CFG['pasta_resultados']}/comparacao_mean_reversion_{data_base}.csv"
    df_resumo.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_resumo


def testar_mean_reversion_walkforward(data_base: date = None):
    """
    Testa estratégia de Mean Reversion através de walk-forward anual.
    
    Usa versão LIGHT (critérios relaxados) para gerar sinais operáveis.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"TESTE DE MEAN REVERSION LIGHT (WALK-FORWARD)")
    print(f"Testando {len(tickers)} papéis em {len(anos)} anos")
    print(f"Anos: {anos}")
    print(f"Estratégia: Compra em sobrevenda (CRITÉRIOS RELAXADOS)")
    print(f"  - RSI < 40 (relaxado de 30)")
    print(f"  - Queda > 3% em 5d (relaxado de 5%)")
    print(f"  - Close > MM50 (relaxado de MM200)")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Resultados por ano
    resultados_por_ano = {}
    
    for ano in anos:
        print(f"\n{'='*60}")
        print(f"ANO: {ano}")
        print(f"{'='*60}")
        
        inicio_ano = date(ano, 1, 1)
        fim_ano    = date(ano, 12, 31)
        
        todos_trades = []
        
        for ticker in tickers:
            try:
                df = preparar_df(ticker, fim_ano)
                
                # Backtest com Mean Reversion LIGHT
                trades = backtest_estrategia(ticker, df, ESTRATEGIA_MEAN_REVERSION_LIGHT)
                
                if trades.empty:
                    continue
                
                # Filtra trades do ano
                trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                trades_ano = trades[
                    (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                    (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                ]
                
                if not trades_ano.empty:
                    todos_trades.append(trades_ano)
            
            except Exception:
                continue
        
        if not todos_trades:
            print("  Sem trades no ano")
            continue
        
        df_todos = pd.concat(todos_trades, ignore_index=True)
        
        # Métricas
        n_total     = len(df_todos)
        n_ganhos    = (df_todos["retorno_%"] > 0).sum()
        taxa_acerto = (n_ganhos / n_total) if n_total > 0 else 0
        retorno_med = df_todos["retorno_%"].mean()
        
        # Distribuição de saídas
        dist = df_todos["resultado"].value_counts()
        n_alvo  = dist.get("alvo", 0)
        n_stop  = dist.get("stop", 0)
        n_tempo = dist.get("tempo", 0)
        
        pct_alvo = n_alvo / n_total if n_total > 0 else 0
        pct_stop = n_stop / n_total if n_total > 0 else 0
        
        # Estimativa retorno anualizado
        dias_med = (pct_alvo * 9) + (pct_stop * 6) + ((1 - pct_alvo - pct_stop) * 15)
        ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
        
        resultados_por_ano[ano] = {
            "n_trades":         n_total,
            "taxa_acerto":      round(taxa_acerto, 3),
            "retorno_med_%":    round(retorno_med, 2),
            "ret_anualizado_%": round(ret_anualizado, 1),
            "n_alvo":           n_alvo,
            "n_stop":           n_stop,
            "n_tempo":          n_tempo,
        }
        
        print(f"\n  Trades: {n_total}")
        print(f"  Taxa acerto: {taxa_acerto:.1%}")
        print(f"  Retorno médio: {retorno_med:.2f}%")
        print(f"  Retorno anualizado: {ret_anualizado:.1f}%")
        print(f"  Saídas: {n_alvo} alvo / {n_stop} stop / {n_tempo} tempo")
    
    # --- ANÁLISE CONSOLIDADA ---
    print(f"\n{'='*70}")
    print("📊 RESUMO POR ANO")
    print(f"{'='*70}\n")
    
    for ano in anos:
        if ano not in resultados_por_ano:
            print(f"{ano}: Sem dados")
            continue
        
        r = resultados_por_ano[ano]
        print(f"{ano}:")
        print(f"  {r['n_trades']} trades | "
              f"Acerto {r['taxa_acerto']:.1%} | "
              f"Ret {r['ret_anualizado_%']:.1f}%/ano | "
              f"{r['n_alvo']}A/{r['n_stop']}S/{r['n_tempo']}T")
    
    # --- VALIDAÇÃO FINAL ---
    print(f"\n{'='*70}")
    print("✅ VALIDAÇÃO FINAL")
    print(f"{'='*70}\n")
    
    if not resultados_por_ano:
        print("❌ Nenhum dado válido para análise.")
        return None
    
    # Estatísticas agregadas
    taxas = [r["taxa_acerto"] for r in resultados_por_ano.values()]
    rets  = [r["ret_anualizado_%"] for r in resultados_por_ano.values()]
    total_trades = sum([r["n_trades"] for r in resultados_por_ano.values()])
    
    taxa_media = np.mean(taxas)
    ret_medio  = np.mean(rets)
    ret_std    = np.std(rets)
    pior_ano   = min(rets)
    anos_positivos = sum([1 for r in rets if r > 0])
    
    print(f"MÉTRICAS CONSOLIDADAS:")
    print(f"  Anos testados: {len(taxas)}")
    print(f"  Total trades: {total_trades}")
    print(f"  Taxa acerto média: {taxa_media:.1%}")
    print(f"  Retorno anualizado médio: {ret_medio:.1f}%")
    print(f"  Desvio padrão: ±{ret_std:.1f}%")
    print(f"  Pior ano: {pior_ano:.1f}%")
    print(f"  Anos positivos: {anos_positivos}/{len(taxas)}")
    print()
    
    # Critérios de aprovação
    aprovado = True
    alertas = []
    
    if taxa_media < 0.65:
        alertas.append(f"Taxa {taxa_media:.1%} < 65%")
        aprovado = False
    
    if pior_ano < -20:
        alertas.append(f"Pior ano {pior_ano:.1f}% < -20%")
        aprovado = False
    
    if ret_medio < 30:
        alertas.append(f"Retorno {ret_medio:.1f}% < 30%")
        aprovado = False
    
    if anos_positivos < len(taxas) * 0.6:
        alertas.append(f"Só {anos_positivos}/{len(taxas)} anos positivos")
        aprovado = False
    
    if total_trades < 50:
        alertas.append(f"Total trades {total_trades} < 50 (pouco frequente)")
        aprovado = False
    
    if aprovado:
        print("✅ MEAN REVERSION APROVADO!")
        print("   A estratégia atende todos os critérios rigorosos.")
        print("   Reversão à média FUNCIONA na B3 de forma consistente.")
        print()
        print(f"   💡 Próximo passo: Usar ESTRATEGIA_MEAN_REVERSION_LIGHT no screening diário")
    else:
        print("❌ MEAN REVERSION REPROVADO")
        print("   Motivos:")
        for alerta in alertas:
            print(f"     - {alerta}")
        print()
        print("   Mesmo com critérios relaxados, reversão não atinge mínimos.")
        print()
        print("   🏁 CONCLUSÃO FINAL: Timing de curto prazo não é viável na B3.")
        print("      Testamos momentum, scoring, mean reversion (strict e light).")
        print("      Nenhuma abordagem técnica de 10-15 dias passa em validação rigorosa.")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar CSV
    dados = []
    for ano, r in resultados_por_ano.items():
        dados.append({"ano": ano, **r})
    
    if dados:
        df_result = pd.DataFrame(dados)
        caminho = f"{CFG['pasta_resultados']}/mean_reversion_light_walkforward_{data_base}.csv"
        df_result.to_csv(caminho, index=False)
        print(f"[✓] Resultados salvos: {caminho}")
    
    return resultados_por_ano


def testar_scoring_system_walkforward(data_base: date = None):
    """
    Testa sistema de scoring com múltiplos thresholds através de walk-forward.
    
    Para cada threshold de score (6, 7, 8, 9, 10):
    - Roda backtest ano a ano
    - Valida estabilidade
    - Identifica threshold ótimo
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Thresholds de score a testar
    scores_testar = [6, 7, 8, 9, 10]
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"TESTE DE SCORING SYSTEM (WALK-FORWARD)")
    print(f"Testando {len(tickers)} papéis × {len(scores_testar)} thresholds × {len(anos)} anos")
    print(f"Anos: {anos}")
    print(f"Thresholds: {scores_testar}")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Resultados: {score_min: {ano: métricas}}
    resultados_por_score = {s: {} for s in scores_testar}
    
    for score_min in scores_testar:
        print(f"\n{'='*60}")
        print(f"TESTANDO SCORE MÍNIMO: {score_min}")
        print(f"{'='*60}")
        
        # Cria estratégia com esse threshold
        est_temp = ESTRATEGIA_MOMENTUM_SCORING.copy()
        est_temp["parametros"] = est_temp["parametros"].copy()
        est_temp["parametros"]["score_minimo"] = score_min
        
        for ano in anos:
            print(f"\n  Ano {ano}:", end=" ")
            
            try:
                inicio_ano = date(ano, 1, 1)
                fim_ano    = date(ano, 12, 31)
                
                todos_trades = []
                
                for ticker in tickers:
                    try:
                        df = preparar_df(ticker, fim_ano)
                        trades = backtest_estrategia(ticker, df, est_temp)
                        
                        if trades.empty:
                            continue
                        
                        # Filtra trades do ano
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        trades_ano = trades[
                            (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                            (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                        ]
                        
                        if not trades_ano.empty:
                            todos_trades.append(trades_ano)
                    
                    except Exception:
                        continue
                
                if not todos_trades:
                    print("sem trades")
                    continue
                
                df_todos = pd.concat(todos_trades, ignore_index=True)
                
                # Métricas
                n_total     = len(df_todos)
                n_ganhos    = (df_todos["retorno_%"] > 0).sum()
                taxa_acerto = (n_ganhos / n_total) if n_total > 0 else 0
                retorno_med = df_todos["retorno_%"].mean()
                
                # Estimativa de retorno anualizado
                dist = df_todos["resultado"].value_counts()
                pct_alvo = dist.get("alvo", 0) / n_total if n_total > 0 else 0
                pct_stop = dist.get("stop", 0) / n_total if n_total > 0 else 0
                
                dias_med = (pct_alvo * 6) + (pct_stop * 4) + ((1 - pct_alvo - pct_stop) * 10)
                ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
                
                resultados_por_score[score_min][ano] = {
                    "n_trades":         n_total,
                    "taxa_acerto":      round(taxa_acerto, 3),
                    "retorno_med_%":    round(retorno_med, 2),
                    "ret_anualizado_%": round(ret_anualizado, 1),
                }
                
                print(f"{n_total} trades | acerto {taxa_acerto:.1%} | ret {ret_anualizado:.0f}%/ano")
            
            except Exception as e:
                print(f"ERRO: {e}")
    
    # --- ANÁLISE COMPARATIVA ---
    print(f"\n{'='*70}")
    print("📊 COMPARAÇÃO POR THRESHOLD DE SCORE")
    print(f"{'='*70}\n")
    
    resumo = []
    
    for score_min in scores_testar:
        anos_data = resultados_por_score[score_min]
        
        if not anos_data:
            continue
        
        # Estatísticas agregadas
        taxas = [anos_data[a]["taxa_acerto"] for a in anos if a in anos_data]
        rets  = [anos_data[a]["ret_anualizado_%"] for a in anos if a in anos_data]
        n_trades_total = sum([anos_data[a]["n_trades"] for a in anos if a in anos_data])
        
        if not taxas:
            continue
        
        taxa_media = np.mean(taxas)
        ret_medio  = np.mean(rets)
        ret_std    = np.std(rets)
        pior_ano   = min(rets) if rets else 0
        anos_positivos = sum([1 for r in rets if r > 0])
        
        resumo.append({
            "score_min":        score_min,
            "anos_testados":    len(taxas),
            "taxa_acerto_med":  round(taxa_media, 3),
            "ret_anualizado_%": round(ret_medio, 1),
            "desvio_padrao_%":  round(ret_std, 1),
            "pior_ano_%":       round(pior_ano, 1),
            "anos_positivos":   f"{anos_positivos}/{len(taxas)}",
            "total_trades":     n_trades_total,
        })
    
    if not resumo:
        print("❌ Nenhum resultado válido.")
        return None
    
    df_resumo = pd.DataFrame(resumo)
    df_resumo = df_resumo.sort_values("ret_anualizado_%", ascending=False)
    
    print(df_resumo.to_string(index=False))
    
    # --- RECOMENDAÇÃO ---
    print(f"\n{'='*70}")
    print("✅ ANÁLISE E RECOMENDAÇÃO")
    print(f"{'='*70}\n")
    
    melhor = df_resumo.iloc[0]
    
    print(f"🏆 MELHOR THRESHOLD: score >= {melhor['score_min']}")
    print(f"   Taxa de acerto média: {melhor['taxa_acerto_med']:.1%}")
    print(f"   Retorno anualizado: {melhor['ret_anualizado_%']:.1f}%")
    print(f"   Desvio padrão: ±{melhor['desvio_padrao_%']:.1f}%")
    print(f"   Pior ano: {melhor['pior_ano_%']:.1f}%")
    print(f"   Anos positivos: {melhor['anos_positivos']}")
    print(f"   Total trades (5 anos): {melhor['total_trades']}")
    print()
    
    # Avalia se é operável
    operavel = True
    alertas = []
    
    if melhor['taxa_acerto_med'] < 0.65:
        alertas.append(f"Taxa {melhor['taxa_acerto_med']:.1%} < 65%")
        operavel = False
    
    if melhor['pior_ano_%'] < -20:
        alertas.append(f"Pior ano {melhor['pior_ano_%']:.1f}% < -20%")
        operavel = False
    
    if melhor['ret_anualizado_%'] < 30:
        alertas.append(f"Retorno {melhor['ret_anualizado_%']:.1f}% < 30%")
        operavel = False
    
    anos_pos = int(melhor['anos_positivos'].split('/')[0])
    anos_tot = int(melhor['anos_positivos'].split('/')[1])
    if anos_pos < anos_tot * 0.6:
        alertas.append(f"Só {anos_pos}/{anos_tot} anos positivos")
        operavel = False
    
    if operavel:
        print("✅ SISTEMA APROVADO!")
        print(f"   Scoring com threshold >= {melhor['score_min']} atende todos os critérios.")
        print(f"   Pode ser operado com confiança.")
        print()
        print(f"💡 Próximo passo: ajustar ESTRATEGIA_MOMENTUM_SCORING para score_minimo={melhor['score_min']}")
    else:
        print("❌ SISTEMA REPROVADO")
        print(f"   Motivos:")
        for alerta in alertas:
            print(f"     - {alerta}")
        print()
        print("   Mesmo com scoring adaptativo, momentum não atinge critérios mínimos.")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar CSV
    caminho = f"{CFG['pasta_resultados']}/scoring_system_walkforward_{data_base}.csv"
    df_resumo.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_resumo


def validar_papeis_individuais_com_regime(data_base: date = None):
    """
    Valida cada papel individualmente através de janelas anuais COM filtro de regime.
    
    Identifica quais papéis têm desempenho CONSISTENTE e ROBUSTO:
    - Taxa de acerto >65% na maioria dos anos
    - Nunca teve ano com perda >-20%
    - Mínimo de trades por ano
    
    Gera lista final de "papéis aprovados" para operação.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    # Janelas anuais
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))  # últimos 5 anos
    
    print(f"\n{'='*70}")
    print(f"VALIDAÇÃO INDIVIDUAL POR PAPEL (COM FILTRO DE REGIME)")
    print(f"Testando {len(tickers)} papéis em {len(anos)} anos")
    print(f"Anos: {anos}")
    print(f"Configuração: alvo 5% / stop 4% / max 10 dias / Ibov>MM200")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Resultados: {ticker: {ano: métricas}}
    resultados_por_ticker = {}
    
    for ticker in tickers:
        print(f"\n{ticker}:", end=" ", flush=True)
        resultados_por_ticker[ticker] = {}
        
        for ano in anos:
            try:
                inicio_ano = date(ano, 1, 1)
                fim_ano    = date(ano, 12, 31)
                
                # Carrega histórico
                df = preparar_df(ticker, fim_ano)
                
                # Backtest com filtro de regime ATIVADO
                trades = backtest_estrategia(ticker, df, ESTRATEGIA_MOMENTUM)
                
                if trades.empty:
                    continue
                
                # Filtra trades dentro do ano
                trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                trades_ano = trades[
                    (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                    (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                ]
                
                if trades_ano.empty:
                    continue
                
                # Métricas
                n_trades    = len(trades_ano)
                n_ganhos    = (trades_ano["retorno_%"] > 0).sum()
                taxa_acerto = (n_ganhos / n_trades) if n_trades > 0 else 0
                retorno_med = trades_ano["retorno_%"].mean()
                
                # Estimativa de retorno anualizado
                dist = trades_ano["resultado"].value_counts()
                pct_alvo = dist.get("alvo", 0) / n_trades if n_trades > 0 else 0
                pct_stop = dist.get("stop", 0) / n_trades if n_trades > 0 else 0
                
                dias_med = (pct_alvo * 6) + (pct_stop * 4) + ((1 - pct_alvo - pct_stop) * 10)
                ret_anualizado = (retorno_med / 100) * (252 / dias_med) * 100 if dias_med > 0 else 0
                
                resultados_por_ticker[ticker][ano] = {
                    "n_trades":         n_trades,
                    "taxa_acerto":      round(taxa_acerto, 3),
                    "retorno_med_%":    round(retorno_med, 2),
                    "ret_anualizado_%": round(ret_anualizado, 1),
                }
                
                print(f"{ano}:{n_trades}t/{taxa_acerto:.1%}", end=" ")
            
            except Exception as e:
                print(f"{ano}:ERRO", end=" ")
                continue
        
        print()  # nova linha após todos os anos
    
    # --- ANÁLISE E APROVAÇÃO ---
    print(f"\n{'='*70}")
    print("🔬 ANÁLISE DE APROVAÇÃO (critérios rigorosos)")
    print(f"{'='*70}\n")
    
    papeis_aprovados = []
    papeis_rejeitados = []
    
    for ticker, anos_data in resultados_por_ticker.items():
        if not anos_data:
            continue
        
        # Critérios de aprovação
        anos_com_dados = [a for a in anos if a in anos_data]
        
        if len(anos_com_dados) < 3:
            # Precisa de pelo menos 3 anos de dados
            papeis_rejeitados.append((ticker, "Dados insuficientes (<3 anos)"))
            continue
        
        # Extrai métricas
        taxas_acerto = [anos_data[a]["taxa_acerto"] for a in anos_com_dados]
        rets_anualizados = [anos_data[a]["ret_anualizado_%"] for a in anos_com_dados]
        n_trades_total = sum([anos_data[a]["n_trades"] for a in anos_com_dados])
        
        taxa_media = np.mean(taxas_acerto)
        ret_medio  = np.mean(rets_anualizados)
        ret_std    = np.std(rets_anualizados)
        pior_ano   = min(rets_anualizados)
        
        # Critérios de aprovação (RIGOROSOS)
        aprovado = True
        motivo_rejeicao = []
        
        if taxa_media < 0.65:
            aprovado = False
            motivo_rejeicao.append(f"Taxa média {taxa_media:.1%} < 65%")
        
        if pior_ano < -20:
            aprovado = False
            motivo_rejeicao.append(f"Ano ruim {pior_ano:.1f}% < -20%")
        
        if n_trades_total < 15:
            aprovado = False
            motivo_rejeicao.append(f"Total trades {n_trades_total} < 15")
        
        if ret_medio < 30:
            aprovado = False
            motivo_rejeicao.append(f"Ret médio {ret_medio:.1f}% < 30%")
        
        anos_positivos = sum([1 for r in rets_anualizados if r > 0])
        if anos_positivos < len(anos_com_dados) * 0.6:  # pelo menos 60% dos anos positivos
            aprovado = False
            motivo_rejeicao.append(f"Só {anos_positivos}/{len(anos_com_dados)} anos positivos")
        
        if aprovado:
            papeis_aprovados.append({
                "ticker":           ticker,
                "anos_testados":    len(anos_com_dados),
                "taxa_acerto_med":  round(taxa_media, 3),
                "ret_anualizado_%": round(ret_medio, 1),
                "desvio_padrao":    round(ret_std, 1),
                "pior_ano_%":       round(pior_ano, 1),
                "total_trades":     n_trades_total,
            })
        else:
            papeis_rejeitados.append((ticker, " | ".join(motivo_rejeicao)))
    
    # --- RESULTADOS ---
    print(f"✅ PAPÉIS APROVADOS: {len(papeis_aprovados)}\n")
    
    if papeis_aprovados:
        df_aprovados = pd.DataFrame(papeis_aprovados)
        df_aprovados = df_aprovados.sort_values("ret_anualizado_%", ascending=False)
        
        print(df_aprovados.to_string(index=False))
        print()
        
        # Salvar lista de aprovados
        tickers_aprovados = df_aprovados["ticker"].tolist()
        caminho_aprovados = "papeis_aprovados_regime.json"
        
        with open(caminho_aprovados, "w") as f:
            json.dump({
                "data_validacao": str(date.today()),
                "criterios": {
                    "taxa_acerto_min": 0.65,
                    "pior_ano_max": -20,
                    "min_trades_total": 15,
                    "ret_medio_min": 30,
                    "pct_anos_positivos": 0.6,
                },
                "configuracao": {
                    "alvo": 0.05,
                    "stop": 0.04,
                    "max_dias": 10,
                    "filtro_regime": "Ibov > MM200",
                },
                "tickers": tickers_aprovados,
            }, f, indent=2)
        
        print(f"[✓] Lista salva: {caminho_aprovados}")
        print(f"\n💡 Use estes {len(tickers_aprovados)} papéis no screening diário.")
    
    else:
        print("❌ Nenhum papel aprovado nos critérios rigorosos.")
        print("\n⚠️  Isso significa que momentum com filtro de regime NÃO é robusto")
        print("    o suficiente para operação na B3.")
    
    # --- AMOSTRA DE REJEITADOS ---
    print(f"\n❌ AMOSTRA DE PAPÉIS REJEITADOS ({len(papeis_rejeitados)} total):\n")
    for ticker, motivo in papeis_rejeitados[:10]:
        print(f"  {ticker}: {motivo}")
    
    if len(papeis_rejeitados) > 10:
        print(f"  ... e mais {len(papeis_rejeitados) - 10} papéis")
    
    print(f"\n{'='*70}\n")
    
    # Salvar CSV completo
    todos_resultados = []
    for ticker, anos_data in resultados_por_ticker.items():
        for ano, metricas in anos_data.items():
            todos_resultados.append({
                "ticker": ticker,
                "ano": ano,
                **metricas
            })
    
    if todos_resultados:
        df_completo = pd.DataFrame(todos_resultados)
        caminho_csv = f"{CFG['pasta_resultados']}/validacao_individual_regime_{data_base}.csv"
        df_completo.to_csv(caminho_csv, index=False)
        print(f"[✓] Dados completos salvos: {caminho_csv}")
    
    return papeis_aprovados


def validar_walkforward_alvos_stops(data_base: date = None, max_dias_fixo: int = 10):
    """
    Valida robustez das combinações alvo/stop através de janelas anuais.
    
    Testa se a configuração "ótima" se mantém estável em diferentes
    períodos (evita overfitting ao passado).
    
    Divide o histórico em anos e testa cada combinação ano a ano.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_papeis_promissores()
    
    # Combinações a testar
    alvos_testar = [0.04, 0.05, 0.06]
    stops_testar = [0.03, 0.04, 0.05]
    
    # Janelas anuais (últimos 5-6 anos)
    ano_atual = data_base.year
    anos = list(range(ano_atual - 5, ano_atual + 1))  # 6 anos
    
    print(f"\n{'='*70}")
    print(f"VALIDAÇÃO WALK-FORWARD (janelas anuais)")
    print(f"Testando {len(alvos_testar)} alvos × {len(stops_testar)} stops em {len(anos)} anos")
    print(f"Anos: {anos}")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Resultados por [ano][combinação]
    resultados_por_ano = {ano: [] for ano in anos}
    
    for ano in anos:
        print(f"\n{'='*60}")
        print(f"ANO: {ano}")
        print(f"{'='*60}")
        
        # Define janela do ano
        inicio_ano = date(ano, 1, 1)
        fim_ano    = date(ano, 12, 31)
        
        for alvo_pct in alvos_testar:
            for stop_pct in stops_testar:
                
                # Criar estratégia temporária
                est_temp = ESTRATEGIA_MOMENTUM.copy()
                est_temp["parametros"] = est_temp["parametros"].copy()
                est_temp["parametros"]["alvo_pct"] = alvo_pct
                est_temp["parametros"]["stop_pct"] = stop_pct
                est_temp["parametros"]["max_dias"] = max_dias_fixo
                
                todos_trades = []
                
                for ticker in tickers:
                    try:
                        # Carrega histórico até fim do ano
                        df = preparar_df(ticker, fim_ano)
                        
                        # Roda backtest
                        trades = backtest_estrategia(ticker, df, est_temp)
                        
                        if trades.empty:
                            continue
                        
                        # Filtra apenas trades com entrada DENTRO do ano
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        trades_ano = trades[
                            (trades["data_entrada"] >= pd.Timestamp(inicio_ano)) &
                            (trades["data_entrada"] <= pd.Timestamp(fim_ano))
                        ]
                        
                        if not trades_ano.empty:
                            todos_trades.append(trades_ano)
                    
                    except Exception:
                        pass
                
                if not todos_trades:
                    continue
                
                df_todos = pd.concat(todos_trades, ignore_index=True)
                
                # Estatísticas
                n_total     = len(df_todos)
                n_ganhos    = (df_todos["retorno_%"] > 0).sum()
                taxa_acerto = (n_ganhos / n_total * 100) if n_total > 0 else 0
                retorno_med = df_todos["retorno_%"].mean()
                
                # Distribuição
                dist = df_todos["resultado"].value_counts()
                pct_alvo = (dist.get("alvo", 0) / n_total * 100) if n_total > 0 else 0
                pct_stop = (dist.get("stop", 0) / n_total * 100) if n_total > 0 else 0
                
                # Dias médios (estimativa)
                dias_med_exp = (
                    (pct_alvo / 100) * (max_dias_fixo * 0.6) +
                    (pct_stop / 100) * (max_dias_fixo * 0.4) +
                    ((100 - pct_alvo - pct_stop) / 100) * max_dias_fixo
                )
                
                # Retorno anualizado
                ret_anualizado = (retorno_med / 100) * (252 / dias_med_exp) * 100 if dias_med_exp > 0 else 0
                
                resultados_por_ano[ano].append({
                    "ano":              ano,
                    "alvo_%":           alvo_pct * 100,
                    "stop_%":           stop_pct * 100,
                    "n_trades":         n_total,
                    "taxa_acerto":      round(taxa_acerto, 1),
                    "retorno_med_%":    round(retorno_med, 2),
                    "ret_anualizado_%": round(ret_anualizado, 1),
                })
    
    # --- ANÁLISE CONSOLIDADA ---
    print(f"\n{'='*70}")
    print("📊 CONSOLIDAÇÃO POR ANO")
    print(f"{'='*70}\n")
    
    # Mostra top 3 de cada ano
    for ano in anos:
        if not resultados_por_ano[ano]:
            print(f"{ano}: Sem dados suficientes")
            continue
        
        df_ano = pd.DataFrame(resultados_por_ano[ano])
        df_ano = df_ano.sort_values("ret_anualizado_%", ascending=False).head(3)
        
        print(f"\n🏆 {ano} - Top 3:")
        for idx, row in df_ano.iterrows():
            pos = "🥇" if idx == 0 else ("🥈" if idx == 1 else "🥉")
            print(f"  {pos} {row['alvo_%']:.0f}%/{row['stop_%']:.0f}% | "
                  f"Ret anual: {row['ret_anualizado_%']:.1f}% | "
                  f"Taxa: {row['taxa_acerto']:.1f}% | "
                  f"Trades: {row['n_trades']}")
    
    # --- ANÁLISE DE ROBUSTEZ ---
    print(f"\n{'='*70}")
    print("🔬 ANÁLISE DE ROBUSTEZ (quantas vezes cada combinação venceu)")
    print(f"{'='*70}\n")
    
    # Conta vitórias por combinação
    vitorias = {}
    ret_por_combo = {}
    
    for ano in anos:
        if not resultados_por_ano[ano]:
            continue
        
        df_ano = pd.DataFrame(resultados_por_ano[ano])
        melhor = df_ano.sort_values("ret_anualizado_%", ascending=False).iloc[0]
        
        combo_key = f"{melhor['alvo_%']:.0f}%/{melhor['stop_%']:.0f}%"
        vitorias[combo_key] = vitorias.get(combo_key, 0) + 1
        
        # Acumula retornos de cada combo
        for _, row in df_ano.iterrows():
            key = f"{row['alvo_%']:.0f}%/{row['stop_%']:.0f}%"
            if key not in ret_por_combo:
                ret_por_combo[key] = []
            ret_por_combo[key].append(row['ret_anualizado_%'])
    
    # Ordena por número de vitórias
    vitorias_sorted = sorted(vitorias.items(), key=lambda x: x[1], reverse=True)
    
    print("Ranking por estabilidade (vitórias em diferentes anos):\n")
    for combo, n_vitorias in vitorias_sorted:
        rets = ret_por_combo.get(combo, [])
        ret_medio = np.mean(rets) if rets else 0
        ret_std   = np.std(rets) if rets else 0
        
        print(f"  {combo}: {n_vitorias} vitórias | "
              f"Ret médio: {ret_medio:.1f}% | "
              f"Desvio: ±{ret_std:.1f}%")
    
    # --- RECOMENDAÇÃO FINAL ---
    print(f"\n{'='*70}")
    print("✅ RECOMENDAÇÃO FINAL")
    print(f"{'='*70}\n")
    
    if vitorias_sorted:
        melhor_combo = vitorias_sorted[0][0]
        n_vitorias   = vitorias_sorted[0][1]
        rets_melhor  = ret_por_combo[melhor_combo]
        ret_medio    = np.mean(rets_melhor)
        ret_std      = np.std(rets_melhor)
        
        print(f"🏆 CONFIGURAÇÃO MAIS ROBUSTA: {melhor_combo}")
        print(f"   Venceu em {n_vitorias} de {len([a for a in anos if resultados_por_ano[a]])} anos testados")
        print(f"   Retorno médio anual: {ret_medio:.1f}%")
        print(f"   Desvio padrão: ±{ret_std:.1f}%")
        print()
        
        # Avalia estabilidade
        cv = (ret_std / ret_medio) if ret_medio > 0 else 999  # coeficiente de variação
        
        if cv < 0.15:
            print("✅ ESTABILIDADE EXCELENTE:")
            print("   Retornos consistentes entre anos. Configuração confiável.")
        elif cv < 0.30:
            print("⚠️  ESTABILIDADE MODERADA:")
            print("   Alguma variação entre anos, mas ainda aceitável.")
        else:
            print("🔴 ESTABILIDADE BAIXA:")
            print("   Alta variação entre anos. Resultados pouco previsíveis.")
            print("   Considere simplificar critérios ou usar configuração mais conservadora.")
        
        # Compara com 5%/4%
        if "5%/4%" in ret_por_combo:
            rets_54 = ret_por_combo["5%/4%"]
            ret_54  = np.mean(rets_54)
            std_54  = np.std(rets_54)
            
            print()
            print(f"📊 COMPARAÇÃO COM 5%/4% (configuração atual):")
            print(f"   Retorno médio: {ret_54:.1f}%")
            print(f"   Desvio padrão: ±{std_54:.1f}%")
            
            if melhor_combo == "5%/4%":
                print()
                print("✅ A configuração atual (5%/4%) É a mais robusta.")
                print("   Mantenha como está.")
            else:
                diff = ret_medio - ret_54
                print()
                print(f"   Diferença de retorno: {diff:+.1f}pp em favor de {melhor_combo}")
                
                if diff > 10 and cv < 0.25:
                    print(f"   ✅ Vale trocar para {melhor_combo} (ganho significativo e estável)")
                elif diff > 5:
                    print(f"   🤔 Considere trocar para {melhor_combo} (ganho moderado)")
                else:
                    print(f"   ⚠️  Mantenha 5%/4% (diferença marginal)")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar CSV consolidado
    todos_resultados = []
    for ano in anos:
        todos_resultados.extend(resultados_por_ano[ano])
    
    df_final = pd.DataFrame(todos_resultados)
    caminho = f"{CFG['pasta_resultados']}/walkforward_alvos_stops_{data_base}.csv"
    df_final.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_final


def testar_alvos_e_stops(data_base: date = None, max_dias_fixo: int = 10):
    """
    Testa múltiplas combinações de alvo e stop para encontrar
    o melhor balanço entre retorno e risco.
    
    Para cada combinação calcula:
    - Taxa de acerto
    - Retorno médio
    - Retorno anualizado
    - Distribuição alvo/stop
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_papeis_promissores()
    
    # Combinações a testar
    alvos_testar = [0.04, 0.05, 0.06]  # 4%, 5%, 6%
    stops_testar = [0.03, 0.04, 0.05]  # 3%, 4%, 5%
    
    print(f"\n{'='*70}")
    print(f"TESTE DE ALVOS E STOPS  |  {len(tickers)} tickers  |  max_dias={max_dias_fixo}")
    print(f"Testando {len(alvos_testar)} alvos × {len(stops_testar)} stops = {len(alvos_testar) * len(stops_testar)} combinações")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    resultados = []
    
    for alvo_pct in alvos_testar:
        for stop_pct in stops_testar:
            combo_nome = f"alvo_{alvo_pct*100:.0f}_stop_{stop_pct*100:.0f}"
            print(f"\n--- Testando alvo={alvo_pct*100:.1f}% / stop={stop_pct*100:.1f}% ---")
            
            # Criar estratégia temporária
            est_temp = ESTRATEGIA_MOMENTUM.copy()
            est_temp["parametros"] = est_temp["parametros"].copy()
            est_temp["parametros"]["alvo_pct"] = alvo_pct
            est_temp["parametros"]["stop_pct"] = stop_pct
            est_temp["parametros"]["max_dias"] = max_dias_fixo
            
            todos_trades = []
            
            for ticker in tickers:
                try:
                    df = preparar_df(ticker, data_base)
                    trades = backtest_estrategia(ticker, df, est_temp)
                    if not trades.empty:
                        todos_trades.append(trades)
                except Exception:
                    pass
            
            if not todos_trades:
                print(f"  Nenhum trade.")
                continue
            
            df_todos = pd.concat(todos_trades, ignore_index=True)
            
            # Estatísticas
            n_total     = len(df_todos)
            n_ganhos    = (df_todos["retorno_%"] > 0).sum()
            taxa_acerto = (n_ganhos / n_total * 100) if n_total > 0 else 0
            retorno_med = df_todos["retorno_%"].mean()
            
            # Distribuição por resultado
            dist = df_todos["resultado"].value_counts()
            n_alvo  = dist.get("alvo", 0)
            n_stop  = dist.get("stop", 0)
            n_tempo = dist.get("tempo", 0)
            
            pct_alvo  = (n_alvo / n_total * 100) if n_total > 0 else 0
            pct_stop  = (n_stop / n_total * 100) if n_total > 0 else 0
            pct_tempo = (n_tempo / n_total * 100) if n_total > 0 else 0
            
            # Dias médios de exposição (estimativa)
            # Alvo = 60% do prazo, Stop = 40% do prazo, Tempo = 100% do prazo
            dias_med_alvo  = max_dias_fixo * 0.6
            dias_med_stop  = max_dias_fixo * 0.4
            dias_med_tempo = max_dias_fixo
            
            dias_exposicao_media = (
                (pct_alvo / 100) * dias_med_alvo +
                (pct_stop / 100) * dias_med_stop +
                (pct_tempo / 100) * dias_med_tempo
            )
            
            # Retorno anualizado
            if dias_exposicao_media > 0:
                ret_anualizado = (retorno_med / 100) * (252 / dias_exposicao_media) * 100
            else:
                ret_anualizado = 0
            
            # R:R
            rr = alvo_pct / stop_pct if stop_pct > 0 else 0
            
            resultados.append({
                "alvo_%":           alvo_pct * 100,
                "stop_%":           stop_pct * 100,
                "r:r":              round(rr, 2),
                "n_trades":         n_total,
                "taxa_acerto":      round(taxa_acerto, 1),
                "retorno_med_%":    round(retorno_med, 2),
                "ret_anualizado_%": round(ret_anualizado, 1),
                "dias_med_exp":     round(dias_exposicao_media, 1),
                "pct_alvo":         round(pct_alvo, 1),
                "pct_stop":         round(pct_stop, 1),
                "pct_tempo":        round(pct_tempo, 1),
            })
            
            print(f"  Trades: {n_total} | Acerto: {taxa_acerto:.1f}% | Ret médio: {retorno_med:.2f}%")
            print(f"  Ret anualizado: {ret_anualizado:.1f}% | Alvo/Stop/Tempo: {pct_alvo:.0f}%/{pct_stop:.0f}%/{pct_tempo:.0f}%")
    
    # --- COMPARAÇÃO ---
    print(f"\n{'='*70}")
    print("📊 COMPARAÇÃO DE COMBINAÇÕES ALVO/STOP")
    print(f"{'='*70}\n")
    
    df_result = pd.DataFrame(resultados)
    
    # Ordena por retorno anualizado
    df_result = df_result.sort_values("ret_anualizado_%", ascending=False)
    
    print(df_result.to_string(index=False))
    
    # --- ANÁLISE ---
    print(f"\n{'='*70}")
    print("💡 ANÁLISE E RECOMENDAÇÃO")
    print(f"{'='*70}\n")
    
    melhor = df_result.iloc[0]
    
    print(f"🏆 MELHOR COMBINAÇÃO (maior retorno anualizado):")
    print(f"   Alvo: {melhor['alvo_%']:.0f}% | Stop: {melhor['stop_%']:.0f}% | R:R: 1:{melhor['r:r']:.2f}")
    print(f"   Taxa de acerto: {melhor['taxa_acerto']:.1f}%")
    print(f"   Retorno médio: {melhor['retorno_med_%']:.2f}%")
    print(f"   Retorno anualizado: {melhor['ret_anualizado_%']:.1f}%")
    print(f"   Dias médios exposição: {melhor['dias_med_exp']:.1f}")
    print()
    
    # Comparar com configuração atual (5%/4%)
    atual = df_result[(df_result['alvo_%'] == 5.0) & (df_result['stop_%'] == 4.0)]
    
    if not atual.empty:
        atual = atual.iloc[0]
        diff_ret = melhor['ret_anualizado_%'] - atual['ret_anualizado_%']
        diff_taxa = melhor['taxa_acerto'] - atual['taxa_acerto']
        
        print(f"📊 COMPARAÇÃO COM CONFIGURAÇÃO ATUAL (5%/4%):")
        print(f"   Ganho de retorno anualizado: {diff_ret:+.1f}pp")
        print(f"   Variação taxa de acerto: {diff_taxa:+.1f}pp")
        print()
        
        if diff_ret > 5:
            print(f"✅ RECOMENDAÇÃO: Troque para {melhor['alvo_%']:.0f}%/{melhor['stop_%']:.0f}%")
            print(f"   Ganho significativo de {diff_ret:.1f}pp no retorno anualizado.")
        elif diff_ret < -5:
            print(f"⚠️  CUIDADO: Melhor combinação tem retorno {abs(diff_ret):.1f}pp MENOR que atual.")
            print(f"   Considere manter 5%/4% se valoriza estabilidade.")
        else:
            print(f"🤷 INDIFERENTE: Diferença de apenas {diff_ret:.1f}pp.")
            print(f"   Pode manter 5%/4% ou trocar sem grande impacto.")
    
    # Observações sobre R:R
    print()
    print("📐 OBSERVAÇÃO SOBRE R:R (risk:reward):")
    print("   R:R > 1.5 = Alvo bem maior que stop (mais agressivo, precisa alta taxa acerto)")
    print("   R:R 1.2-1.5 = Balanceado (recomendado para swing trade)")
    print("   R:R < 1.2 = Stop largo demais para o alvo (perde muito quando erra)")
    
    print()
    print(f"{'='*70}\n")
    
    # Salvar
    caminho = f"{CFG['pasta_resultados']}/teste_alvos_stops_{data_base}.csv"
    df_result.to_csv(caminho, index=False)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_result


def testar_multiplos_prazos(data_base: date = None):
    """
    Roda backtest com diferentes valores de max_dias para encontrar
    o prazo ótimo de saída por tempo.
    
    Compara: taxa de acerto, retorno médio e % de saídas por tempo.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_papeis_promissores()
    prazos_testar = [10, 15, 20, 25, 30]
    
    print(f"\n{'='*60}")
    print(f"TESTE DE PRAZOS MÁXIMOS  |  {len(tickers)} tickers")
    print(f"Testando: {prazos_testar} dias úteis")
    print(f"{'='*60}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    resultados_por_prazo = {}
    
    for max_dias in prazos_testar:
        print(f"\n--- Testando max_dias = {max_dias} ---")
        
        # Cria estratégia temporária com esse prazo
        estrategia_temp = ESTRATEGIA_MOMENTUM.copy()
        estrategia_temp["parametros"] = estrategia_temp["parametros"].copy()
        estrategia_temp["parametros"]["max_dias"] = max_dias
        
        todos_trades = []
        
        for ticker in tickers:
            print(f"  {ticker}...", end=" ", flush=True)
            try:
                df = preparar_df(ticker, data_base)
                trades = backtest_estrategia(ticker, df, estrategia_temp)
                
                if not trades.empty:
                    todos_trades.append(trades)
                    print("ok")
                else:
                    print("sem trades")
            except Exception as e:
                print(f"erro: {e}")
        
        if not todos_trades:
            print(f"  Nenhum trade para max_dias={max_dias}")
            continue
        
        df_todos = pd.concat(todos_trades, ignore_index=True)
        
        # Estatísticas
        n_total      = len(df_todos)
        n_ganhos     = (df_todos["retorno_%"] > 0).sum()
        taxa_acerto  = (n_ganhos / n_total * 100) if n_total > 0 else 0
        retorno_med  = df_todos["retorno_%"].mean()
        retorno_tot  = df_todos["retorno_%"].sum()
        
        # Distribuição por tipo de saída
        dist_resultado = df_todos["resultado"].value_counts()
        n_tempo = dist_resultado.get("tempo", 0)
        n_alvo  = dist_resultado.get("alvo", 0)
        n_stop  = dist_resultado.get("stop", 0)
        
        pct_tempo = (n_tempo / n_total * 100) if n_total > 0 else 0
        pct_alvo  = (n_alvo / n_total * 100) if n_total > 0 else 0
        pct_stop  = (n_stop / n_total * 100) if n_total > 0 else 0
        
        resultados_por_prazo[max_dias] = {
            "n_trades":     n_total,
            "taxa_acerto":  taxa_acerto,
            "retorno_med":  retorno_med,
            "retorno_tot":  retorno_tot,
            "pct_tempo":    pct_tempo,
            "pct_alvo":     pct_alvo,
            "pct_stop":     pct_stop,
            "n_tempo":      n_tempo,
            "n_alvo":       n_alvo,
            "n_stop":       n_stop,
        }
        
        print(f"\n  Resultados:")
        print(f"    Trades       : {n_total}")
        print(f"    Taxa acerto  : {taxa_acerto:.1f}%")
        print(f"    Retorno médio: {retorno_med:.2f}%")
        print(f"    Saídas por tempo: {n_tempo} ({pct_tempo:.1f}%)")
        print(f"    Saídas por alvo : {n_alvo} ({pct_alvo:.1f}%)")
        print(f"    Saídas por stop : {n_stop} ({pct_stop:.1f}%)")
    
    # --- COMPARAÇÃO LADO A LADO ---
    print(f"\n{'='*60}")
    print("📊 COMPARAÇÃO DE PRAZOS MÁXIMOS")
    print(f"{'='*60}\n")
    
    # Tabela comparativa
    df_comp = pd.DataFrame(resultados_por_prazo).T
    df_comp.index.name = "max_dias"
    
    print(df_comp[["n_trades", "taxa_acerto", "retorno_med", "pct_tempo", "pct_alvo", "pct_stop"]].to_string())
    
    # --- ANÁLISE E RECOMENDAÇÃO ---
    print(f"\n{'='*60}")
    print("💡 ANÁLISE")
    print(f"{'='*60}\n")
    
    # Encontra o prazo com melhor balanço
    # Critérios: taxa de acerto alta + retorno médio alto + poucas saídas por tempo
    df_comp["score"] = (
        df_comp["taxa_acerto"] * 0.4 +           # 40% peso: taxa de acerto
        df_comp["retorno_med"] * 10 * 0.4 +      # 40% peso: retorno médio (escalado)
        (100 - df_comp["pct_tempo"]) * 0.2       # 20% peso: menos saídas por tempo
    )
    
    melhor_prazo = df_comp["score"].idxmax()
    
    print("Critérios de avaliação:")
    print("  • Taxa de acerto: quanto maior, melhor")
    print("  • Retorno médio: quanto maior, melhor")
    print("  • % saídas por tempo: quanto menor, melhor (indica que alvo/stop definem a saída)")
    print()
    
    print(f"🏆 PRAZO RECOMENDADO: {melhor_prazo} dias úteis")
    print()
    print("Justificativa:")
    
    resultado_melhor = resultados_por_prazo[melhor_prazo]
    print(f"  • Taxa de acerto: {resultado_melhor['taxa_acerto']:.1f}%")
    print(f"  • Retorno médio: {resultado_melhor['retorno_med']:.2f}%")
    print(f"  • Saídas por tempo: {resultado_melhor['pct_tempo']:.1f}% ({resultado_melhor['n_tempo']} de {resultado_melhor['n_trades']} trades)")
    print()
    
    # Observações adicionais
    if resultado_melhor['pct_tempo'] > 15:
        print("⚠️  OBSERVAÇÃO:")
        print(f"   Mais de 15% dos trades saem por tempo ({resultado_melhor['pct_tempo']:.1f}%).")
        print("   Considere:")
        print("   - Aumentar o prazo para 25-30 dias, OU")
        print("   - Revisar se alvo/stop estão muito distantes")
    elif resultado_melhor['pct_tempo'] < 5:
        print("✅ OBSERVAÇÃO:")
        print(f"   Apenas {resultado_melhor['pct_tempo']:.1f}% dos trades saem por tempo.")
        print("   O prazo está bem calibrado — alvo e stop definem quase todas as saídas.")
    
    print()
    print(f"{'='*60}\n")
    
    # Salvar CSV
    caminho = f"{CFG['pasta_resultados']}/teste_prazos_{data_base}.csv"
    df_comp.to_csv(caminho)
    print(f"[✓] Resultados salvos: {caminho}")
    
    return df_comp


def backtest_estrategia(ticker: str, df: pd.DataFrame, est: dict) -> pd.DataFrame:
    p      = est["parametros"]
    trades = []
    
    em_posicao    = False
    preco_entrada = None
    data_entrada  = None
    dias_posicao  = 0
    
    for idx, row in df.iterrows():
        if em_posicao:
            dias_posicao += 1
            high  = _val(row, "high")
            low   = _val(row, "low")
            close = _val(row, "close")
            open  = _val(row, "open")
            
            alvo_preco = preco_entrada * (1 + p["alvo_pct"])
            stop_preco = preco_entrada * (1 - p["stop_pct"])
            
            resultado   = None
            preco_saida = close
            
            if low <= stop_preco:
                resultado   = "stop"
                #preco_saida = stop_preco
                preco_saida = min(stop_preco, open)
            elif high >= alvo_preco:
                resultado   = "alvo"
                preco_saida = alvo_preco
            elif dias_posicao >= p["max_dias"]:
                resultado   = "tempo"
                preco_saida = close
            
            if resultado:
                #retorno_pct = (preco_saida / preco_entrada - 1) * 100
                #retorno_pct = 100 * ( (preco_saida / preco_entrada) - 1 - 0.004 )  # considerando 0.4% de custos totais
             
                # Forma mais rigorosa:
                taxas = 0.004
                retorno_pct = 100 * ( (preco_saida * (1 - taxas/2)) / (preco_entrada * (1 + taxas/2)) - 1 )

                trades.append({
                    "ticker":        ticker,
                    "estrategia":    est["nome"],
                    "data_entrada":  str(data_entrada),
                    "data_saida":    str(idx),
                    "dias":          dias_posicao,
                    "preco_entrada": round(preco_entrada, 2),
                    "preco_saida":   round(preco_saida, 2),
                    "retorno_%":     round(retorno_pct, 2),
                    "resultado":     resultado,
                })
                em_posicao = False
        
        else:
            try:
                disparou = est["fn_sinal"](row, p)
            except Exception:
                disparou = False
            
            if disparou:
                em_posicao    = True
                preco_entrada = _val(row, "close")
                data_entrada  = idx
                dias_posicao  = 0
    
    return pd.DataFrame(trades)


def backtest_estratificado_por_alerta(data_base: date = None):
    """
    Roda backtest nos papéis promissores e estratifica os resultados
    por classificação de alerta (🟢 BOM, 🟡 MODERADO, 🔴 ATENÇÃO).
    
    Valida se a classificação tem valor preditivo real.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_papeis_promissores()
    
    print(f"\n{'='*60}")
    print(f"BACKTEST ESTRATIFICADO  |  {len(tickers)} tickers  |  até {data_base}")
    print(f"{'='*60}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Armazena todos os trades com classificação de alerta
    todos_trades_com_alerta = []
    
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = preparar_df(ticker, data_base)
            
            # --- Backtest normal ---
            trades = backtest_estrategia(ticker, df, ESTRATEGIA_MOMENTUM)
            
            if trades.empty:
                print("sem trades")
                continue
            
            # --- Para cada trade, classificar o alerta que teria no dia da entrada ---
            for idx, trade in trades.iterrows():
                data_entrada_str = trade["data_entrada"]
                
                # Busca a linha do df na data de entrada
                try:
                    data_entrada_obj = pd.to_datetime(data_entrada_str).date()
                    row_entrada = df.loc[data_entrada_obj]
                    
                    # Classifica o sinal que gerou esse trade
                    # (precisa recriar o dict como em avaliar_sinais_no_dia)
                    sinal_dict = {
                        "rsi":         _val(row_entrada, "rsi"),
                        "adx":         _val(row_entrada, "ADX_14"),
                        "vol_rel":     _val(row_entrada, "vol_relativo"),
                        "ret_5d_%":    _retorno_nd(row_entrada, 5) * 100,
                        "norm_mm200":  _val(row_entrada, "close_norm_mm200"),
                    }
                    
                    alerta = _classificar_sinal(pd.Series(sinal_dict))
                    
                    # Extrai apenas a classificação (✅ ou ❌)
                    if alerta.startswith("✅"):
                        classe = "✅ OPERAR"
                    elif alerta.startswith("❌"):
                        classe = "❌ EVITAR"
                    else:
                        classe = "SEM CLASSIFICAÇÃO"
                    
                    # Adiciona a classificação ao trade
                    trade_com_alerta = trade.copy()
                    trade_com_alerta["alerta"] = classe
                    trade_com_alerta["alerta_completo"] = alerta
                    todos_trades_com_alerta.append(trade_com_alerta)
                    
                except Exception as e_class:
                    # Se falhar classificação, marca como "erro"
                    trade_com_alerta = trade.copy()
                    trade_com_alerta["alerta"] = "ERRO CLASSIFICAÇÃO"
                    trade_com_alerta["alerta_completo"] = str(e_class)
                    todos_trades_com_alerta.append(trade_com_alerta)
            
            print(f"{len(trades)} trades")
            
        except Exception as e:
            print(f"ERRO: {e}")
    
    if not todos_trades_com_alerta:
        print("\n⚠️  Nenhum trade simulado.")
        return
    
    df_todos = pd.DataFrame(todos_trades_com_alerta)
    
    # Salva CSV completo
    caminho_completo = f"{CFG['pasta_resultados']}/backtest_estratificado_{data_base}.csv"
    df_todos.to_csv(caminho_completo, index=False)
    print(f"\n[✓] Trades salvos: {caminho_completo}")
    
    # --- ANÁLISE ESTRATIFICADA ---
    print(f"\n{'='*60}")
    print("📊 ANÁLISE POR CLASSIFICAÇÃO BINÁRIA")
    print(f"{'='*60}\n")
    
    for classe in ["✅ OPERAR", "❌ EVITAR"]:
        df_classe = df_todos[df_todos["alerta"] == classe]
        
        if df_classe.empty:
            print(f"{classe}: Nenhum trade nesta categoria\n")
            continue
        
        n_trades    = len(df_classe)
        n_ganhos    = (df_classe["retorno_%"] > 0).sum()
        n_perdas    = (df_classe["retorno_%"] <= 0).sum()
        taxa_acerto = (n_ganhos / n_trades * 100) if n_trades > 0 else 0
        
        retorno_med   = df_classe["retorno_%"].mean()
        retorno_total = df_classe["retorno_%"].sum()
        retorno_std   = df_classe["retorno_%"].std()
        
        melhor_trade = df_classe["retorno_%"].max()
        pior_trade   = df_classe["retorno_%"].min()
        
        # Distribuição por resultado
        por_resultado = df_classe["resultado"].value_counts().to_dict()
        
        print(f"{classe}:")
        print(f"  Trades        : {n_trades}")
        print(f"  Taxa acerto   : {taxa_acerto:.1f}% ({n_ganhos} ganhos / {n_perdas} perdas)")
        print(f"  Retorno médio : {retorno_med:.2f}%")
        print(f"  Retorno total : {retorno_total:.1f}%")
        print(f"  Desvio padrão : {retorno_std:.2f}%")
        print(f"  Melhor trade  : {melhor_trade:.2f}%")
        print(f"  Pior trade    : {pior_trade:.2f}%")
        print(f"  Por resultado : {por_resultado}")
        print()
    
    # --- COMPARAÇÃO ESTATÍSTICA ---
    print(f"{'='*60}")
    print("📈 COMPARAÇÃO BINÁRIA (validação do filtro)")
    print(f"{'='*60}\n")
    
    df_operar = df_todos[df_todos["alerta"] == "✅ OPERAR"]
    df_evitar = df_todos[df_todos["alerta"] == "❌ EVITAR"]
    
    ret_operar = df_operar["retorno_%"].mean() if not df_operar.empty else 0
    ret_evitar = df_evitar["retorno_%"].mean() if not df_evitar.empty else 0
    
    taxa_operar = (df_operar["retorno_%"] > 0).mean() * 100 if not df_operar.empty else 0
    taxa_evitar = (df_evitar["retorno_%"] > 0).mean() * 100 if not df_evitar.empty else 0
    
    print("Retorno médio:")
    print(f"  ✅ OPERAR : {ret_operar:.2f}%")
    print(f"  ❌ EVITAR : {ret_evitar:.2f}%")
    print(f"  Diferença : {ret_operar - ret_evitar:+.2f}% a favor de OPERAR")
    print()
    
    print("Taxa de acerto:")
    print(f"  ✅ OPERAR : {taxa_operar:.1f}%")
    print(f"  ❌ EVITAR : {taxa_evitar:.1f}%")
    print(f"  Diferença : {taxa_operar - taxa_evitar:+.1f}pp a favor de OPERAR")
    print()
    
    # Interpretação
    print(f"{'='*60}")
    print("✅ INTERPRETAÇÃO")
    print(f"{'='*60}\n")
    
    # Valida se OPERAR é estatisticamente superior a EVITAR
    superioridade_retorno = ret_operar > ret_evitar + 0.5  # pelo menos 0.5pp melhor
    superioridade_taxa    = taxa_operar > taxa_evitar + 5  # pelo menos 5pp melhor
    
    if superioridade_retorno and superioridade_taxa:
        print("✅ VALIDAÇÃO POSITIVA:")
        print("   O filtro binário TEM valor preditivo claro.")
        print(f"   Sinais ✅ OPERAR superam ❌ EVITAR tanto em retorno (+{ret_operar - ret_evitar:.1f}pp)")
        print(f"   quanto em taxa de acerto (+{taxa_operar - taxa_evitar:.1f}pp).")
        print()
        print("   💡 RECOMENDAÇÃO: Use o filtro com confiança.")
        print("      - Opere APENAS sinais ✅ OPERAR")
        print("      - Ignore completamente sinais ❌ EVITAR")
        print(f"      - Expectativa de taxa de acerto: ~{taxa_operar:.0f}%")
    
    elif superioridade_retorno or superioridade_taxa:
        print("⚠️  VALIDAÇÃO PARCIAL:")
        print("   O filtro mostra algum valor preditivo, mas não é conclusivo.")
        print()
        print("   💡 RECOMENDAÇÃO: Use o filtro, mas com cautela.")
        print("      - Priorize sinais ✅ OPERAR")
        print("      - Considere sinais ❌ EVITAR apenas em contexto muito favorável")
    
    else:
        print("❌ VALIDAÇÃO NEGATIVA:")
        print("   O filtro binário NÃO demonstra valor preditivo claro.")
        print()
        print("   💡 RECOMENDAÇÃO: Refine os critérios de classificação")
        print("      ou opere todos os sinais igualmente.")
    
    print()
    print(f"{'='*60}\n")
    
    return df_todos
    p      = est["parametros"]
    trades = []
    
    em_posicao    = False
    preco_entrada = None
    data_entrada  = None
    dias_posicao  = 0
    
    for idx, row in df.iterrows():
        if em_posicao:
            dias_posicao += 1
            high  = _val(row, "high")
            low   = _val(row, "low")
            close = _val(row, "close")
            
            alvo_preco = preco_entrada * (1 + p["alvo_pct"])
            stop_preco = preco_entrada * (1 - p["stop_pct"])
            
            resultado   = None
            preco_saida = close
            
            if low <= stop_preco:
                resultado   = "stop"
                preco_saida = stop_preco
            elif high >= alvo_preco:
                resultado   = "alvo"
                preco_saida = alvo_preco
            elif dias_posicao >= p["max_dias"]:
                resultado   = "tempo"
                preco_saida = close
            
            if resultado:
                retorno_pct = (preco_saida / preco_entrada - 1) * 100
                trades.append({
                    "ticker":        ticker,
                    "estrategia":    est["nome"],
                    "data_entrada":  str(data_entrada),
                    "data_saida":    str(idx),
                    "dias":          dias_posicao,
                    "preco_entrada": round(preco_entrada, 2),
                    "preco_saida":   round(preco_saida, 2),
                    "retorno_%":     round(retorno_pct, 2),
                    "resultado":     resultado,
                })
                em_posicao = False
        
        else:
            try:
                disparou = est["fn_sinal"](row, p)
            except Exception:
                disparou = False
            
            if disparou:
                em_posicao    = True
                preco_entrada = _val(row, "close")
                data_entrada  = idx
                dias_posicao  = 0
    
    return pd.DataFrame(trades)


def calibrar_papeis_promissores(data_base: date = None):
    """
    NOTA: Esta função é legado do sistema de Momentum.
    
    Para Mean Reversion LIGHT, NÃO é necessário "calibrar" papéis promissores.
    O screening diário roda em TODO o universo (69 papéis).
    
    Mantida para compatibilidade e testes futuros de estratégias
    que possam se beneficiar de pré-filtragem.
    """
    """
    Roda backtest em TODO o universo, identifica papéis com boa performance,
    e salva a lista de promissores.
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    
    print(f"\n{'='*60}")
    print(f"CALIBRAÇÃO  |  {len(tickers)} tickers  |  até {data_base}")
    print(f"{'='*60}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    metricas_por_ticker = {}
    todos_trades = []
    
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = preparar_df(ticker, data_base)
            trades = backtest_estrategia(ticker, df, ESTRATEGIA_MOMENTUM)
            
            if trades.empty:
                print("sem trades")
                metricas_por_ticker[ticker] = {"n_trades": 0}
                continue
            
            n_trades    = len(trades)
            taxa_acerto = (trades["retorno_%"] > 0).sum() / n_trades
            retorno_med = trades["retorno_%"].mean() / 100  # como fração
            
            metricas_por_ticker[ticker] = {
                "n_trades":     n_trades,
                "taxa_acerto":  round(taxa_acerto, 3),
                "retorno_medio": round(retorno_med, 4),
            }
            
            print(f"{n_trades} trades | acerto={taxa_acerto:.1%} | ret_med={retorno_med:.2%}")
            todos_trades.append(trades)
            
        except Exception as e:
            print(f"ERRO: {e}")
            metricas_por_ticker[ticker] = {"erro": str(e)}
    
    # Identificar promissores
    promissores = []
    for ticker, m in metricas_por_ticker.items():
        if (
            m.get("n_trades", 0) >= CFG["min_trades"]
            and m.get("taxa_acerto", 0) >= CFG["min_taxa_acerto"]
            and m.get("retorno_medio", 0) >= CFG["min_retorno_medio"]
        ):
            promissores.append(ticker)
    
    salvar_papeis_promissores(promissores, metricas_por_ticker)
    
    # Salvar backtest completo
    if todos_trades:
        df_todos = pd.concat(todos_trades, ignore_index=True)
        caminho_bt = f"{CFG['pasta_resultados']}/backtest_calibracao_{data_base}.csv"
        df_todos.to_csv(caminho_bt, index=False)
        
        # Resumo geral
        print(f"\n{'='*60}")
        print("RESUMO GERAL DO BACKTEST")
        print(f"{'='*60}")
        n_total     = len(df_todos)
        taxa_global = (df_todos["retorno_%"] > 0).sum() / n_total * 100
        ret_medio   = df_todos["retorno_%"].mean()
        ret_total   = df_todos["retorno_%"].sum()
        
        print(f"  Trades totais  : {n_total}")
        print(f"  Taxa de acerto : {taxa_global:.1f}%")
        print(f"  Retorno médio  : {ret_medio:.2f}%")
        print(f"  Retorno total  : {ret_total:.1f}%")
        print(f"\n  Papéis promissores: {len(promissores)} de {len(tickers)}")
        print(f"  {promissores}")
        print(f"\n[✓] Backtest salvo: {caminho_bt}")
    
    return promissores, metricas_por_ticker


# ---------------------------------------------------------------------------
# Avaliação de sinais (screening diário)
# ---------------------------------------------------------------------------

def avaliar_sinais_no_dia(ticker: str, df: pd.DataFrame, data_base: date) -> list[dict]:
    row   = df.iloc[-1]
    close = _val(row, "close")
    
    sinais = []
    for est in ESTRATEGIAS_ATIVAS:
        p = est["parametros"]
        try:
            disparou = est["fn_sinal"](row, p)
        except Exception as e:
            print(f"    [ERRO fn_sinal {est['nome']}]: {e}")
            disparou = False
        
        if not disparou:
            continue
        
        alvo = round(close * (1 + p["alvo_pct"]), 2) if not np.isnan(close) else None
        stop = round(close * (1 - p["stop_pct"]), 2) if not np.isnan(close) else None
        rr   = round(p["alvo_pct"] / p["stop_pct"], 1)
        
        # --- CÁLCULO DE POSIÇÃO ---
        capital_risco = CFG["capital_disponivel"] * CFG["risco_por_trade_pct"]
        distancia_stop = close - stop if (close and stop) else 0
        
        if distancia_stop > 0:
            qtd_acoes = int(capital_risco / distancia_stop)
            valor_posicao = qtd_acoes * close
            pct_capital = (valor_posicao / CFG["capital_disponivel"]) * 100
        else:
            qtd_acoes = 0
            valor_posicao = 0
            pct_capital = 0
        
        sinais.append({
            "ticker":      ticker,
            "estrategia":  est["nome"],
            "data":        str(data_base),
            "close":       round(close, 2) if not np.isnan(close) else None,
            "alvo":        alvo,
            "stop":        stop,
            "r:r":         f"1:{rr}",
            "max_dias":    p["max_dias"],
            # Gestão de posição
            "qtd":         qtd_acoes,
            "valor_R$":    round(valor_posicao, 2),
            "pct_capital": round(pct_capital, 1),
            # Indicadores
            "rsi":         round(_val(row, "rsi"), 1),
            "adx":         round(_val(row, "ADX_14"), 1),
            "vol_rel":     round(_val(row, "vol_relativo"), 2),
            "ret_5d_%":    round(_retorno_nd(row, 5) * 100, 2),
            "ret_20d_%":   round(_retorno_nd(row, 20) * 100, 2),
            "norm_mm200":  round(_val(row, "close_norm_mm200"), 3),
        })
    
    return sinais


def _gerar_relatorio_texto(df_sinais: pd.DataFrame, data_base: date, n_evitar: int = 0) -> str:
    """Gera relatório em texto puro (para terminal e corpo do email)."""
    linhas = []
    linhas.append("=" * 80)
    linhas.append(f"  🎯  SINAIS PARA {data_base + timedelta(days=1)} (COMPRA NA ABERTURA)")
    linhas.append("=" * 80)
    linhas.append("")
    
    # Cabeçalho de capital
    linhas.append(f"💰 CAPITAL DISPONÍVEL: R$ {CFG['capital_disponivel']:,.2f}")
    linhas.append(f"   Risco por trade: {CFG['risco_por_trade_pct']*100:.1f}% = R$ {CFG['capital_disponivel'] * CFG['risco_por_trade_pct']:,.2f}")
    linhas.append(f"   Máx posições simultâneas: {CFG['max_posicoes_simultaneas']}")
    linhas.append("")
    
    if df_sinais.empty and n_evitar == 0:
        linhas.append("⚠️  Nenhum sinal hoje.")
        return "\n".join(linhas)
    
    if df_sinais.empty and n_evitar > 0:
        linhas.append(f"⚠️  Nenhum sinal ✅ OPERAR hoje.")
        linhas.append(f"    {n_evitar} {'sinal' if n_evitar == 1 else 'sinais'} ❌ EVITAR (filtrado{'s' if n_evitar > 1 else ''})")
        linhas.append("")
        linhas.append("💡 Recomendação: Não operar hoje. Aguarde sinais de melhor qualidade.")
        return "\n".join(linhas)
    
    # Tabela de sinais (só OPERAR)
    linhas.append(df_sinais.to_string(index=False))
    linhas.append("")
    
    # Rodapé explicativo
    linhas.append("=" * 80)
    linhas.append("📊 INFORMAÇÕES")
    linhas.append("=" * 80)
    linhas.append("")
    
    linhas.append(f"✅ Sinais operáveis: {len(df_sinais)}")
    if n_evitar > 0:
        linhas.append(f"❌ Sinais filtrados: {n_evitar} (não exibidos - consulte CSV se necessário)")
    linhas.append("")
    
    linhas.append("🎯 GESTÃO DE POSIÇÃO:")
    linhas.append("  • qtd        = quantidade de ações a comprar")
    linhas.append("  • valor_R$   = valor total da posição")
    linhas.append("  • pct_capital = % do capital que essa posição representa")
    linhas.append("")
    
    linhas.append("📈 COLUNAS:")
    linhas.append("  • close      = preço de referência (compra próximo disso)")
    linhas.append("  • alvo       = venda com lucro (+5%)")
    linhas.append("  • stop       = venda com prejuízo (-4%)")
    linhas.append("  • r:r        = ratio risco:retorno (1:1.2)")
    linhas.append("  • max_dias   = sai no dia 20 se não atingir alvo/stop")
    linhas.append("  • rsi/adx/vol_rel = indicadores técnicos de validação")
    linhas.append("  • classificacao = ✅ OPERAR (0-1 alertas) | taxa de acerto esperada: ~78%")
    linhas.append("")
    
    # Recomendação baseada em quantidade
    n_sinais = len(df_sinais)
    if n_sinais == 1:
        linhas.append("💡 RECOMENDAÇÃO:")
        linhas.append("   1 sinal disponível. Opere se tiver capital disponível.")
    elif n_sinais <= CFG["max_posicoes_simultaneas"]:
        linhas.append("💡 RECOMENDAÇÃO:")
        linhas.append(f"   {n_sinais} sinais. Pode operar todos se tiver capital.")
    else:
        linhas.append("💡 RECOMENDAÇÃO:")
        linhas.append(f"   {n_sinais} sinais (mais que o máx recomendado de {CFG['max_posicoes_simultaneas']}).")
        linhas.append(f"   Priorize os {CFG['max_posicoes_simultaneas']} primeiros ou escolha por maior ret_20d_%.")
    
    linhas.append("")
    linhas.append("=" * 80)
    
    return "\n".join(linhas)


def _gerar_relatorio_html(df_sinais: pd.DataFrame, data_base: date, n_evitar: int = 0) -> str:
    """Gera relatório em HTML para email."""
    html = []
    html.append("<html><head><meta charset='utf-8'></head><body style='font-family: monospace; font-size: 13px;'>")
    html.append("<h2>🎯 SCREENING B3 - SINAIS DE MOMENTUM</h2>")
    html.append(f"<p><strong>Data dos sinais:</strong> {data_base + timedelta(days=1)} (compra na abertura)</p>")
    html.append(f"<p><strong>💰 Capital disponível:</strong> R$ {CFG['capital_disponivel']:,.2f}</p>")
    html.append(f"<p><strong>Risco por trade:</strong> {CFG['risco_por_trade_pct']*100:.1f}% = R$ {CFG['capital_disponivel'] * CFG['risco_por_trade_pct']:,.2f}</p>")
    html.append("<hr>")
    
    if df_sinais.empty and n_evitar == 0:
        html.append("<p>⚠️ <strong>Nenhum sinal hoje.</strong></p>")
        html.append("</body></html>")
        return "".join(html)
    
    if df_sinais.empty and n_evitar > 0:
        html.append(f"<p>⚠️ <strong>Nenhum sinal ✅ OPERAR hoje.</strong></p>")
        html.append(f"<p>{n_evitar} {'sinal' if n_evitar == 1 else 'sinais'} ❌ EVITAR (filtrado{'s' if n_evitar > 1 else ''})</p>")
        html.append("<p><strong>💡 Recomendação:</strong> Não operar hoje. Aguarde sinais de melhor qualidade.</p>")
        html.append("</body></html>")
        return "".join(html)
    
    # Tabela HTML
    html.append("<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; font-size: 12px;'>")
    html.append("<thead style='background-color: #28a745; color: white;'>")  # Verde para OPERAR
    html.append("<tr>")
    for col in df_sinais.columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr>")
    html.append("</thead>")
    html.append("<tbody>")
    
    for idx, row in df_sinais.iterrows():
        # Todas as linhas em verde claro (só tem sinais OPERAR)
        html.append(f"<tr style='background-color: #d4edda;'>")
        for col in df_sinais.columns:
            valor = row[col]
            # Formata valores numéricos
            if col == "valor_R$" and not pd.isna(valor):
                valor = f"R$ {valor:,.2f}"
            elif col == "pct_capital" and not pd.isna(valor):
                valor = f"{valor:.1f}%"
            html.append(f"<td>{valor}</td>")
        html.append("</tr>")
    
    html.append("</tbody>")
    html.append("</table>")
    
    # Resumo
    html.append("<hr>")
    html.append("<h3>📊 Resumo</h3>")
    html.append(f"<p>✅ <strong>Sinais operáveis:</strong> {len(df_sinais)}</p>")
    if n_evitar > 0:
        html.append(f"<p>❌ <strong>Sinais filtrados:</strong> {n_evitar} (não exibidos)</p>")
    
    # Recomendação
    n_sinais = len(df_sinais)
    html.append("<hr>")
    if n_sinais == 1:
        html.append("<p><strong>💡 Recomendação:</strong> 1 sinal disponível. Opere se tiver capital.</p>")
    elif n_sinais <= CFG["max_posicoes_simultaneas"]:
        html.append(f"<p><strong>💡 Recomendação:</strong> {n_sinais} sinais. Pode operar todos se tiver capital.</p>")
    else:
        html.append(f"<p><strong>💡 Recomendação:</strong> {n_sinais} sinais (mais que o máx de {CFG['max_posicoes_simultaneas']}). ")
        html.append(f"Priorize os {CFG['max_posicoes_simultaneas']} primeiros ou escolha por maior ret_20d_%.</p>")
    
    html.append("<hr>")
    html.append("<p style='font-size: 11px; color: #666;'>")
    html.append("📈 <strong>Legenda:</strong><br>")
    html.append("• qtd = quantidade de ações | valor_R$ = valor total | pct_capital = % do capital<br>")
    html.append("• close = preço ref | alvo = +5% | stop = -4%<br>")
    html.append("• classificacao: ✅ OPERAR (0-1 alertas, taxa de acerto ~78%)")
    html.append("</p>")
    html.append("</body></html>")
    
    return "".join(html)


def rodar_screening(data_base: date = None, ticker_unico: str = None):
    """
    Screening diário: busca sinais em TODO o universo.
    
    Para Mean Reversion LIGHT, não usamos lista de "papéis promissores"
    porque sinais são raros e podem aparecer em qualquer papel.
    """
    if data_base is None:
        data_base = date.today()
    
    if ticker_unico:
        tickers = [ticker_unico]
    else:
        # USA UNIVERSO COMPLETO (não papeis_promissores)
        tickers = carregar_universo_completo()
    
    print(f"\n{'='*60}")
    print(f"SCREENING  |  {data_base}  |  {len(tickers)} tickers")
    print(f"Estratégia: Mean Reversion LIGHT")
    print(f"{'='*60}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    todos_sinais = []
    
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = preparar_df(ticker, data_base)
            sinais = avaliar_sinais_no_dia(ticker, df, data_base)
            if sinais:
                print(f"SINAL")
            else:
                print("sem sinal")
            todos_sinais.extend(sinais)
        except Exception as e:
            print(f"ERRO: {e}")
            traceback.print_exc()
    
    if not todos_sinais:
        print("\n⚠️  Nenhum sinal hoje.")
        df_sinais_completo = pd.DataFrame()
        df_sinais_operar = pd.DataFrame()
    else:
        df_sinais_completo = pd.DataFrame(todos_sinais)
        
        # --- CLASSIFICAÇÃO BINÁRIA ---
        df_sinais_completo["classificacao"] = df_sinais_completo.apply(_classificar_sinal, axis=1)
        
        # Separar OPERAR vs EVITAR
        df_sinais_operar = df_sinais_completo[
            df_sinais_completo["classificacao"].str.startswith("✅")
        ].copy()
        
        df_sinais_evitar = df_sinais_completo[
            df_sinais_completo["classificacao"].str.startswith("❌")
        ].copy()
    
    # --- SALVAR CSV COMPLETO (com OPERAR + EVITAR) ---
    caminho = f"{CFG['pasta_resultados']}/sinais_{data_base}.csv"
    if not df_sinais_completo.empty:
        df_sinais_completo.to_csv(caminho, index=False)
    
    # --- GERAR RELATÓRIOS (só com sinais OPERAR) ---
    relatorio_texto = _gerar_relatorio_texto(
        df_sinais_operar, 
        data_base,
        n_evitar=len(df_sinais_completo) - len(df_sinais_operar) if not df_sinais_completo.empty else 0
    )
    relatorio_html = _gerar_relatorio_html(
        df_sinais_operar, 
        data_base,
        n_evitar=len(df_sinais_completo) - len(df_sinais_operar) if not df_sinais_completo.empty else 0
    )
    
    # Imprimir no terminal
    print(relatorio_texto)
    
    if not df_sinais_completo.empty:
        print(f"\n[✓] CSV completo salvo (inclui sinais EVITAR): {caminho}")
    
    # --- ENVIAR EMAIL ---
    if CFG["enviar_email"]:
        try:
            n_operar = len(df_sinais_operar)
            
            if n_operar == 0:
                # Email simples quando não há sinais
                assunto = "✓ Sem ordens de compra"
                corpo_texto = f"""Screening B3 - {data_base}

Nenhum sinal de compra hoje.

{len(tickers)} papéis analisados.
Estratégia: Mean Reversion LIGHT
Próximo screening: amanhã após 18h

---
Sistema E3 - Screening automático"""
                
                corpo_html = f"""<html>
<body style="font-family: Arial, sans-serif; padding: 20px;">
<h2 style="color: #666;">Screening B3 - {data_base}</h2>
<p style="font-size: 16px; color: #333;">
    <strong>Nenhum sinal de compra hoje.</strong>
</p>
<p style="color: #666;">
    {len(tickers)} papéis analisados.<br>
    Estratégia: Mean Reversion LIGHT<br>
    Próximo screening: amanhã após 18h
</p>
<hr style="border: 1px solid #ddd; margin: 20px 0;">
<p style="color: #999; font-size: 12px;">Sistema E3 - Screening automático</p>
</body>
</html>"""
            else:
                # Email normal com sinais
                assunto = f"🎯 Screening B3 - {n_operar} {'sinal' if n_operar == 1 else 'sinais'} para {data_base + timedelta(days=1)}"
                corpo_texto = relatorio_texto
                corpo_html = relatorio_html
            
            utils.enviar_email_gmail_com_app_password(
                assunto=assunto,
                corpo_texto=corpo_texto,
                corpo_html=corpo_html
            )
            print(f"\n[✓] Email enviado com sucesso")
        except Exception as e:
            print(f"\n[ERRO] Falha ao enviar email: {e}")
    
    return df_sinais_completo


def _classificar_sinal(row: pd.Series) -> str:
    """
    Classifica o sinal em ✅ OPERAR ou ❌ EVITAR.
    
    Critério validado estatisticamente:
    - 0-1 alertas → ✅ OPERAR (taxa de acerto ~78-80%)
    - 2+ alertas  → ❌ EVITAR (taxa de acerto ~57%)
    """
    alertas = []
    
    # --- Critérios de alerta ---
    rsi = row.get("rsi", 50)
    if rsi > 75:
        alertas.append("RSI alto")
    
    norm = row.get("norm_mm200", 1.0)
    if norm > 1.25:
        alertas.append("Preço esticado")
    
    vol_rel = row.get("vol_rel", 1.0)
    if vol_rel < 0.7:
        alertas.append("Volume baixo")
    
    ret_5d = row.get("ret_5d_%", 0)
    if ret_5d < 0.5:
        alertas.append("Momentum fraco")
    
    adx = row.get("adx", 30)
    if adx < 27:
        alertas.append("Tendência fraca")
    
    # --- Classificação binária ---
    n_alertas = len(alertas)
    
    if n_alertas <= 1:
        # 0 ou 1 alerta = sinal de qualidade
        if n_alertas == 0:
            return "✅ OPERAR"
        else:
            return f"✅ OPERAR ({alertas[0]})"
    else:
        # 2+ alertas = sinal arriscado
        return f"❌ EVITAR ({', '.join(alertas[:2])})"


def _imprimir_rodape_explicativo(df_sinais: pd.DataFrame):
    """
    Imprime guia de interpretação das colunas e recomendações.
    """
    print(f"\n{'='*60}")
    print("📊 GUIA DE INTERPRETAÇÃO")
    print(f"{'='*60}")
    
    print("\n🎯 OPERAÇÃO (compra na abertura de amanhã):")
    print("  • close  = preço de referência (compra próximo disso)")
    print("  • alvo   = venda com lucro (+5%)")
    print("  • stop   = venda com prejuízo (-4%)")
    print("  • r:r    = ratio risco:retorno (ideal ≥ 1.2)")
    print("  • max_dias = sai no dia 20 se não atingir alvo/stop")
    
    print("\n📈 INDICADORES (validação do sinal):")
    print("  • rsi:")
    print("      < 50  = papel ainda tem espaço para subir")
    print("      50-70 = zona normal")
    print("      > 70  = ⚠️  sobrecomprado, risco maior")
    print("      > 75  = 🔴 muito sobrecomprado, evitar")
    
    print("\n  • adx:")
    print("      ≥ 30  = tendência forte (ideal)")
    print("      25-30 = tendência aceitável")
    print("      < 25  = ⚠️  tendência fraca, sinal limítrofe")
    
    print("\n  • vol_rel (volume relativo à média 20d):")
    print("      > 1.5  = volume alto (pode ser notícia)")
    print("      0.8-1.5 = volume normal")
    print("      < 0.8  = ⚠️  volume baixo, menos liquidez")
    print("      < 0.7  = 🔴 liquidez ruim, evitar")
    
    print("\n  • ret_5d_% / ret_20d_% (momentum):")
    print("      ret_20d > 10% = momentum forte")
    print("      ret_5d > 2%   = ainda acelerando (bom sinal)")
    print("      ret_5d < 1%   = ⚠️  perdendo força recente")
    
    print("\n  • norm_mm200 (preço / MM200):")
    print("      1.00-1.15 = distância saudável")
    print("      1.15-1.25 = esticado, mas aceitável")
    print("      > 1.25    = 🔴 muito esticado, risco de correção")
    
    print("\n⚠️  CLASSIFICAÇÃO DO SINAL:")
    print("  🟢 BOM      = todos os indicadores dentro do ideal")
    print("  🟡 MODERADO = 1 alerta, ainda operável com cautela")
    print("  🔴 ATENÇÃO  = 2+ alertas, considere pular ou reduzir posição")
    
    # Estatísticas dos sinais do dia
    n_bom      = (df_sinais["alerta"].str.startswith("🟢")).sum()
    n_moderado = (df_sinais["alerta"].str.startswith("🟡")).sum()
    n_atencao  = (df_sinais["alerta"].str.startswith("🔴")).sum()
    
    print(f"\n📊 RESUMO DOS SINAIS DE HOJE:")
    print(f"  🟢 Bons: {n_bom}")
    print(f"  🟡 Moderados: {n_moderado}")
    print(f"  🔴 Atenção: {n_atencao}")
    
    if n_bom == 0 and n_moderado == 0:
        print(f"\n  💡 Recomendação: Todos os sinais têm alertas. Considere não operar hoje.")
    elif n_bom > 0:
        print(f"\n  💡 Recomendação: Priorize os sinais 🟢. Se tiver capital limitado,")
        print(f"     concentre nos {min(n_bom, 2)} primeiros da lista.")
    
    print(f"\n{'='*60}")

"""
NOVAS FUNÇÕES DE VALIDAÇÃO - UNIVERSO LÍQUIDO
Versão CORRIGIDA - Walk-forward POR ANO dentro de cada ticker

Para adicionar no e3_screening_regras.py antes da função main()
"""

def testar_mean_rev_light(data_base: date = None):
    """
    VALIDAÇÃO MEAN REVERSION: LIGHT vs SUPER-LIGHT (universo líquido)
    """
    if data_base is None:
        data_base = date.today()
    
    # Carregar universo LÍQUIDO
    caminho_universo = CFG["universo_liquido_file"]
    if Path(caminho_universo).exists():
        with open(caminho_universo, 'r') as f:
            tickers = [linha.strip() for linha in f if linha.strip()]
        print(f"✓ Universo líquido: {len(tickers)} papéis")
    else:
        print("⚠️  Usando universo completo...")
        tickers = carregar_universo_completo()
    
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"VALIDAÇÃO: MEAN REVERSION LIGHT vs SUPER-LIGHT")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Definir estratégias
    estrategias = {
        "MEAN REV LIGHT": ESTRATEGIA_MEAN_REVERSION_LIGHT,
        "MEAN REV SUPER-LIGHT": {
            "nome": "mean_reversion_super_light",
            "descricao": "Mean Reversion SUPER-LIGHT",
            "parametros": {
                "rsi_max": 50,
                "queda_5d_min": -0.02,
                "close_acima_mm50": False,
                "close_acima_mm20": True,
                "ibov_acima_mm200": True,
                "vol_relativo_min": 0.5,
                "adx_max": 40,
                "rsi_min": 20,
                "alvo_pct": 0.03,
                "stop_pct": 0.04,
                "max_dias": 15,
            },
            "fn_sinal": lambda row, p: (
                _val(row, "rsi") <= p["rsi_max"]
                and _val(row, "rsi") >= p["rsi_min"]
                and _retorno_nd(row, 5) <= p["queda_5d_min"]
                and _val(row, "close") > _val(row, "mm20", 0)
                and _val(row, "close_ibov") > _val(row, "mm200_ibov", 0)
                and _val(row, "vol_relativo") >= p["vol_relativo_min"]
                and _val(row, "ADX_14") <= p["adx_max"]
            ),
        },
    }
    
    resultados = {nome: {ano: [] for ano in anos} for nome in estrategias.keys()}
    
    # Loop por ticker (externo)
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        ticker_ok = False
        
        try:
            # WALK-FORWARD CORRETO: processar cada ANO separadamente
            for ano in anos:
                try:
                    # Preparar dados ATÉ o final do ano (não todos os anos)
                    df = preparar_df(ticker, date(ano, 12, 31))
                    
                    if df.empty or len(df) < 50:
                        continue
                    
                    ticker_ok = True
                    
                    # Rodar backtest
                    for nome_estrategia, estrategia in estrategias.items():
                        trades = backtest_estrategia(ticker, df, estrategia)
                        
                        if trades.empty:
                            continue
                        
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        
                        # Filtrar trades deste ano
                        inicio = pd.Timestamp(date(ano, 1, 1))
                        fim = pd.Timestamp(date(ano, 12, 31))
                        
                        trades_ano = trades[
                            (trades["data_entrada"] >= inicio) &
                            (trades["data_entrada"] <= fim)
                        ]
                        
                        if not trades_ano.empty:
                            resultados[nome_estrategia][ano].append(trades_ano)
                
                except Exception:
                    continue
            
            if ticker_ok:
                print("✓")
            else:
                print("skip")
        
        except Exception:
            print("erro")
            continue
    
    # Análise
    print(f"\n{'='*70}")
    print("📊 ANÁLISE")
    print(f"{'='*70}\n")
    
    relatorio = {}
    
    for nome in estrategias.keys():
        print(f"\n{'='*60}")
        print(f"🔬 {nome}")
        print(f"{'='*60}\n")
        
        params = estrategias[nome]["parametros"]
        dias_medio = params["max_dias"]
        
        metricas_por_ano = []
        todos = []
        
        for ano in anos:
            lista = resultados[nome][ano]
            
            if not lista:
                metricas_por_ano.append({
                    "ano": ano,
                    "trades": 0,
                    "taxa": 0,
                    "ret_medio": 0,
                    "ret_anual": 0,
                })
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            todos.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n
            ret_medio = df_ano["retorno_%"].mean()
            ret_anual = ret_medio * (252 / dias_medio)
            
            metricas_por_ano.append({
                "ano": ano,
                "trades": n,
                "taxa": taxa,
                "ret_medio": ret_medio,
                "ret_anual": ret_anual,
            })
        
        # Tabela
        print("📅 ANO A ANO:\n")
        
        df_metricas = pd.DataFrame(metricas_por_ano)
        
        print(f"{'Ano':<6} {'Trades':<8} {'Taxa':<10} {'Ret/trade':<12} {'Ret Anual':<12}")
        print(f"{'-'*60}")
        
        for _, row in df_metricas.iterrows():
            ano_str = str(int(row['ano']))
            trades_str = str(int(row['trades']))
            taxa_str = f"{row['taxa']:.1%}" if row['trades'] > 0 else "-"
            ret_str = f"{row['ret_medio']:.2f}%" if row['trades'] > 0 else "-"
            anual_str = f"{row['ret_anual']:.1f}%" if row['trades'] > 0 else "-"
            
            print(f"{ano_str:<6} {trades_str:<8} {taxa_str:<10} {ret_str:<12} {anual_str:<12}")
        
        if not todos:
            print("\n❌ 0 trades.\n")
            relatorio[nome] = {"aprovado": False}
            continue
        
        # Consolidado
        df_total = pd.concat(todos, ignore_index=True)
        
        n_total = len(df_total)
        ops_ano = n_total / len([a for a in anos if a != ano_atual])
        taxa = (df_total["retorno_%"] > 0).sum() / n_total
        ret_medio = df_total["retorno_%"].mean()
        ret_std = df_total["retorno_%"].std()
        ret_anual = ret_medio * (252 / dias_medio)
        
        sharpe = (ret_medio / ret_std * np.sqrt(252 / dias_medio)) if ret_std > 0 else 0
        
        # Drawdown
        df_total = df_total.sort_values("data_entrada").reset_index(drop=True)
        df_total["equity"] = (1 + df_total["retorno_%"] / 100).cumprod()
        df_total["peak"] = df_total["equity"].cummax()
        df_total["drawdown"] = (df_total["equity"] / df_total["peak"] - 1) * 100
        max_dd = df_total["drawdown"].min()
        
        # Anos
        anos_com = df_metricas[df_metricas["trades"] > 0]
        anos_pos = (anos_com["ret_anual"] > 0).sum()
        anos_neg = (anos_com["ret_anual"] < 0).sum()
        pior = anos_com["ret_anual"].min() if len(anos_com) > 0 else 0
        melhor = anos_com["ret_anual"].max() if len(anos_com) > 0 else 0
        
        print(f"\n📈 CONSOLIDADO:\n")
        print(f"  {n_total} trades | {ops_ano:.1f}/ano")
        print(f"  Taxa: {taxa:.1%} | Sharpe: {sharpe:.2f}")
        print(f"  Ret/trade: {ret_medio:.2f}% | Ret anual: {ret_anual:.1f}%")
        print(f"  DD: {max_dd:.1f}% | Pior: {pior:.1f}% | Melhor: {melhor:.1f}%")
        print(f"  Anos +/-: {anos_pos}/{anos_neg}")
        
        relatorio[nome] = {
            "trades": n_total,
            "ops_ano": ops_ano,
            "taxa": taxa * 100,
            "sharpe": sharpe,
            "ret_anual": ret_anual,
            "drawdown": max_dd,
            "pior": pior,
            "melhor": melhor,
        }
    
    # Comparação
    print(f"\n{'='*70}")
    print("🏆 LIGHT vs SUPER-LIGHT")
    print(f"{'='*70}\n")
    
    light = relatorio.get("MEAN REV LIGHT", {})
    super_light = relatorio.get("MEAN REV SUPER-LIGHT", {})
    
    print(f"{'Métrica':<25} {'LIGHT':<15} {'SUPER-LIGHT':<15} {'Dif':<10}")
    print(f"{'-'*70}")
    
    for label, key, fmt in [
        ("Ops/ano", "ops_ano", ".1f"),
        ("Taxa (%)", "taxa", ".1f"),
        ("Sharpe", "sharpe", ".2f"),
        ("Ret/ano (%)", "ret_anual", ".1f"),
        ("DD (%)", "drawdown", ".1f"),
    ]:
        v1 = light.get(key, 0)
        v2 = super_light.get(key, 0)
        diff = v2 - v1
        s1 = f"{v1:{fmt}}"
        s2 = f"{v2:{fmt}}"
        sdiff = f"{diff:+{fmt}}"
        print(f"{label:<25} {s1:<15} {s2:<15} {sdiff:<10}")
    
    print(f"\n{'='*70}\n")
    
    # Salvar
    caminho = f"{CFG['pasta_resultados']}/validacao_mean_rev_liquido_{data_base}.csv"
    df_resumo = pd.DataFrame([light, super_light], index=["LIGHT", "SUPER-LIGHT"])
    df_resumo.to_csv(caminho)
    print(f"[✓] Salvo: {caminho}\n")



def analise_robustez_tickers(lista_trades, nome_est_alvo="swing_balanceado_d"):
    if not lista_trades: return
    
    # Consolida todos os trades em um único DataFrame
    df_all = pd.concat(lista_trades)
    print(df_all['estrategia'].unique())
    df_est = df_all[df_all['estrategia'] == nome_est_alvo]
    print(f'Quantidade de trades da estrategia {nome_est_alvo}: {df_est.shape[0]}')
    # 1. Agrupamento por Ticker
    stats_ticker = df_est.groupby('ticker').agg({
        'retorno_%': ['count', 'mean', 'sum'],
        'resultado': lambda x: (x == 'alvo').sum() / len(x) * 100
    }).reset_index()
    
    stats_ticker.columns = ['Ticker', 'Trades', 'Ret_Medio_%', 'Ret_Acumulado_%', 'Taxa_Acerto_%']
    stats_ticker = stats_ticker.sort_values(by='Ret_Acumulado_%', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"📊 RANKING POR TICKER (Estratégia: {nome_est_alvo})")
    print(f"{'='*60}")
    print(stats_ticker.to_string(index=False))
    
    # 2. Validação "Ruthless": Teste de Exclusão do Melhor Ticker
    melhor_ticker = stats_ticker.iloc[0]['Ticker']
    ret_sem_melhor = stats_ticker.iloc[1:]['Ret_Acumulado_%'].sum()
    total_original = stats_ticker['Ret_Acumulado_%'].sum()
    dependencia = (stats_ticker.iloc[0]['Ret_Acumulado_%'] / total_original) * 100
    
    print(f"\n{'='*60}")
    print(f"🛡️ TESTE DE ROBUSTEZ (STRESS TEST)")
    print(f"{'='*60}")
    print(f"Melhor Ticker: {melhor_ticker} (Contribuiu com {dependencia:.1f}% do lucro total)")
    
    if dependencia > 40:
        print("❌ ALERTA: Dependência ALTA. O sistema está viciado em um único ativo.")
    elif dependencia > 25:
        print("⚠️ AVISO: Dependência moderada. Verifique a liquidez deste ativo.")
    else:
        print("✅ ROBUSTO: O lucro está bem distribuído entre os ativos.")
    print(f"{'='*60}\n")

def testar_swing_balanceado(data_base: date = None):
    """
    VALIDAÇÃO SWING BALANCEADO: A vs B vs C (universo líquido)
    """
    if data_base is None:
        data_base = date.today()
    
    # Carregar universo LÍQUIDO
    caminho_universo = CFG["universo_liquido_file"]
    if Path(caminho_universo).exists():
        with open(caminho_universo, 'r') as f:
            tickers = [linha.strip() for linha in f if linha.strip()]
        print(f"✓ Universo líquido: {len(tickers)} papéis")
    else:
        print("⚠️  Usando universo completo...")
        tickers = carregar_universo_completo()
    
    if CFG['testar_universo_reduzido']:
        # lista aleatoria de 5 tickers
        random.seed(42)
        tickers = random.sample(tickers, min(5, len(tickers)))
        print(f"⚠️  Testando universo reduzido: {tickers}")


    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"VALIDAÇÃO: SWING BALANCEADO A vs B vs C vs D vs E")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    estrategias = {
        "BALANCEADO A": ESTRATEGIA_SWING_BALANCEADO_A,
        #"BALANCEADO B": ESTRATEGIA_SWING_BALANCEADO_B,
        #"BALANCEADO C": ESTRATEGIA_SWING_BALANCEADO_C,
        "BALANCEADO D": ESTRATEGIA_SWING_BALANCEADO_D,
        #"BALANCEADO E": ESTRATEGIA_SWING_BALANCEADO_E,
    }
    
    resultados = {nome: {ano: [] for ano in anos} for nome in estrategias.keys()}
    
    todos_os_trades_detalhado = []

    # Loop por ticker (externo)
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        ticker_ok = False
        
        try:
            # WALK-FORWARD CORRETO: processar cada ANO separadamente
            for ano in anos:
                try:
                    # Preparar dados ATÉ o final do ano (não todos os anos)
                    df = preparar_df(ticker, date(ano, 12, 31))
                    
                    if df.empty or len(df) < 50:
                        continue
                    
                    ticker_ok = True
                    
                    # Rodar backtest
                    for nome_estrategia, estrategia in estrategias.items():
                        trades = backtest_estrategia(ticker, df, estrategia)
                        
                        if trades.empty:
                            continue
                        
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        
                        # Filtrar trades deste ano
                        inicio = pd.Timestamp(date(ano, 1, 1))
                        fim = pd.Timestamp(date(ano, 12, 31))
                        
                        trades_ano = trades[
                            (trades["data_entrada"] >= inicio) &
                            (trades["data_entrada"] <= fim)
                        ]
                        
                        if not trades_ano.empty:
                            resultados[nome_estrategia][ano].append(trades_ano)
                            todos_os_trades_detalhado.append(trades_ano)
                
                except Exception:
                    continue
            
            if ticker_ok:
                print("✓")
            else:
                print("skip")
        
        except Exception:
            print("erro")
            continue   
    

    # Análise
    print(f"\n{'='*70}")
    print("📊 ANÁLISE")
    print(f"{'='*70}\n")
    
    relatorio = {}
    
    for nome in estrategias.keys():
        print(f"\n{'='*60}")
        print(f"🔬 {nome}")
        print(f"{'='*60}\n")
        
        params = estrategias[nome]["parametros"]
        dias_medio = params["max_dias"]
        
        metricas_por_ano = []
        todos = []
        
        for ano in anos:
            lista = resultados[nome][ano]
            
            if not lista:
                metricas_por_ano.append({
                    "ano": ano,
                    "trades": 0,
                    "taxa": 0,
                    "ret_medio": 0,
                    "ret_anual": 0,
                })
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            todos.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n
            ret_medio = df_ano["retorno_%"].mean()
            ret_anual = ret_medio * (252 / dias_medio)
            
            metricas_por_ano.append({
                "ano": ano,
                "trades": n,
                "taxa": taxa,
                "ret_medio": ret_medio,
                "ret_anual": ret_anual,
            })
        
        # Tabela
        print("📅 ANO A ANO:\n")
        
        df_metricas = pd.DataFrame(metricas_por_ano)
        
        print(f"{'Ano':<6} {'Trades':<8} {'Taxa':<10} {'Ret/trade':<12} {'Ret Anual':<12}")
        print(f"{'-'*60}")
        
        for _, row in df_metricas.iterrows():
            ano_str = str(int(row['ano']))
            trades_str = str(int(row['trades']))
            taxa_str = f"{row['taxa']:.1%}" if row['trades'] > 0 else "-"
            ret_str = f"{row['ret_medio']:.2f}%" if row['trades'] > 0 else "-"
            anual_str = f"{row['ret_anual']:.1f}%" if row['trades'] > 0 else "-"
            
            print(f"{ano_str:<6} {trades_str:<8} {taxa_str:<10} {ret_str:<12} {anual_str:<12}")
        
        if not todos:
            print("\n❌ 0 trades.\n")
            relatorio[nome] = {"aprovado": False}
            continue
        
        # Consolidado
        df_total = pd.concat(todos, ignore_index=True)
        
        n_total = len(df_total)
        ops_ano = n_total / len([a for a in anos if a != ano_atual])
        taxa = (df_total["retorno_%"] > 0).sum() / n_total
        ret_medio = df_total["retorno_%"].mean()
        ret_std = df_total["retorno_%"].std()
        ret_anual = ret_medio * (252 / dias_medio)
        
        sharpe = (ret_medio / ret_std * np.sqrt(252 / dias_medio)) if ret_std > 0 else 0
        
        # Drawdown
        df_total = df_total.sort_values("data_entrada").reset_index(drop=True)
        df_total["equity"] = (1 + df_total["retorno_%"] / 100).cumprod()
        df_total["peak"] = df_total["equity"].cummax()
        df_total["drawdown"] = (df_total["equity"] / df_total["peak"] - 1) * 100
        max_dd = df_total["drawdown"].min()
        
        # Anos
        anos_com = df_metricas[df_metricas["trades"] > 0]
        anos_pos = (anos_com["ret_anual"] > 0).sum()
        anos_neg = (anos_com["ret_anual"] < 0).sum()
        pior = anos_com["ret_anual"].min() if len(anos_com) > 0 else 0
        melhor = anos_com["ret_anual"].max() if len(anos_com) > 0 else 0
        
        print(f"\n📈 CONSOLIDADO:\n")
        print(f"  {n_total} trades | {ops_ano:.1f}/ano")
        print(f"  Taxa: {taxa:.1%} | Sharpe: {sharpe:.2f}")
        print(f"  Ret/trade: {ret_medio:.2f}% | Ret anual: {ret_anual:.1f}%")
        print(f"  DD: {max_dd:.1f}% | Pior: {pior:.1f}% | Melhor: {melhor:.1f}%")
        print(f"  Anos +/-: {anos_pos}/{anos_neg}")
        
        relatorio[nome] = {
            "trades": n_total,
            "ops_ano": ops_ano,
            "taxa": taxa * 100,
            "sharpe": sharpe,
            "ret_anual": ret_anual,
            "drawdown": max_dd,
            "pior": pior,
            "melhor": melhor,
        }
    
    # Comparação
    print(f"\n{'='*70}")
    print("🏆 A vs B vs C")
    print(f"{'='*70}\n")
    
    a = relatorio.get("BALANCEADO A", {})
    b = relatorio.get("BALANCEADO B", {})
    c = relatorio.get("BALANCEADO C", {})
    d = relatorio.get("BALANCEADO D", {})
    e = relatorio.get("BALANCEADO E", {})
    
    print(f"{'Métrica':<20} {'A':<15} {'B':<15} {'C':<15} {'D':<15} {'E':<15}")
    print(f"{'-'*90}")
    
    for label, key, fmt in [
        ("Ops/ano", "ops_ano", ".1f"),
        ("Taxa (%)", "taxa", ".1f"),
        ("Sharpe", "sharpe", ".2f"),
        ("Ret/ano (%)", "ret_anual", ".1f"),
        ("DD (%)", "drawdown", ".1f"),
    ]:
        va = a.get(key, 0)
        vb = b.get(key, 0)
        vc = c.get(key, 0)
        vd = d.get(key, 0)
        ve = e.get(key, 0)
        sa = f"{va:{fmt}}"
        sb = f"{vb:{fmt}}"
        sc = f"{vc:{fmt}}"
        sd = f"{vd:{fmt}}"
        se = f"{ve:{fmt}}"
        print(f"{label:<20} {sa:<15} {sb:<15} {sc:<15} {sd:<15} {se:<15}")
    
    print(f"\n{'='*70}\n")
    
    # Salvar
    caminho = f"{CFG['pasta_resultados']}/validacao_swing_liquido_{data_base}.csv"
    df_resumo = pd.DataFrame([a, b, c, d, e], index=["A", "B", "C", "D", "E"])
    df_resumo.to_csv(caminho)
    print(f"[✓] Salvo: {caminho}\n")

    analise_robustez_tickers(todos_os_trades_detalhado, "swing_balanceado_d")


def testar_deteccoes_swing_low(data_base: date = None):
    """
    Testa 3 novas detecções de swing low APRIMORADAS:
    1. CONFIRMADO: Espera recuperação começar
    2. CAPITULAÇÃO: Volume alto + RSI baixo
    3. RIGOROSO: Múltiplas confirmações
    
    Compara com detecção ATUAL (permissiva)
    Objetivo: 20-40 ops/ano com 70-80% taxa
    """
    if data_base is None:
        data_base = date.today()
    
    # Carregar universo LÍQUIDO
    caminho_universo = CFG.get("universo_liquido_file", "")
    if Path(caminho_universo).exists():
        with open(caminho_universo, 'r') as f:
            tickers = [linha.strip() for linha in f if linha.strip()]
        print(f"✓ Universo líquido: {len(tickers)} papéis")
    else:
        print("⚠️ Usando universo completo...")
        tickers = carregar_universo_completo()
    
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"TESTE: DETECÇÕES DE SWING LOW APRIMORADAS")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    # Testar 4 versões: ATUAL + 3 novas
    # Usaremos parâmetros do BALANCEADO A como base
    versoes = {
        "ATUAL (permissiva)": {
            "tipo_swing": "swing_low_3",
            "valor_swing": "swing_low_value_3",
        },
        "CONFIRMADO": {
            "tipo_swing": "swing_confirmado_5",
            "valor_swing": "swing_confirmado_value_5",
        },
        "CAPITULAÇÃO": {
            "tipo_swing": "swing_capitulacao_5",
            "valor_swing": "swing_capitulacao_value_5",
        },
        "RIGOROSO": {
            "tipo_swing": "swing_rigoroso_5",
            "valor_swing": "swing_rigoroso_value_5",
        },
    }
    
    # Parâmetros base (BALANCEADO A)
    parametros_base = {
        "max_distancia_suporte": 0.025,
        "vol_relativo_min": 0.8,
        "rsi_min": 25,
        "rsi_max": 55,
        "close_acima_mm50": True,
        "alvo_pct": 0.06,
        "stop_pct": 0.03,
        "max_dias": 20,
    }
    
    resultados = {nome: {ano: [] for ano in anos} for nome in versoes.keys()}
    
    # LOOP CORRETO: Ticker externo (carrega dados 1× por ticker)
    for i, ticker in enumerate(tickers, 1):
        if i % 5 == 0:
            print(f"[{i}/{len(tickers)}]...", end=" ", flush=True)
        
        try:
            # Para cada ticker, processar todos os anos
            for ano in anos:
                try:
                    # PREPARAR DF UMA VEZ POR TICKER/ANO
                    df = preparar_df(ticker, date(ano, 12, 31))
                    
                    if df.empty or len(df) < 50:
                        continue
                    
                    # TESTAR TODAS AS VERSÕES neste mesmo DF
                    for nome_versao, config in versoes.items():
                        
                        # Verificar se colunas existem
                        if config["tipo_swing"] not in df.columns:
                            continue
                        if config["valor_swing"] not in df.columns:
                            continue
                        
                        # Backtest manual (similar ao backtest_estrategia)
                        trades_ano = []
                        
                        for idx in range(50, len(df)):
                            row = df.iloc[idx]
                            
                            # Verificar sinal
                            eh_swing = row.get(config["tipo_swing"], False)
                            valor_swing = row.get(config["valor_swing"], np.inf)
                            
                            if not eh_swing or pd.isna(valor_swing) or valor_swing <= 0:
                                continue
                            
                            close = row.get("close", 0)
                            if close <= 0:
                                continue
                            
                            # Verificar distância do suporte
                            if close > valor_swing * (1 + parametros_base["max_distancia_suporte"]):
                                continue
                            
                            # Verificar volume
                            vol_rel = row.get("vol_relativo", 0)
                            if vol_rel < parametros_base["vol_relativo_min"]:
                                continue
                            
                            # Verificar RSI
                            rsi = row.get("rsi", 50)
                            if rsi < parametros_base["rsi_min"] or rsi > parametros_base["rsi_max"]:
                                continue
                            
                            # Verificar MM50
                            mm50 = row.get("mm50", 0)
                            if close <= mm50:
                                continue
                            
                            # ENTRADA
                            data_entrada = row.name
                            preco_entrada = close
                            
                            # Simular saída
                            alvo = preco_entrada * (1 + parametros_base["alvo_pct"])
                            stop = preco_entrada * (1 - parametros_base["stop_pct"])
                            
                            resultado = None
                            dias_em_trade = 0
                            
                            for j in range(idx + 1, min(idx + parametros_base["max_dias"] + 1, len(df))):
                                dias_em_trade += 1
                                high_dia = df.iloc[j]["high"]
                                low_dia = df.iloc[j]["low"]
                                close_dia = df.iloc[j]["close"]
                                
                                # Verifica alvo
                                if high_dia >= alvo:
                                    resultado = "alvo"
                                    preco_saida = alvo
                                    break
                                
                                # Verifica stop
                                if low_dia <= stop:
                                    resultado = "stop"
                                    preco_saida = stop
                                    break
                                
                                # Último dia
                                if j == min(idx + parametros_base["max_dias"], len(df) - 1):
                                    resultado = "timeout"
                                    preco_saida = close_dia
                                    break
                            
                            if resultado:
                                # Calcular retorno com taxas
                                taxas = 0.004
                                retorno_pct = 100 * ((preco_saida * (1 - taxas/2)) / (preco_entrada * (1 + taxas/2)) - 1)
                                
                                # Filtrar por ano
                                if data_entrada.year == ano:
                                    trades_ano.append({
                                        "data_entrada": data_entrada,
                                        "retorno_%": retorno_pct,
                                        "resultado": resultado,
                                        "dias": dias_em_trade,
                                    })
                        
                        if trades_ano:
                            resultados[nome_versao][ano].append(pd.DataFrame(trades_ano))
                
                except Exception:
                    continue
        
        except Exception:
            continue
    
    print("\n✓ Backtests concluídos")
    
    # Análise
    print(f"\n{'='*70}")
    print("📊 ANÁLISE COMPARATIVA")
    print(f"{'='*70}\n")
    
    resumo = []
    
    for nome_versao in versoes.keys():
        print(f"\n{'='*60}")
        print(f"🔬 {nome_versao}")
        print(f"{'='*60}\n")
        
        metricas_por_ano = []
        todos = []
        
        for ano in anos:
            lista = resultados[nome_versao][ano]
            
            if not lista:
                metricas_por_ano.append({
                    "ano": ano,
                    "trades": 0,
                    "taxa": 0,
                    "ret_medio": 0,
                    "ret_anual": 0,
                })
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            todos.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n
            ret_medio = df_ano["retorno_%"].mean()
            
            # Anualizar
            dias_medio = df_ano["dias"].mean() if "dias" in df_ano.columns else 15
            ret_anual = ret_medio * (252 / dias_medio)
            
            metricas_por_ano.append({
                "ano": ano,
                "trades": n,
                "taxa": taxa,
                "ret_medio": ret_medio,
                "ret_anual": ret_anual,
            })
        
        # Tabela ano a ano
        print("📅 ANO A ANO:\n")
        df_metricas = pd.DataFrame(metricas_por_ano)
        
        print(f"{'Ano':<6} {'Trades':<8} {'Taxa':<10} {'Ret/trade':<12} {'Ret Anual':<12}")
        print(f"{'-'*60}")
        
        for _, row in df_metricas.iterrows():
            ano_str = str(int(row['ano']))
            trades_str = str(int(row['trades']))
            taxa_str = f"{row['taxa']:.1%}" if row['trades'] > 0 else "-"
            ret_str = f"{row['ret_medio']:.2f}%" if row['trades'] > 0 else "-"
            anual_str = f"{row['ret_anual']:.1f}%" if row['trades'] > 0 else "-"
            
            print(f"{ano_str:<6} {trades_str:<8} {taxa_str:<10} {ret_str:<12} {anual_str:<12}")
        
        if not todos:
            print("\n❌ 0 trades.\n")
            resumo.append({
                "versao": nome_versao,
                "ops_ano": 0,
                "taxa": 0,
                "sharpe": 0,
                "ret_anual": 0,
                "dd": 0,
            })
            continue
        
        # Consolidado
        df_total = pd.concat(todos, ignore_index=True)
        
        n_total = len(df_total)
        ops_ano = n_total / len([a for a in anos if a != ano_atual])
        taxa = (df_total["retorno_%"] > 0).sum() / n_total
        ret_medio = df_total["retorno_%"].mean()
        ret_std = df_total["retorno_%"].std()
        
        dias_medio = df_total["dias"].mean() if "dias" in df_total.columns else 15
        ret_anual = ret_medio * (252 / dias_medio)
        
        sharpe = (ret_medio / ret_std * np.sqrt(252 / dias_medio)) if ret_std > 0 else 0
        
        # Drawdown
        df_total = df_total.sort_values("data_entrada").reset_index(drop=True)
        df_total["equity"] = (1 + df_total["retorno_%"] / 100).cumprod()
        df_total["peak"] = df_total["equity"].cummax()
        df_total["drawdown"] = (df_total["equity"] / df_total["peak"] - 1) * 100
        max_dd = df_total["drawdown"].min()
        
        # Anos
        anos_com = df_metricas[df_metricas["trades"] > 0]
        anos_pos = (anos_com["ret_anual"] > 0).sum()
        anos_neg = (anos_com["ret_anual"] < 0).sum()
        pior = anos_com["ret_anual"].min() if len(anos_com) > 0 else 0
        melhor = anos_com["ret_anual"].max() if len(anos_com) > 0 else 0
        
        print(f"\n📈 CONSOLIDADO:\n")
        print(f"  {n_total} trades | {ops_ano:.1f}/ano")
        print(f"  Taxa: {taxa:.1%} | Sharpe: {sharpe:.2f}")
        print(f"  Ret/trade: {ret_medio:.2f}% | Ret anual: {ret_anual:.1f}%")
        print(f"  DD: {max_dd:.1f}% | Pior: {pior:.1f}% | Melhor: {melhor:.1f}%")
        print(f"  Anos +/-: {anos_pos}/{anos_neg}")
        
        resumo.append({
            "versao": nome_versao,
            "ops_ano": round(ops_ano, 1),
            "taxa": round(taxa * 100, 1),
            "sharpe": round(sharpe, 2),
            "ret_anual": round(ret_anual, 1),
            "dd": round(max_dd, 1),
            "pior": round(pior, 1),
            "melhor": round(melhor, 1),
            "anos_pos_neg": f"{anos_pos}/{anos_neg}",
        })
    
    # Comparação final
    print(f"\n{'='*70}")
    print("🏆 COMPARAÇÃO FINAL")
    print(f"{'='*70}\n")
    
    df_resumo = pd.DataFrame(resumo)
    print(df_resumo.to_string(index=False))
    
    # Recomendação
    print(f"\n{'='*70}")
    print("💡 RECOMENDAÇÃO")
    print(f"{'='*70}\n")
    
    # Filtrar versões com dados válidos
    validas = df_resumo[df_resumo["ops_ano"] > 0].copy()
    
    if len(validas) == 0:
        print("❌ Nenhuma versão gerou trades suficientes.")
        return
    
    # Ordenar por Sharpe
    validas = validas.sort_values("sharpe", ascending=False)
    
    melhor = validas.iloc[0]
    
    print(f"🥇 MELHOR DETECÇÃO: {melhor['versao']}")
    print(f"   Ops/ano: {melhor['ops_ano']:.1f}")
    print(f"   Taxa: {melhor['taxa']:.1f}%")
    print(f"   Sharpe: {melhor['sharpe']:.2f}")
    print(f"   Retorno: {melhor['ret_anual']:.1f}%/ano")
    print(f"   Anos +/-: {melhor['anos_pos_neg']}")
    print()
    
    # Critérios de aprovação
    if melhor['ops_ano'] >= 20 and melhor['ops_ano'] <= 40:
        print(f"✅ Frequência IDEAL (20-40 ops/ano)")
    elif melhor['ops_ano'] < 20:
        print(f"⚠️ Frequência BAIXA ({melhor['ops_ano']:.0f} ops/ano)")
    else:
        print(f"⚠️ Frequência ALTA ({melhor['ops_ano']:.0f} ops/ano)")
    
    if melhor['taxa'] >= 70:
        print(f"✅ Taxa EXCELENTE ({melhor['taxa']:.0f}%)")
    elif melhor['taxa'] >= 60:
        print(f"⚠️ Taxa BOA ({melhor['taxa']:.0f}%)")
    else:
        print(f"❌ Taxa RUIM ({melhor['taxa']:.0f}%)")
    
    if melhor['sharpe'] >= 3.0:
        print(f"✅ Sharpe EXCEPCIONAL ({melhor['sharpe']:.1f})")
    elif melhor['sharpe'] >= 2.0:
        print(f"✅ Sharpe EXCELENTE ({melhor['sharpe']:.1f})")
    elif melhor['sharpe'] >= 1.0:
        print(f"⚠️ Sharpe BOM ({melhor['sharpe']:.1f})")
    else:
        print(f"❌ Sharpe RUIM ({melhor['sharpe']:.1f})")
    
    print()
    
    # Salvar
    caminho = f"{CFG['pasta_resultados']}/teste_deteccoes_swing_{data_base}.csv"
    df_resumo.to_csv(caminho, index=False)
    print(f"[✓] Salvo: {caminho}\n")


def testar_estrategias_gemini(data_base: date = None):
    """
    Testa 3 novas estrategias sugeridas pela GEMINI:
    1. ESTRATEGIA_VOLATILITY_SQUEEZE
    2. ESTRATEGIA_RS_PULLBACK 
    3. ESTRATEGIA_GAP_REVERSION_SNIPER
    
    Objetivo: 20-40 ops/ano com 70-80% taxa
    """
    if data_base is None:
        data_base = date.today()
    
    # Carregar universo LÍQUIDO
    caminho_universo = CFG["universo_liquido_file"]
    if Path(caminho_universo).exists():
        with open(caminho_universo, 'r') as f:
            tickers = [linha.strip() for linha in f if linha.strip()]
        print(f"✓ Universo líquido: {len(tickers)} papéis")
    else:
        print("⚠️  Usando universo completo...")
        tickers = carregar_universo_completo()
    
    if CFG['testar_universo_reduzido']:
        # lista aleatoria de 5 tickers
        random.seed(42)
        tickers = random.sample(tickers, min(5, len(tickers)))
        print(f"⚠️  Testando universo reduzido: {tickers}")


    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    print(f"\n{'='*70}")
    print(f"VALIDAÇÃO: ESTRATEGIAS GEMINI")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    
    estrategias = {
        "TURNING POINT": ESTRATEGIA_TURNING_POINT,
        "THE STRIKE ZONE": ESTRATEGIA_THE_STRIKE_ZONE,
        "MEAN REVERSION FAST": ESTRATEGIA_MEAN_REVERSION_FAST,
    }
    
    resultados = {nome: {ano: [] for ano in anos} for nome in estrategias.keys()}
    
    todos_os_trades_detalhado = []

    # Loop por ticker (externo)
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        ticker_ok = False
        
        try:
            # WALK-FORWARD CORRETO: processar cada ANO separadamente
            for ano in anos:
                try:
                    # Preparar dados ATÉ o final do ano (não todos os anos)
                    df = preparar_df(ticker, date(ano, 12, 31))
                    
                    if df.empty or len(df) < 50:
                        continue
                    
                    ticker_ok = True
                    
                    # Rodar backtest
                    for nome_estrategia, estrategia in estrategias.items():
                        trades = backtest_estrategia(ticker, df, estrategia)
                        
                        if trades.empty:
                            continue
                        
                        trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                        
                        # Filtrar trades deste ano
                        inicio = pd.Timestamp(date(ano, 1, 1))
                        fim = pd.Timestamp(date(ano, 12, 31))
                        
                        trades_ano = trades[
                            (trades["data_entrada"] >= inicio) &
                            (trades["data_entrada"] <= fim)
                        ]
                        
                        if not trades_ano.empty:
                            resultados[nome_estrategia][ano].append(trades_ano)
                            todos_os_trades_detalhado.append(trades_ano)
                
                except Exception:
                    continue
            
            if ticker_ok:
                print("✓")
            else:
                print("skip")
        
        except Exception:
            print("erro")
            continue   
    

    # Análise
    print(f"\n{'='*70}")
    print("📊 ANÁLISE")
    print(f"{'='*70}\n")
    
    relatorio = {}
    
    for nome in estrategias.keys():
        print(f"\n{'='*60}")
        print(f"🔬 {nome}")
        print(f"{'='*60}\n")
        
        params = estrategias[nome]["parametros"]
        dias_medio = params["max_dias"]
        
        metricas_por_ano = []
        todos = []
        
        for ano in anos:
            lista = resultados[nome][ano]
            
            if not lista:
                metricas_por_ano.append({
                    "ano": ano,
                    "trades": 0,
                    "taxa": 0,
                    "ret_medio": 0,
                    "ret_anual": 0,
                })
                continue
            
            df_ano = pd.concat(lista, ignore_index=True)
            todos.append(df_ano)
            
            n = len(df_ano)
            taxa = (df_ano["retorno_%"] > 0).sum() / n
            ret_medio = df_ano["retorno_%"].mean()
            ret_anual = ret_medio * (252 / dias_medio)
            
            metricas_por_ano.append({
                "ano": ano,
                "trades": n,
                "taxa": taxa,
                "ret_medio": ret_medio,
                "ret_anual": ret_anual,
            })
        
        # Tabela
        print("📅 ANO A ANO:\n")
        
        df_metricas = pd.DataFrame(metricas_por_ano)
        
        print(f"{'Ano':<6} {'Trades':<8} {'Taxa':<10} {'Ret/trade':<12} {'Ret Anual':<12}")
        print(f"{'-'*60}")
        
        for _, row in df_metricas.iterrows():
            ano_str = str(int(row['ano']))
            trades_str = str(int(row['trades']))
            taxa_str = f"{row['taxa']:.1%}" if row['trades'] > 0 else "-"
            ret_str = f"{row['ret_medio']:.2f}%" if row['trades'] > 0 else "-"
            anual_str = f"{row['ret_anual']:.1f}%" if row['trades'] > 0 else "-"
            
            print(f"{ano_str:<6} {trades_str:<8} {taxa_str:<10} {ret_str:<12} {anual_str:<12}")
        
        if not todos:
            print("\n❌ 0 trades.\n")
            relatorio[nome] = {"aprovado": False}
            continue
        
        # Consolidado
        df_total = pd.concat(todos, ignore_index=True)
        
        n_total = len(df_total)
        ops_ano = n_total / len([a for a in anos if a != ano_atual])
        taxa = (df_total["retorno_%"] > 0).sum() / n_total
        ret_medio = df_total["retorno_%"].mean()
        ret_std = df_total["retorno_%"].std()
        ret_anual = ret_medio * (252 / dias_medio)
        
        sharpe = (ret_medio / ret_std * np.sqrt(252 / dias_medio)) if ret_std > 0 else 0
        
        # Drawdown
        df_total = df_total.sort_values("data_entrada").reset_index(drop=True)
        df_total["equity"] = (1 + df_total["retorno_%"] / 100).cumprod()
        df_total["peak"] = df_total["equity"].cummax()
        df_total["drawdown"] = (df_total["equity"] / df_total["peak"] - 1) * 100
        max_dd = df_total["drawdown"].min()
        
        # Anos
        anos_com = df_metricas[df_metricas["trades"] > 0]
        anos_pos = (anos_com["ret_anual"] > 0).sum()
        anos_neg = (anos_com["ret_anual"] < 0).sum()
        pior = anos_com["ret_anual"].min() if len(anos_com) > 0 else 0
        melhor = anos_com["ret_anual"].max() if len(anos_com) > 0 else 0
        
        print(f"\n📈 CONSOLIDADO:\n")
        print(f"  {n_total} trades | {ops_ano:.1f}/ano")
        print(f"  Taxa: {taxa:.1%} | Sharpe: {sharpe:.2f}")
        print(f"  Ret/trade: {ret_medio:.2f}% | Ret anual: {ret_anual:.1f}%")
        print(f"  DD: {max_dd:.1f}% | Pior: {pior:.1f}% | Melhor: {melhor:.1f}%")
        print(f"  Anos +/-: {anos_pos}/{anos_neg}")
        
        relatorio[nome] = {
            "trades": n_total,
            "ops_ano": ops_ano,
            "taxa": taxa * 100,
            "sharpe": sharpe,
            "ret_anual": ret_anual,
            "drawdown": max_dd,
            "pior": pior,
            "melhor": melhor,
        } 

    

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Screening B3 - Momentum Relativo")
    parser.add_argument("--calibrar",  action="store_true",
                        help="Roda backtest no universo completo e gera lista de promissores")
    parser.add_argument("--validar-alertas", action="store_true",
                        help="Valida se classificação de alertas tem valor preditivo")
    parser.add_argument("--testar-prazos", action="store_true",
                        help="Testa múltiplos valores de max_dias para encontrar o prazo ótimo")
    parser.add_argument("--testar-alvos-stops", action="store_true",
                        help="Testa múltiplas combinações de alvo e stop")
    parser.add_argument("--validar-walkforward", action="store_true",
                        help="Valida robustez de alvos/stops através de janelas anuais (evita overfitting)")
    parser.add_argument("--validar-papeis-regime", action="store_true",
                        help="Valida papéis individuais com filtro de regime - GERA LISTA FINAL")
    parser.add_argument("--testar-scoring", action="store_true",
                        help="Testa sistema de scoring adaptativo com walk-forward")
    parser.add_argument("--testar-mean-reversion", action="store_true",
                        help="Testa estratégia de reversão à média (ALTERNATIVA ao momentum)")
    parser.add_argument("--comparar-mean-reversion", action="store_true",
                        help="Compara LIGHT vs SUPER-LIGHT lado a lado (frequência vs qualidade)")
    parser.add_argument("--testar-alvos-stops-light-fixos", action="store_true",
                        help="Testa alvos/stops fixos (%) para LIGHT - OTIMIZAÇÃO FINAL")
    parser.add_argument("--testar-alvos-stops-light-atr", action="store_true",
                        help="Testa alvos/stops baseados em ATR para LIGHT")
    parser.add_argument("--otimizar-light-completo", action="store_true",
                        help="🎯 OTIMIZAÇÃO COMPLETA: Fixos + ATR + Comparação (40-50min)")
    parser.add_argument("--testar-breakout", action="store_true",
                        help="🔬 DIAGNÓSTICO BREAKOUT: Adiciona filtros incrementalmente (10min)")
    parser.add_argument("--testar-pullback", action="store_true",
                        help="🎯 TESTE PULLBACK: Diagnóstico incremental (15min)")
    parser.add_argument("--pullback-perfis", action="store_true",
                        help="🏆 PULLBACK PERFIS FINAIS: 3 configurações diferentes (10min)")
    parser.add_argument("--testar-swing", action="store_true",
                        help="🎯 SWING HIGHS/LOWS: 4 perfis (Conservador|Agressivo|Scalper|Posicional) (15min)")
    parser.add_argument("--validar-swing", action="store_true",
                        help="🔬 VALIDAÇÃO PROFUNDA: Swing Scalper & Posicional (walk-forward, drawdown, distribuição) (15min)")
    parser.add_argument("--swing-seletivo", action="store_true",
                        help="🎯 SWING SELETIVO: 3 versões com meta 20-30 ops/ano (walk-forward correto) (20min)")
    parser.add_argument("--swing-balanceado", action="store_true",
                        help="⚖️  SWING BALANCEADO: Relaxar SELETIVO A para 25-40 ops/ano (walk-forward correto) (20min)")
    parser.add_argument("--max-dias", type=int, default=10,
                        help="Prazo máximo para testes (padrão: 10)")
    parser.add_argument("--data",      type=str, default=None,
                        help="Data base YYYY-MM-DD (padrão: hoje)")
    parser.add_argument("--ticker",    type=str, default=None,
                        help="Rodar em um ticker específico (debug)")
    
    parser.add_argument("--testar-mean-rev-light", action="store_true",
                        help="Testa MEAN REVERSION LIGHT vs SUPER-LIGHT no universo líquido (validação incremental)")
    parser.add_argument("--testar-swing-balanceado", action="store_true",
                        help="Testa SWING BALANCEADO A vs B vs C no universo líquido (validação incremental)")
    parser.add_argument("--testar-deteccoes-swing", action="store_true",
                        help="🔬 Testa 3 detecções aprimoradas de swing low: CONFIRMADO, CAPITULAÇÃO, RIGOROSO")
    parser.add_argument("--testar-estrategias-gemini", action="store_true",
                        help="Testa estratégias GEMINI (momentum + mean reversion) com walk-forward")
    args = parser.parse_args()
    
    data_base = date.fromisoformat(args.data) if args.data else date.today()
    
    if args.calibrar:
        calibrar_papeis_promissores(data_base)
    elif args.validar_alertas:
        backtest_estratificado_por_alerta(data_base)
    elif args.testar_prazos:
        testar_multiplos_prazos(data_base)
    elif args.testar_alvos_stops:
        testar_alvos_e_stops(data_base, max_dias_fixo=args.max_dias)
    elif args.validar_walkforward:
        validar_walkforward_alvos_stops(data_base, max_dias_fixo=args.max_dias)
    elif args.validar_papeis_regime:
        validar_papeis_individuais_com_regime(data_base)
    elif args.testar_scoring:
        testar_scoring_system_walkforward(data_base)
    elif args.testar_mean_reversion:
        testar_mean_reversion_walkforward(data_base)
    elif args.comparar_mean_reversion:
        comparar_niveis_mean_reversion(data_base)
    elif args.testar_alvos_stops_light_fixos:
        testar_alvos_stops_light_fixos(data_base)
    elif args.testar_alvos_stops_light_atr:
        testar_alvos_stops_light_atr(data_base)
    elif args.otimizar_light_completo:
        otimizar_alvos_stops_light_completo(data_base)
    elif args.testar_breakout:
        diagnosticar_breakout_incremental(data_base)  # Diagnóstico incremental
    elif args.testar_pullback:
        testar_pullback_completo(data_base)  # Teste completo de Pullback
    elif args.pullback_perfis:
        testar_pullback_perfis_finais(data_base)  # Teste final: 3 perfis
    elif args.testar_swing:
        testar_swing_completo(data_base)  # Teste Swing: 4 perfis
    elif args.validar_swing:
        validar_swing_profundamente(data_base)  # Validação profunda
    elif args.swing_seletivo:
        validar_swing_seletivo(data_base)  # Seletivo 20-30 ops/ano
    elif args.swing_balanceado:
        validar_swing_balanceado(data_base)  # Balanceado 25-40 ops/ano
    elif args.testar_mean_rev_light:
        testar_mean_rev_light(data_base)  # Teste MEAN REVERSION
    elif args.testar_swing_balanceado:
        testar_swing_balanceado(data_base)  # Teste SWING BALANCEADO
    elif args.testar_deteccoes_swing:
        testar_deteccoes_swing_low(data_base)  # Teste DETECÇÕES APRIMORADAS
    elif args.testar_estrategias_gemini:
        testar_estrategias_gemini(data_base)  # Teste ESTRATÉGIAS GEMINI
    else:
        rodar_screening(data_base, ticker_unico=args.ticker)


if __name__ == "__main__":
    main()