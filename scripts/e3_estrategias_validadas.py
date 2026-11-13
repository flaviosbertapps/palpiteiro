#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3 ESTRATÉGIAS VALIDADAS - Backup para Re-testes

Este arquivo contém APENAS as estratégias que passaram na validação rigorosa.
Use para re-testar ou comparar novas estratégias.

Estratégias Aprovadas:
1. Mean Reversion LIGHT
2. Swing BALANCEADO A

Estratégias Reprovadas (para referência):
- Momentum
- Breakout
- Pullback
- Swing SCALPER (muito frequente)
- Swing POSICIONAL (muito raro)
"""

# ==============================================================================
# ✅ ESTRATÉGIAS APROVADAS
# ==============================================================================

ESTRATEGIA_MEAN_REVERSION_LIGHT = {
    "nome": "mean_reversion_light",
    "descricao": "Mean Reversion LIGHT - APROVADO",
    
    "metricas_validacao": {
        "ops_ano": 3.5,
        "taxa_acerto": 81.7,
        "sharpe": 1.38,
        "retorno_anual": 48.9,
        "drawdown_max": -15.0,
        "anos_negativos": 0,
        "pior_ano": 35.0,
    },
    
    "parametros": {
        "rsi_min": 0,
        "rsi_max": 30,
        "close_abaixo_mm20": True,
        "ibov_acima_mm200": True,
        "alvo_pct": 0.14,
        "stop_pct": 0.06,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        row.get("rsi", 100) <= p["rsi_max"]
        and row.get("close", 0) < row.get("mm20", float('inf'))
        and row.get("close_ibov", 0) > row.get("mm200_ibov", 0)
    ),
}

ESTRATEGIA_SWING_BALANCEADO_A = {
    "nome": "swing_balanceado",
    "descricao": "Swing BALANCEADO A - APROVADO",
    
    "metricas_validacao": {
        "ops_ano": 27.2,
        "taxa_acerto": 86.0,
        "sharpe": 5.19,
        "retorno_anual": 57.8,
        "drawdown_max": -8.7,
        "anos_negativos": 0,
        "pior_ano": 52.8,
    },
    
    "parametros": {
        "janela_swing": 3,
        "max_distancia_suporte": 0.025,
        "vol_relativo_min": 0.8,
        "rsi_min": 25,
        "rsi_max": 55,
        "close_acima_mm50": True,
        "alvo_pct": 0.06,
        "stop_pct": 0.03,
        "max_dias": 20,
    },
    
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela_swing"])
        and row.get("close", 0) <= row.get("swing_low", float('inf')) * (1 + p["max_distancia_suporte"])
        and row.get("vol_relativo", 0) >= p["vol_relativo_min"]
        and row.get("rsi", 0) >= p["rsi_min"]
        and row.get("rsi", 100) <= p["rsi_max"]
        and row.get("close", 0) > row.get("mm50", 0)
    ),
}

# ==============================================================================
# 📊 COMPARAÇÃO DAS APROVADAS
# ==============================================================================

COMPARACAO_ESTRATEGIAS = """
┌──────────────────────┬──────────┬──────┬────────┬──────────┬──────────┐
│ Estratégia           │ Ops/ano  │ Taxa │ Sharpe │ Ret/ano  │ DD Máx   │
├──────────────────────┼──────────┼──────┼────────┼──────────┼──────────┤
│ Mean Reversion LIGHT │   3.5    │ 81.7%│  1.38  │  48.9%   │  -15.0%  │
│ Swing BALANCEADO     │  27.2    │ 86.0%│  5.19  │  57.8%   │   -8.7%  │
└──────────────────────┴──────────┴──────┴────────┴──────────┴──────────┘

PORTFÓLIO RECOMENDADO:
- 70% Swing BALANCEADO (frequência + qualidade)
- 30% Mean Reversion LIGHT (diversificação + alta taxa)
= ~21 operações/ano no total
"""

# ==============================================================================
# ❌ ESTRATÉGIAS REPROVADAS (para referência histórica)
# ==============================================================================

ESTRATEGIAS_REPROVADAS = {
    "momentum": {
        "motivo": "Taxa 45% (aleatório)",
        "ops_ano": 0,
        "taxa": 45.0,
    },
    "breakout": {
        "motivo": "Taxa 50-53% (aleatório)",
        "ops_ano": 115.4,
        "taxa": 52.9,
    },
    "pullback": {
        "motivo": "Taxa 41-49% (aleatório)",
        "ops_ano": 60.6,
        "taxa": 49.2,
    },
    "swing_scalper": {
        "motivo": "Aprovado mas MUITO FREQUENTE (72.8 ops/ano = inoperável manualmente)",
        "ops_ano": 72.8,
        "taxa": 74.7,
        "sharpe": 5.23,
        "nota": "Excelente estratégia MAS requer automação",
    },
    "swing_posicional": {
        "motivo": "Aprovado mas MUITO RARO (2.4 ops/ano = amostra insuficiente)",
        "ops_ano": 2.4,
        "taxa": 91.7,
        "sharpe": 3.58,
        "nota": "Taxa altíssima mas trades insuficientes",
    },
}

# ==============================================================================
# 🔧 HELPER FUNCTIONS
# ==============================================================================

def _eh_swing_low(row, janela: int) -> bool:
    """Verifica se é swing low (mínima local)."""
    swing_col = f"swing_low_{janela}"
    return row.get(swing_col, False) == True


# ==============================================================================
# 📝 NOTAS DE VALIDAÇÃO
# ==============================================================================

NOTAS_VALIDACAO = """
CRITÉRIOS DE APROVAÇÃO (todos devem ser atendidos):
✅ Taxa de acerto >= 65%
✅ Sharpe Ratio >= 1.0
✅ Retorno anualizado >= 30%
✅ Drawdown máximo >= -30%
✅ Pior ano >= -20%
✅ 0 anos negativos (desejável)
✅ Mínimo 20 trades (significância estatística)

PERÍODO DE VALIDAÇÃO:
- Walk-forward 2022-2026 (5 anos)
- Dados específicos por ano (sem lookahead bias)
- Testado em 69 papéis líquidos B3

METODOLOGIA:
1. Backtest ano a ano separado
2. Cálculo de métricas consolidadas
3. Análise de drawdown
4. Distribuição de saídas (alvo/stop/tempo)
5. Consistência entre anos
"""

if __name__ == "__main__":
    print("="*70)
    print("E3 ESTRATÉGIAS VALIDADAS")
    print("="*70)
    print(COMPARACAO_ESTRATEGIAS)
    print("\n" + NOTAS_VALIDACAO)
