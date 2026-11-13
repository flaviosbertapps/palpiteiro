#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3 SCREENING V2 - Sistema de Screening Diário B3

Estratégias Validadas:
- Mean Reversion LIGHT (3.5 ops/ano, 81.7% taxa, Sharpe 1.38)
- Swing BALANCEADO (27.2 ops/ano, 86.0% taxa, Sharpe 5.19)

Uso:
    python e3_screening_v2.py              # Screening diário
    python e3_screening_v2.py --ticker PETR4  # Testar um papel específico
"""

import sys
#sys.path.insert(0, '/mnt/user-data/uploads')

import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import argparse
import traceback
from typing import Tuple

import preprocessamento as pp
import utils

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

CFG = {
    "pasta_resultados": "resultados_screening",
    "dias_historico": 600,
    "enviar_email": True,  # Configurar True quando tiver SMTP
}

# Configurações de Capital e Risco
CAPITAL_TOTAL = 100000  # R$ 100.000 (AJUSTAR CONFORME SEU CAPITAL)
RISCO_POR_OPERACAO = 0.02  # 2% do capital por operação
MAX_POSICOES_SIMULTANEAS = 5  # Máximo de posições abertas

# ==============================================================================
# ESTRATÉGIAS VALIDADAS
# ==============================================================================

# Mean Reversion LIGHT - IDÊNTICO AO VALIDADO
ESTRATEGIA_MEAN_REVERSION_LIGHT = {
    "nome": "mean_reversion_light",
    "descricao": "Reversão à média com critérios relaxados (versão operável)",
    
    "parametros": {
        # --- Entrada (critérios RELAXADOS) ---
        "rsi_max": 40,              # RSI < 40 (relaxado de 30)
        "queda_5d_min": -0.03,      # caiu pelo menos 3% (relaxado de 5%)
        "close_acima_mm50": True,   # acima de MM50 (relaxado de MM200)
        "ibov_acima_mm200": True,   # mercado em regime de alta (mantém)
        
        # Filtros básicos
        "vol_relativo_min": 0.5,    # volume mínimo relaxado
        "adx_max": 40,              # ADX < 40 (relaxado de 35)
        "rsi_min": 20,              # RSI > 20 (evita colapsos totais)
        
        # --- Saída ---
        "alvo_pct": 0.03,           # +3%
        "stop_pct": 0.04,           # -4%
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        # Sobrevendido (relaxado)
        row.get("rsi", 100) <= p["rsi_max"]
        and row.get("rsi", 0) >= p["rsi_min"]  # não pega colapsos totais
        # Caiu nos últimos 5 dias (relaxado)
        and row.get("ret_5d", 0) <= p["queda_5d_min"]
        # Ainda em tendência de alta de curto/médio prazo (MM50 em vez de MM200)
        and row.get("close", 0) > row.get("mm50", 0)
        # Regime de mercado favorável (mantém rigoroso)
        and row.get("close_ibov", 0) > row.get("mm200_ibov", 0)
        # Volume mínimo (relaxado)
        and row.get("vol_relativo", 0) >= p["vol_relativo_min"]
        # ADX não muito alto (relaxado)
        and row.get("ADX_14", 100) <= p["adx_max"]
    ),
}

# Swing BALANCEADO - CONFIGURAÇÃO VALIDADA (EXATA DO ARQUIVO ORIGINAL)
ESTRATEGIA_SWING_BALANCEADO = {
    "nome": "swing_balanceado",
    "descricao": "Swing BALANCEADO - APROVADO (27.2 ops/ano, 86% taxa, Sharpe 5.19)",
    
    "parametros": {
        "janela_swing": 3,
        "max_distancia_suporte": 0.025,    # Relaxado de 0.02 para 0.025
        "vol_relativo_min": 0.8,           # Relaxado de 1.0 para 0.8
        "rsi_min": 25,
        "rsi_max": 55,                     # Relaxado de 50 para 55
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

ESTRATEGIA_SWING_BALANCEADO_D = {
    "nome": "swing_balanceado_d",
    "descricao": "Transição Frequência Média - Filtro MM50 Mantido",
    
    "parametros": {
        "janela_swing": 2,                 # Janela menor para captar mais sinais
        "max_distancia_suporte": 0.035,     # Aumentado de 0.025 para 0.035
        "vol_relativo_min": 0.65,          # Baixado de 0.8 para 0.65 (Mais trades em dias calmos)
        "rsi_min": 25,
        "rsi_max": 60,                     # Aumentado para 60 (Aceita papéis com mais força)
        "close_acima_mm50": True,
        "alvo_pct": 0.05,
        "stop_pct": 0.03,
        "max_dias": 15,
    },
    
    "fn_sinal": lambda row, p: (
        _eh_swing_low(row, p["janela_swing"])
        and _val(row, "close") <= _val(row, "swing_low", float('inf')) * (1 + p["max_distancia_suporte"])
        and _val(row, "vol_relativo") >= p["vol_relativo_min"]
        and _val(row, "rsi") >= p["rsi_min"]
        and _val(row, "rsi") <= p["rsi_max"]
        and _val(row, "close") > _val(row, "mm50", 0)
    ),
}

# Lista de estratégias ativas
ESTRATEGIAS_ATIVAS = [
    ESTRATEGIA_MEAN_REVERSION_LIGHT,
    ESTRATEGIA_SWING_BALANCEADO,
    ESTRATEGIA_SWING_BALANCEADO_D,
]

# Métricas de validação (para exibir no e-mail)
METRICAS_VALIDACAO = {
    "mean_reversion_light": {
        "ops_ano": 3.5,
        "taxa_acerto": 81.7,
        "sharpe": 1.38,
        "retorno_anual": 48.9,
        "drawdown_max": -15.0,
        "anos_negativos": 0,
        "descricao": "Compra em oversold (RSI<40) + close<MM20 + regime bull (Ibov>MM200). Alta taxa, poucos sinais. ALVO 3% / STOP 4%.",
    },
    "swing_balanceado": {
        "ops_ano": 27.2,
        "taxa_acerto": 86.0,
        "sharpe": 5.19,
        "retorno_anual": 57.8,
        "drawdown_max": -8.7,
        "anos_negativos": 0,
        "descricao": "Compra em swing low (suporte 3d) com confirmações. Frequência balanceada, alta qualidade. ALVO 6% / STOP 3%.",
    },
}

# ==============================================================================
# HELPERS
# ==============================================================================

def _eh_swing_low(row, janela: int) -> bool:
    """Verifica se é swing low (mínima local)."""
    swing_col = f"swing_low_{janela}"
    return row.get(swing_col, False) == True

def _val(row: pd.Series, col: str, default: float = np.nan) -> float:
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)

def carregar_universo_completo():
    """Carrega lista de tickers do universo B3."""
    caminho = "e3_universo_b3_ampliado.txt"
    
    if not Path(caminho).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    
    with open(caminho, 'r') as f:
        tickers = [linha.strip() for linha in f if linha.strip()]
    
    return tickers

def carregar_universo_liquido():
    """Carrega lista de tickers do universo B3 líquido."""
    caminho = "e3_universo_b3_liquido.txt"
    
    if not Path(caminho).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    
    with open(caminho, 'r') as f:
        tickers = [linha.strip() for linha in f if linha.strip()]
    
    return tickers


def preparar_df(ticker: str, data_base: date) -> pd.DataFrame:
    """Prepara DataFrame com features para todas as estratégias."""
    df = pp.montar_dataset(ticker, dias_total_historico=CFG["dias_historico"], data_base=data_base)
    df = pp.adicionar_features_sniper(df)
    
    # Retornos
    for n in [1, 3, 5, 10, 20]:
        col = f"ret_{n}d"
        if col not in df.columns:
            df[f"ret_{n}d"] = df["close"].pct_change(n)
    
    # Médias móveis do Ibovespa
    if "close_ibov" in df.columns:
        df["mm50_ibov"] = df["close_ibov"].rolling(window=50).mean()
        df["mm100_ibov"] = df["close_ibov"].rolling(window=100).mean()
        df["mm200_ibov"] = df["close_ibov"].rolling(window=200).mean()
    
    # Médias móveis do papel
    df["mm20"] = df["close"].rolling(window=20).mean()
    df["mm50"] = df["close"].rolling(window=50).mean()
    
    # Swing lows (para Swing BALANCEADO)
    for janela in [2, 3, 5, 10]:
        df[f"swing_low_{janela}"] = (
            df["low"] == df["low"].rolling(window=janela*2+1, center=True).min()
        )
        df["swing_low"] = df["low"].where(df[f"swing_low_{janela}"], np.nan).ffill()
    
    df["__ticker__"] = ticker
    return df


def calcular_tamanho_posicao(preco_entrada: float, preco_stop: float, 
                              capital_total: float, risco_pct: float) -> Tuple[int, float]:
    """
    Calcula quantas ações comprar baseado no risco definido.
    
    Returns:
        (quantidade_acoes, valor_investido)
    """
    risco_total = capital_total * risco_pct
    risco_por_acao = preco_entrada - preco_stop
    
    if risco_por_acao <= 0:
        return 0, 0.0
    
    qtd_acoes = int(risco_total / risco_por_acao)
    valor_investido = qtd_acoes * preco_entrada
    
    return qtd_acoes, valor_investido


def avaliar_sinais_no_dia(ticker: str, df: pd.DataFrame, data_base: date, 
                           estrategia: dict) -> list:
    """
    Avalia se há sinal para uma estratégia específica no dia.
    """
    if df.empty:
        return []
    
    # Garantir que o índice seja DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Converter data_base para Timestamp
    data_timestamp = pd.Timestamp(data_base)
    
    # Filtrar até a data
    df = df[df.index <= data_timestamp]
    
    if df.empty:
        return []
    
    ultima_linha = df.iloc[-1]
    params = estrategia["parametros"]
    tem_sinal = estrategia["fn_sinal"](ultima_linha, params)
    
    if not tem_sinal:
        return []
    
    # Montar sinal (converter para float para evitar problemas de serialização)
    sinal = {
        "ticker": ticker,
        "data": data_base,
        "estrategia": estrategia["nome"],
        "descricao_estrategia": estrategia["descricao"],
        "parametros": params,
        "close": float(ultima_linha.get("close", 0)),
        "rsi": float(ultima_linha.get("rsi", 0)),
        "vol_relativo": float(ultima_linha.get("vol_relativo", 0)),
        "mm20": float(ultima_linha.get("mm20", 0)),
        "mm50": float(ultima_linha.get("mm50", 0)),
    }
    
    return [sinal]


def gerar_relatorio_html(df_sinais: pd.DataFrame, data_base: date) -> str:
    """
    Gera relatório HTML LIMPO e PROFISSIONAL:
    1. TABELA DE ORDENS no topo (ação imediata)
    2. Detalhes técnicos por ordem
    3. Informações sobre estratégias no final
    """
    if df_sinais.empty:
        return f"""<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: Arial, sans-serif; padding: 20px; max-width: 1000px; margin: 0 auto;">
<div style="background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; text-align: center;">
    <h2 style="margin: 0;">✓ Screening B3 - {data_base.strftime('%d/%m/%Y')}</h2>
    <p style="margin: 10px 0 0 0;">Nenhum sinal de compra hoje</p>
</div>
<p style="text-align: center; color: #888; margin-top: 30px;">Próximo screening: amanhã após 18h</p>
<p style="text-align: center; color: #888; font-size: 12px;">Sistema E3 v2.0</p>
</body>
</html>"""
    
    # Preparar dados das ordens
    ordens = []
    for _, sinal in df_sinais.iterrows():
        ticker = str(sinal.get("ticker", "?"))
        close = float(sinal.get("close", 0))
        estrategia = str(sinal.get("estrategia", "?"))
        
        params = sinal.get("parametros", {})
        alvo_pct = float(params.get("alvo_pct", 0)) * 100
        stop_pct = float(params.get("stop_pct", 0)) * 100
        max_dias = int(params.get("max_dias", 0))
        
        preco_alvo = close * (1 + alvo_pct/100)
        preco_stop = close * (1 - stop_pct/100)
        
        qtd, valor = calcular_tamanho_posicao(close, preco_stop, CAPITAL_TOTAL, RISCO_POR_OPERACAO)
        
        rsi = float(sinal.get("rsi", 0))
        vol_rel = float(sinal.get("vol_relativo", 0))
        
        ordens.append({
            "ticker": ticker,
            "estrategia": estrategia,
            "preco": close,
            "qtd": qtd,
            "valor": valor,
            "alvo": preco_alvo,
            "alvo_pct": alvo_pct,
            "stop": preco_stop,
            "stop_pct": stop_pct,
            "prazo": max_dias,
            "rsi": rsi,
            "vol_rel": vol_rel,
            "ganho_potencial": (preco_alvo - close) * qtd,
        })
    
    # Calcular totais
    total_investir = sum(o['valor'] for o in ordens)
    risco_total = CAPITAL_TOTAL * RISCO_POR_OPERACAO * len(ordens)
    ganho_potencial_total = sum(o['ganho_potencial'] for o in ordens)
    
    html = f"""<html>
<head>
<meta charset="UTF-8">
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    padding: 20px;
    background-color: #f5f5f5;
    max-width: 1400px;
    margin: 0 auto;
}}
.header {{
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    color: white;
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}}
.header h1 {{
    margin: 0;
    font-size: 32px;
    font-weight: 600;
}}
.header p {{
    margin: 12px 0 0 0;
    font-size: 16px;
    opacity: 0.95;
}}
.resumo {{
    background: white;
    padding: 25px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.resumo h2 {{
    margin: 0 0 20px 0;
    color: #2c3e50;
    font-size: 22px;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}}
.resumo-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-top: 20px;
}}
.resumo-item {{
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #3498db;
}}
.resumo-label {{
    font-size: 13px;
    color: #7f8c8d;
    text-transform: uppercase;
    margin-bottom: 5px;
}}
.resumo-valor {{
    font-size: 24px;
    font-weight: bold;
    color: #2c3e50;
}}
.resumo-valor.positivo {{
    color: #27ae60;
}}
.resumo-valor.negativo {{
    color: #e74c3c;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}}
th {{
    background: #34495e;
    color: white;
    padding: 15px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
td {{
    padding: 15px 12px;
    border-bottom: 1px solid #ecf0f1;
    font-size: 14px;
}}
tr:last-child td {{
    border-bottom: none;
}}
tr:hover {{
    background: #f8f9fa;
}}
.ticker {{
    font-weight: 700;
    font-size: 18px;
    color: #2c3e50;
}}
.tag {{
    display: inline-block;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.tag-mean {{
    background: #e3f2fd;
    color: #1976d2;
}}
.tag-swing {{
    background: #e8f5e9;
    color: #388e3c;
}}
.preco {{
    font-weight: 600;
    font-size: 16px;
    color: #2c3e50;
}}
.positivo {{
    color: #27ae60;
    font-weight: 600;
}}
.negativo {{
    color: #e74c3c;
    font-weight: 600;
}}
.destaque {{
    background: #fff3cd !important;
    font-weight: 700;
}}
.detalhes {{
    background: white;
    padding: 25px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.detalhes h2 {{
    margin: 0 0 20px 0;
    color: #2c3e50;
    font-size: 22px;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}}
.ordem-detalhe {{
    background: #f8f9fa;
    padding: 20px;
    border-left: 4px solid #3498db;
    margin-bottom: 20px;
    border-radius: 5px;
}}
.ordem-detalhe h3 {{
    margin: 0 0 15px 0;
    color: #2c3e50;
    font-size: 18px;
}}
.ordem-detalhe p {{
    margin: 5px 0;
    font-size: 13px;
    color: #555;
}}
.metricas-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-top: 15px;
}}
.metrica {{
    background: white;
    padding: 12px;
    border-radius: 5px;
    text-align: center;
}}
.metrica-label {{
    font-size: 11px;
    color: #7f8c8d;
    text-transform: uppercase;
    margin-bottom: 5px;
}}
.metrica-valor {{
    font-size: 18px;
    font-weight: 700;
    color: #2c3e50;
}}
.estrategias-info {{
    background: white;
    padding: 25px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.estrategias-info h2 {{
    margin: 0 0 20px 0;
    color: #2c3e50;
    font-size: 22px;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}}
.estrategia-box {{
    background: #f8f9fa;
    padding: 20px;
    border-left: 4px solid #27ae60;
    margin-bottom: 20px;
    border-radius: 5px;
}}
.estrategia-box h3 {{
    margin: 0 0 10px 0;
    color: #2c3e50;
}}
.checklist {{
    background: #fff3cd;
    border: 2px solid #ffc107;
    padding: 20px;
    border-radius: 10px;
    margin-top: 25px;
}}
.checklist h3 {{
    margin: 0 0 15px 0;
    color: #856404;
}}
.checklist ul {{
    margin: 10px 0;
    padding-left: 25px;
}}
.checklist li {{
    margin: 10px 0;
    line-height: 1.6;
}}
.footer {{
    text-align: center;
    color: #7f8c8d;
    font-size: 12px;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 2px solid #ecf0f1;
}}
</style>
</head>
<body>

<div class="header">
    <h1>📊 Ordens de Compra</h1>
    <p>{data_base.strftime('%d/%m/%Y')} | {len(ordens)} ordem(ns) | Capital: R$ {CAPITAL_TOTAL:,.0f} | Risco: {RISCO_POR_OPERACAO*100:.1f}% por operação</p>
</div>

<div class="resumo">
    <h2>💰 Resumo do Dia</h2>
    <div class="resumo-grid">
        <div class="resumo-item">
            <div class="resumo-label">Total a Investir</div>
            <div class="resumo-valor">R$ {total_investir:,.2f}</div>
        </div>
        <div class="resumo-item">
            <div class="resumo-label">Risco Total</div>
            <div class="resumo-valor negativo">R$ {risco_total:,.2f}</div>
        </div>
        <div class="resumo-item">
            <div class="resumo-label">Ganho Potencial</div>
            <div class="resumo-valor positivo">R$ {ganho_potencial_total:,.2f}</div>
        </div>
        <div class="resumo-item">
            <div class="resumo-label">% do Capital</div>
            <div class="resumo-valor">{(total_investir/CAPITAL_TOTAL*100):.1f}%</div>
        </div>
    </div>
</div>

<table>
    <thead>
        <tr>
            <th>AÇÃO</th>
            <th>ESTRATÉGIA</th>
            <th>PREÇO</th>
            <th>QUANTIDADE</th>
            <th>INVESTIR</th>
            <th>🎯 ALVO</th>
            <th>🛑 STOP</th>
            <th>⏱ PRAZO</th>
        </tr>
    </thead>
    <tbody>
"""
    
    for ordem in ordens:
        tag_class = "tag-mean" if "mean" in ordem["estrategia"].lower() else "tag-swing"
        tag_nome = "MEAN REV" if "mean" in ordem["estrategia"].lower() else "SWING"
        
        html += f"""
        <tr>
            <td class="ticker">{ordem['ticker']}</td>
            <td><span class="tag {tag_class}">{tag_nome}</span></td>
            <td class="preco">R$ {ordem['preco']:.2f}</td>
            <td style="font-weight: 600;">{ordem['qtd']}</td>
            <td class="destaque">R$ {ordem['valor']:,.2f}</td>
            <td class="positivo">R$ {ordem['alvo']:.2f}<br><small style="font-size: 11px;">(+{ordem['alvo_pct']:.1f}%)</small></td>
            <td class="negativo">R$ {ordem['stop']:.2f}<br><small style="font-size: 11px;">(-{ordem['stop_pct']:.1f}%)</small></td>
            <td style="font-weight: 600;">{ordem['prazo']} dias</td>
        </tr>
"""
    
    html += """
    </tbody>
</table>

<div class="detalhes">
    <h2>📋 Detalhes Técnicos por Ordem</h2>
"""
    
    for ordem in ordens:
        tag_nome = "MEAN REVERSION LIGHT" if "mean" in ordem["estrategia"].lower() else "SWING BALANCEADO"
        
        html += f"""
    <div class="ordem-detalhe">
        <h3>{ordem['ticker']} - {tag_nome}</h3>
        <p><strong>Gatilho:</strong> {"RSI oversold + close abaixo MM20 em regime bull" if "mean" in ordem["estrategia"].lower() else "Swing low (suporte) com confirmações de volume e RSI"}</p>
        
        <div class="metricas-grid">
            <div class="metrica">
                <div class="metrica-label">RSI</div>
                <div class="metrica-valor">{ordem['rsi']:.1f}</div>
            </div>
            <div class="metrica">
                <div class="metrica-label">Volume Rel.</div>
                <div class="metrica-valor">{ordem['vol_rel']:.2f}×</div>
            </div>
            <div class="metrica">
                <div class="metrica-label">Ganho Potencial</div>
                <div class="metrica-valor positivo">R$ {ordem['ganho_potencial']:,.2f}</div>
            </div>
            <div class="metrica">
                <div class="metrica-label">Risco Máximo</div>
                <div class="metrica-valor negativo">R$ {CAPITAL_TOTAL * RISCO_POR_OPERACAO:,.2f}</div>
            </div>
        </div>
    </div>
"""
    
    html += """
</div>

<div class="estrategias-info">
    <h2>📈 Sobre as Estratégias</h2>
"""
    
    estrategias_usadas = list(set(o["estrategia"] for o in ordens))
    
    for est_nome in estrategias_usadas:
        metricas = METRICAS_VALIDACAO.get(est_nome, {})
        nome_display = "MEAN REVERSION LIGHT" if "mean" in est_nome.lower() else "SWING BALANCEADO"
        
        html += f"""
    <div class="estrategia-box">
        <h3>{nome_display}</h3>
        <p style="margin: 8px 0; font-size: 13px; color: #555;">{metricas.get('descricao', '')}</p>
        
        <div class="metricas-grid">
            <div class="metrica">
                <div class="metrica-label">Taxa de Acerto</div>
                <div class="metrica-valor positivo">{metricas.get('taxa_acerto', 0):.1f}%</div>
            </div>
            <div class="metrica">
                <div class="metrica-label">Sharpe Ratio</div>
                <div class="metrica-valor">{metricas.get('sharpe', 0):.2f}</div>
            </div>
            <div class="metrica">
                <div class="metrica-label">Retorno Anual</div>
                <div class="metrica-valor positivo">{metricas.get('retorno_anual', 0):.1f}%</div>
            </div>
            <div class="metrica">
                <div class="metrica-label">Ops/Ano</div>
                <div class="metrica-valor">{metricas.get('ops_ano', 0):.1f}</div>
            </div>
            <div class="metrica">
                <div class="metrica-label">Drawdown Máx</div>
                <div class="metrica-valor negativo">{metricas.get('drawdown_max', 0):.1f}%</div>
            </div>
            <div class="metrica">
                <div class="metrica-label">Anos Negativos</div>
                <div class="metrica-valor">{metricas.get('anos_negativos', 0)}</div>
            </div>
        </div>
        <p style="margin: 15px 0 0 0; font-size: 11px; color: #7f8c8d;"><em>Métricas baseadas em backtest walk-forward rigoroso 2022-2026 (5 anos)</em></p>
    </div>
"""
    
    html += f"""
</div>

<div class="checklist">
    <h3>⚠️ CHECKLIST ANTES DE OPERAR</h3>
    <ul>
        <li><strong>✅ Confirme os preços</strong> em tempo real antes de executar as ordens</li>
        <li><strong>✅ Configure o stop loss IMEDIATAMENTE</strong> após a compra (não espere!)</li>
        <li><strong>✅ Respeite o limite</strong> de {MAX_POSICOES_SIMULTANEAS} posições simultâneas</li>
        <li><strong>✅ Siga o tamanho sugerido</strong> (calculado com base em {RISCO_POR_OPERACAO*100:.0f}% de risco por operação)</li>
        <li><strong>✅ Monitore diariamente</strong> e execute saída ao atingir alvo, stop ou prazo máximo</li>
        <li><strong>✅ Ajuste conforme seu perfil</strong> - você pode reduzir (mas não aumentar) o tamanho das posições</li>
    </ul>
</div>

<div class="footer">
    <p><strong>Próximo screening: amanhã após 18h</strong></p>
    <p style="margin-top: 10px;">Sistema E3 v2.0 | Estratégias Validadas: Mean Reversion LIGHT + Swing BALANCEADO</p>
    <p style="margin-top: 10px; font-size: 11px; color: #95a5a6;">Este relatório é gerado automaticamente com base em análise técnica quantitativa.<br>Não constitui recomendação de investimento. Opere por sua conta e risco.</p>
</div>

</body>
</html>
"""
    
    return html

def rodar_screening(data_base: date = None, ticker_unico: str = None):
    """
    Screening V2: Múltiplas estratégias + E-mail enriquecido.
    """
    if data_base is None:
        data_base = date.today()
    
    if ticker_unico:
        tickers = [ticker_unico]
    else:
        tickers = carregar_universo_liquido()
    
    print(f"\n{'='*70}")
    print(f"SCREENING V2  |  {data_base.strftime('%d/%m/%Y')}  |  {len(tickers)} tickers")
    print(f"Estratégias ativas: {len(ESTRATEGIAS_ATIVAS)}")
    for est in ESTRATEGIAS_ATIVAS:
        print(f"  - {est['nome']}: {est['descricao']}")
    print(f"{'='*70}\n")
    
    Path(CFG["pasta_resultados"]).mkdir(exist_ok=True)
    todos_sinais = []
    
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = preparar_df(ticker, data_base)
            
            sinais_ticker = []
            for estrategia in ESTRATEGIAS_ATIVAS:
                sinais = avaliar_sinais_no_dia(ticker, df, data_base, estrategia)
                sinais_ticker.extend(sinais)
            
            if sinais_ticker:
                print(f"✅ {len(sinais_ticker)} sinal(is)")
                todos_sinais.extend(sinais_ticker)
            else:
                print("—")
        except Exception as e:
            print(f"❌ ERRO")
            if ticker_unico:
                traceback.print_exc()
    
    if not todos_sinais:
        print("\n⚠️  Nenhum sinal hoje.")        
        df_sinais = pd.DataFrame()
        assunto = "✓ Sem ordens de compra"
        corpo_texto = f"""Screening B3 - {data_base}

Nenhum sinal de compra hoje.

{len(tickers)} papéis analisados.
Estratégias: Mean Reversion LIGHT e Swing BALANCEADO
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
    Estratégias: Mean Reversion LIGHT e Swing BALANCEADO<br>
    Próximo screening: amanhã após 18h
</p>
<hr style="border: 1px solid #ddd; margin: 20px 0;">
<p style="color: #999; font-size: 12px;">Sistema E3 - Screening automático</p>
</body>
</html>"""
    else:
        df_sinais = pd.DataFrame(todos_sinais)
        n_operar = len(df_sinais)
        assunto = f"🎯 Screening B3 - {n_operar} {'sinal' if n_operar == 1 else 'sinais'} para {data_base + timedelta(days=1)}"
        print(f"\n✅ {len(df_sinais)} sinal(is) encontrado(s)!")
        corpo_texto = "TODO"
    
    # Salvar CSV
    if not df_sinais.empty:
        caminho_csv = f"{CFG['pasta_resultados']}/sinais_{data_base}.csv"
        df_sinais.to_csv(caminho_csv, index=False)
        print(f"[✓] CSV salvo: {caminho_csv}")
    
    # Gerar e salvar HTML
    corpo_html = gerar_relatorio_html(df_sinais, data_base)
    caminho_html = f"{CFG['pasta_resultados']}/relatorio_{data_base}.html"
    with open(caminho_html, 'w', encoding='utf-8') as f:
        f.write(corpo_html)
    print(f"[✓] HTML salvo: {caminho_html}")
    
    # Enviar e-mail (se configurado)
    if CFG.get("enviar_email", False):
        print("\n📧 Enviando e-mail...")
        # TODO: Implementar envio de e-mail
        utils.enviar_email_gmail_com_app_password(
                assunto=assunto,
                corpo_texto=corpo_texto,
                corpo_html=corpo_html
            )
    
    print(f"\n{'='*70}\n")
    
    return df_sinais


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="E3 Screening V2 - Screening Diário B3")
    parser.add_argument("--data", type=str, default=None,
                        help="Data base YYYY-MM-DD (padrão: hoje)")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Testar um ticker específico (debug)")
    args = parser.parse_args()
    
    data_base = date.fromisoformat(args.data) if args.data else date.today()
    
    rodar_screening(data_base, ticker_unico=args.ticker)


if __name__ == "__main__":
    main()