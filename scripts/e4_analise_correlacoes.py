"""
Análise de Correlações - Papéis, Câmbio, Índices Globais
Detecta relações, liderança (lag) e oportunidades de convergência
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. BUSCA DE DADOS EXTERNOS
# =============================================================================

def buscar_usd_brl(data_inicio: date, data_fim: date) -> pd.DataFrame:
    """
    Busca cotação USD/BRL (dólar)
    
    Ticker: BRL=X (yahoo finance)
    """
    print("📊 Buscando USD/BRL...", end=" ", flush=True)
    
    try:
        ticker = yf.Ticker("BRL=X")
        df = ticker.history(start=data_inicio, end=data_fim)
        
        if df.empty:
            print("❌ Falhou")
            return pd.DataFrame()
        
        df = df[['Close']].copy()
        df.columns = ['usd_brl']
        df.index = pd.to_datetime(df.index).date
        
        print(f"✓ {len(df)} dias")
        return df
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return pd.DataFrame()


def buscar_eur_usd(data_inicio: date, data_fim: date) -> pd.DataFrame:
    """
    Busca par EUR/USD
    
    Ticker: EURUSD=X
    """
    print("📊 Buscando EUR/USD...", end=" ", flush=True)
    
    try:
        ticker = yf.Ticker("EURUSD=X")
        df = ticker.history(start=data_inicio, end=data_fim)
        
        if df.empty:
            print("❌ Falhou")
            return pd.DataFrame()
        
        df = df[['Close']].copy()
        df.columns = ['eur_usd']
        df.index = pd.to_datetime(df.index).date
        
        print(f"✓ {len(df)} dias")
        return df
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return pd.DataFrame()


def buscar_dxy(data_inicio: date, data_fim: date) -> pd.DataFrame:
    """
    Busca DXY (Dollar Index)
    
    Ticker: DX-Y.NYB
    """
    print("📊 Buscando DXY (Dollar Index)...", end=" ", flush=True)
    
    try:
        ticker = yf.Ticker("DX-Y.NYB")
        df = ticker.history(start=data_inicio, end=data_fim)
        
        if df.empty:
            print("❌ Falhou")
            return pd.DataFrame()
        
        df = df[['Close']].copy()
        df.columns = ['dxy']
        df.index = pd.to_datetime(df.index).date
        
        print(f"✓ {len(df)} dias")
        return df
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return pd.DataFrame()


def buscar_sp500(data_inicio: date, data_fim: date) -> pd.DataFrame:
    """
    Busca S&P 500
    
    Ticker: ^GSPC
    """
    print("📊 Buscando S&P 500...", end=" ", flush=True)
    
    try:
        ticker = yf.Ticker("^GSPC")
        df = ticker.history(start=data_inicio, end=data_fim)
        
        if df.empty:
            print("❌ Falhou")
            return pd.DataFrame()
        
        df = df[['Close']].copy()
        df.columns = ['sp500']
        df.index = pd.to_datetime(df.index).date
        
        print(f"✓ {len(df)} dias")
        return df
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return pd.DataFrame()


def buscar_vix(data_inicio: date, data_fim: date) -> pd.DataFrame:
    """
    Busca VIX (índice de volatilidade/medo)
    
    Ticker: ^VIX
    """
    print("📊 Buscando VIX...", end=" ", flush=True)
    
    try:
        ticker = yf.Ticker("^VIX")
        df = ticker.history(start=data_inicio, end=data_fim)
        
        if df.empty:
            print("❌ Falhou")
            return pd.DataFrame()
        
        df = df[['Close']].copy()
        df.columns = ['vix']
        df.index = pd.to_datetime(df.index).date
        
        print(f"✓ {len(df)} dias")
        return df
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return pd.DataFrame()


def buscar_wti(data_inicio: date, data_fim: date) -> pd.DataFrame:
    """
    Busca WTI (petróleo)
    
    Ticker: CL=F
    """
    print("📊 Buscando WTI (petróleo)...", end=" ", flush=True)
    
    try:
        ticker = yf.Ticker("CL=F")
        df = ticker.history(start=data_inicio, end=data_fim)
        
        if df.empty:
            print("❌ Falhou")
            return pd.DataFrame()
        
        df = df[['Close']].copy()
        df.columns = ['wti']
        df.index = pd.to_datetime(df.index).date
        
        print(f"✓ {len(df)} dias")
        return df
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return pd.DataFrame()


def buscar_ibovespa(data_inicio: date, data_fim: date) -> pd.DataFrame:
    """
    Busca Ibovespa
    
    Ticker: ^BVSP
    """
    print("📊 Buscando Ibovespa...", end=" ", flush=True)
    
    try:
        ticker = yf.Ticker("^BVSP")
        df = ticker.history(start=data_inicio, end=data_fim)
        
        if df.empty:
            print("❌ Falhou")
            return pd.DataFrame()
        
        df = df[['Close']].copy()
        df.columns = ['ibov']
        df.index = pd.to_datetime(df.index).date
        
        print(f"✓ {len(df)} dias")
        return df
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return pd.DataFrame()


def buscar_dados_macro(data_inicio: date = None, data_fim: date = None) -> pd.DataFrame:
    """
    Busca TODOS os dados macro em um único DataFrame
    
    Returns:
        DataFrame com colunas: usd_brl, eur_usd, dxy, sp500, vix, wti, ibov
    """
    if data_inicio is None:
        data_inicio = date.today() - timedelta(days=365*3)  # 3 anos
    
    if data_fim is None:
        data_fim = date.today()
    
    print(f"\n{'='*70}")
    print(f"BUSCANDO DADOS MACRO ({data_inicio} a {data_fim})")
    print(f"{'='*70}\n")
    
    # Buscar cada série
    dfs = []
    
    df_usd = buscar_usd_brl(data_inicio, data_fim)
    if not df_usd.empty:
        dfs.append(df_usd)
    
    df_eur = buscar_eur_usd(data_inicio, data_fim)
    if not df_eur.empty:
        dfs.append(df_eur)
    
    df_dxy = buscar_dxy(data_inicio, data_fim)
    if not df_dxy.empty:
        dfs.append(df_dxy)
    
    df_sp = buscar_sp500(data_inicio, data_fim)
    if not df_sp.empty:
        dfs.append(df_sp)
    
    df_vix = buscar_vix(data_inicio, data_fim)
    if not df_vix.empty:
        dfs.append(df_vix)
    
    df_wti = buscar_wti(data_inicio, data_fim)
    if not df_wti.empty:
        dfs.append(df_wti)
    
    df_ibov = buscar_ibovespa(data_inicio, data_fim)
    if not df_ibov.empty:
        dfs.append(df_ibov)
    
    # Combinar tudo
    if not dfs:
        print("\n❌ Nenhum dado foi carregado!")
        return pd.DataFrame()
    
    df_macro = pd.concat(dfs, axis=1)
    df_macro = df_macro.sort_index()
    
    print(f"\n{'='*70}")
    print(f"✓ Dados macro carregados: {len(df_macro)} dias")
    print(f"  Colunas: {', '.join(df_macro.columns)}")
    print(f"  Período: {df_macro.index.min()} a {df_macro.index.max()}")
    print(f"{'='*70}\n")
    
    return df_macro


# =============================================================================
# 2. ANÁLISE DE CORRELAÇÕES
# =============================================================================

def calcular_correlacao_rolling(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    window: int = 60
) -> pd.Series:
    """
    Calcula correlação rolling entre duas séries
    
    Args:
        df: DataFrame com as séries
        col1: Nome da primeira coluna
        col2: Nome da segunda coluna
        window: Janela de correlação (dias)
    
    Returns:
        Series com correlação rolling
    """
    return df[col1].rolling(window).corr(df[col2])


def calcular_correlacao_com_lag(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    max_lag: int = 5
) -> pd.DataFrame:
    """
    Calcula correlação entre col1 e col2 com diferentes lags
    
    Args:
        df: DataFrame com as séries
        col1: Coluna líder (antecipa)
        col2: Coluna seguidora
        max_lag: Lag máximo a testar (dias)
    
    Returns:
        DataFrame com correlações para cada lag
    """
    resultados = []
    
    for lag in range(0, max_lag + 1):
        # col1 hoje vs col2 daqui a 'lag' dias
        corr = df[col1].corr(df[col2].shift(-lag))
        
        resultados.append({
            'lag': lag,
            'correlacao': corr,
        })
    
    return pd.DataFrame(resultados)


def matriz_correlacao_papeis(
    tickers: List[str],
    data_inicio: date,
    data_fim: date,
    window: int = 252
) -> pd.DataFrame:
    """
    Calcula matriz de correlação entre papéis
    
    Args:
        tickers: Lista de tickers
        data_inicio: Data inicial
        data_fim: Data final
        window: Janela para correlação (dias)
    
    Returns:
        DataFrame com matriz de correlação
    """
    print(f"\n{'='*70}")
    print(f"CALCULANDO CORRELAÇÃO ENTRE PAPÉIS")
    print(f"{'='*70}\n")
    
    # Buscar dados de todos os tickers
    dados = {}
    
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=data_inicio, end=data_fim)
            
            if not df.empty:
                dados[ticker.replace('.SA', '')] = df['Close']
        
        except Exception:
            continue
    
    if not dados:
        print("❌ Nenhum dado carregado!")
        return pd.DataFrame()
    
    # Criar DataFrame com todos os closes
    df_closes = pd.DataFrame(dados)
    
    # Calcular retornos diários
    df_returns = df_closes.pct_change().dropna()
    
    # Matriz de correlação
    corr_matrix = df_returns.corr()
    
    print(f"✓ Matriz de correlação calculada: {len(corr_matrix)} papéis")
    
    return corr_matrix


def detectar_descorrelacoes(
    df: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    correlacao_esperada: float,
    threshold_std: float = 2.0
) -> pd.DataFrame:
    """
    Detecta momentos de descorrelação anormal entre dois ativos
    
    Args:
        df: DataFrame com preços dos dois ativos
        ticker1: Primeiro ticker
        ticker2: Segundo ticker
        correlacao_esperada: Correlação histórica esperada
        threshold_std: Quantos desvios padrão = anormal
    
    Returns:
        DataFrame com oportunidades detectadas
    """
    # Calcular retornos
    ret1 = df[ticker1].pct_change()
    ret2 = df[ticker2].pct_change()
    
    # Spread esperado vs real
    spread = ret1 - (correlacao_esperada * ret2)
    
    # Desvio padrão do spread
    spread_std = spread.std()
    
    # Z-score do spread
    z_score = spread / spread_std
    
    # Detectar oportunidades (|z| > threshold)
    oportunidades = pd.DataFrame({
        'data': df.index,
        'ret1': ret1,
        'ret2': ret2,
        'spread': spread,
        'z_score': z_score,
        'oportunidade': abs(z_score) > threshold_std,
    })
    
    return oportunidades[oportunidades['oportunidade']]


# =============================================================================
# 3. VISUALIZAÇÕES
# =============================================================================

def plotar_heatmap_correlacao(corr_matrix: pd.DataFrame, titulo: str = "Correlações"):
    """
    Plota heatmap de matriz de correlação
    """
    plt.figure(figsize=(14, 12))
    
    # Heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlação'}
    )
    
    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return plt.gcf()


def plotar_correlacao_temporal(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    window: int = 60
):
    """
    Plota evolução temporal da correlação entre duas séries
    """
    corr_rolling = calcular_correlacao_rolling(df, col1, col2, window)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Série 1
    axes[0].plot(df.index, df[col1], label=col1, linewidth=1)
    axes[0].set_ylabel(col1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Correlação rolling
    axes[1].plot(corr_rolling.index, corr_rolling, label=f'Correlação {window}d', color='purple', linewidth=2)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Correlação')
    axes[1].set_xlabel('Data')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'Correlação Temporal: {col1} vs {col2}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# =============================================================================
# 4. FUNÇÃO PRINCIPAL DE ANÁLISE
# =============================================================================

def analisar_correlacoes_completo(
    tickers: List[str],
    data_inicio: date = None,
    data_fim: date = None,
    pasta_saida: str = "resultados_correlacoes"
):
    """
    Análise completa de correlações:
    - Entre papéis
    - Papéis vs câmbio
    - Papéis vs índices globais
    - Correlações com lag
    - Detecção de oportunidades
    
    Args:
        tickers: Lista de tickers BR
        data_inicio: Data inicial
        data_fim: Data final
        pasta_saida: Pasta para salvar resultados
    """
    if data_inicio is None:
        data_inicio = date.today() - timedelta(days=365*3)
    
    if data_fim is None:
        data_fim = date.today()
    
    Path(pasta_saida).mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ANÁLISE COMPLETA DE CORRELAÇÕES")
    print(f"Período: {data_inicio} a {data_fim}")
    print(f"{'='*70}\n")
    
    # 1. Buscar dados macro
    df_macro = buscar_dados_macro(data_inicio, data_fim)
    
    if df_macro.empty:
        print("❌ Não foi possível buscar dados macro!")
        return
    
    # 2. Matriz de correlação entre papéis
    corr_papeis = matriz_correlacao_papeis(
        [f"{t}.SA" for t in tickers],
        data_inicio,
        data_fim
    )
    
    if not corr_papeis.empty:
        # Salvar
        corr_papeis.to_csv(f"{pasta_saida}/correlacao_papeis.csv")
        
        # Plotar
        fig = plotar_heatmap_correlacao(corr_papeis, "Correlação entre Papéis (Retornos Diários)")
        fig.savefig(f"{pasta_saida}/heatmap_papeis.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Matriz de correlação entre papéis salva em {pasta_saida}/")
    
    # 3. Top pares correlacionados
    if not corr_papeis.empty:
        print(f"\n{'='*70}")
        print("TOP 10 PARES MAIS CORRELACIONADOS")
        print(f"{'='*70}\n")
        
        # Extrair pares únicos
        pares = []
        for i in range(len(corr_papeis)):
            for j in range(i+1, len(corr_papeis)):
                ticker1 = corr_papeis.index[i]
                ticker2 = corr_papeis.columns[j]
                corr = corr_papeis.iloc[i, j]
                
                pares.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'correlacao': corr
                })
        
        df_pares = pd.DataFrame(pares).sort_values('correlacao', ascending=False).head(10)
        
        print(df_pares.to_string(index=False))
        df_pares.to_csv(f"{pasta_saida}/top_pares_correlacionados.csv", index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Análise completa! Resultados em: {pasta_saida}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Teste rápido
    tickers = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'WEGE3']
    analisar_correlacoes_completo(tickers)