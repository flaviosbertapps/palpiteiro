#!/usr/bin/env python3
"""
CLI para Análise de Correlações
Uso: python e4_analise_correlacoes_cli.py [opção]
"""

import argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from e4_analise_correlacoes import *

# Universo líquido (27 papéis)
UNIVERSO_LIQUIDO = [
    'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3', 'B3SA3', 'WEGE3', 'RENT3',
    'ELET3', 'MGLU3', 'VIIA3', 'SUZB3', 'JBSS3', 'SANB11', 'EGIE3', 'PRIO3',
    'UGPA3', 'LREN3', 'CSNA3', 'RADL3', 'RAIL3', 'ENEV3', 'CSAN3', 'BEEF3',
    'GGBR4', 'CPFE3', 'BRFS3'
]


def analisar_correlacao_papel_vs_macro(ticker: str, data_inicio: date, data_fim: date):
    """
    Analisa correlação de UM papel vs todas as variáveis macro
    """
    print(f"\n{'='*70}")
    print(f"CORRELAÇÃO: {ticker} vs VARIÁVEIS MACRO")
    print(f"{'='*70}\n")
    
    # Buscar dados do papel
    print(f"📊 Buscando dados de {ticker}...", end=" ", flush=True)
    try:
        t = yf.Ticker(f"{ticker}.SA")
        df_papel = t.history(start=data_inicio, end=data_fim)
        df_papel = df_papel[['Close']].copy()
        df_papel.columns = [ticker]
        df_papel.index = pd.to_datetime(df_papel.index).date
        print(f"✓ {len(df_papel)} dias")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    # Buscar dados macro
    df_macro = buscar_dados_macro(data_inicio, data_fim)
    
    if df_macro.empty:
        print("❌ Dados macro não disponíveis")
        return
    
    # Combinar
    df_combined = pd.concat([df_papel, df_macro], axis=1).dropna()
    
    if df_combined.empty:
        print("❌ Sem sobreposição de datas")
        return
    
    # Calcular retornos diários
    df_returns = df_combined.pct_change().dropna()
    
    # Correlações sem lag
    print(f"\n{'='*70}")
    print("CORRELAÇÕES (sem lag)")
    print(f"{'='*70}\n")
    
    correlacoes = []
    for col in df_macro.columns:
        if col in df_returns.columns:
            corr = df_returns[ticker].corr(df_returns[col])
            correlacoes.append({
                'variavel': col,
                'correlacao': corr,
                'lag': 0
            })
            print(f"{col:15s}: {corr:+.3f}")
    
    # Correlações COM lag (macro antecipa papel)
    print(f"\n{'='*70}")
    print("CORRELAÇÕES COM LAG (macro antecipa papel)")
    print(f"{'='*70}\n")
    
    for col in df_macro.columns:
        if col in df_returns.columns:
            print(f"\n{col}:")
            
            for lag in range(1, 6):
                # Macro hoje vs papel daqui a 'lag' dias
                corr = df_returns[col].corr(df_returns[ticker].shift(-lag))
                correlacoes.append({
                    'variavel': col,
                    'correlacao': corr,
                    'lag': lag
                })
                print(f"  Lag {lag}d: {corr:+.3f}")
    
    # Salvar
    df_corr = pd.DataFrame(correlacoes)
    pasta = f"resultados_correlacoes/{ticker}"
    Path(pasta).mkdir(parents=True, exist_ok=True)
    
    df_corr.to_csv(f"{pasta}/correlacoes_macro.csv", index=False)
    
    # Top correlações
    print(f"\n{'='*70}")
    print("TOP 5 CORRELAÇÕES (com lag)")
    print(f"{'='*70}\n")
    
    top = df_corr.nlargest(5, 'correlacao', keep='all')
    print(top.to_string(index=False))
    
    print(f"\n✓ Resultados salvos em: {pasta}/")


def analisar_lideranca_sp500_ibov(data_inicio: date, data_fim: date):
    """
    Testa se S&P 500 antecipa movimentos do Ibovespa
    """
    print(f"\n{'='*70}")
    print("LIDERANÇA: S&P 500 → IBOVESPA")
    print(f"{'='*70}\n")
    
    # Buscar dados
    df_sp = buscar_sp500(data_inicio, data_fim)
    df_ibov = buscar_ibovespa(data_inicio, data_fim)
    
    if df_sp.empty or df_ibov.empty:
        print("❌ Dados não disponíveis")
        return
    
    # Combinar
    df = pd.concat([df_sp, df_ibov], axis=1).dropna()
    
    # Retornos
    df_ret = df.pct_change().dropna()
    
    # Testar lags
    print("Correlação S&P 500 hoje vs Ibov em t+N:\n")
    
    resultados = []
    for lag in range(0, 6):
        corr = df_ret['sp500'].corr(df_ret['ibov'].shift(-lag))
        resultados.append({
            'lag': lag,
            'correlacao': corr
        })
        
        dias_str = "hoje" if lag == 0 else f"daqui a {lag}d"
        print(f"  Lag {lag}d (Ibov {dias_str}): {corr:+.3f}")
    
    # Melhor lag
    df_res = pd.DataFrame(resultados)
    melhor = df_res.loc[df_res['correlacao'].abs().idxmax()]
    
    print(f"\n🏆 Melhor correlação:")
    print(f"   Lag {melhor['lag']:.0f} dias: {melhor['correlacao']:+.3f}")
    
    if melhor['lag'] > 0 and melhor['correlacao'] > 0.5:
        print(f"\n💡 HIPÓTESE VALIDADA!")
        print(f"   S&P 500 ANTECIPA Ibov em {melhor['lag']:.0f} dia(s)")
        print(f"   Correlação: {melhor['correlacao']:.1%}")
        print(f"\n   Estratégia possível:")
        print(f"   → S&P cai forte hoje → Short Ibov amanhã")
    
    # Salvar
    pasta = "resultados_correlacoes/lideranca"
    Path(pasta).mkdir(parents=True, exist_ok=True)
    df_res.to_csv(f"{pasta}/sp500_ibov_lag.csv", index=False)
    
    print(f"\n✓ Resultados salvos em: {pasta}/")


def analisar_todos_papeis_vs_macro(data_inicio: date, data_fim: date):
    """
    Análise BATCH: Todos os 27 papéis vs variáveis macro
    Gera ranking consolidado e priorizado
    """
    print(f"\n{'='*70}")
    print(f"ANÁLISE BATCH: 27 PAPÉIS vs VARIÁVEIS MACRO")
    print(f"{'='*70}\n")
    
    # Buscar dados macro UMA VEZ
    df_macro = buscar_dados_macro(data_inicio, data_fim)
    
    if df_macro.empty:
        print("❌ Dados macro não disponíveis")
        return
    
    # Preparar estrutura de resultados
    todos_resultados = []
    
    print(f"\n{'='*70}")
    print(f"PROCESSANDO {len(UNIVERSO_LIQUIDO)} PAPÉIS...")
    print(f"{'='*70}\n")
    
    # Para cada papel
    for i, ticker in enumerate(UNIVERSO_LIQUIDO, 1):
        print(f"[{i}/{len(UNIVERSO_LIQUIDO)}] {ticker}...", end=" ", flush=True)
        
        try:
            # Buscar dados do papel
            t = yf.Ticker(f"{ticker}.SA")
            df_papel = t.history(start=data_inicio, end=data_fim)
            
            if df_papel.empty or len(df_papel) < 30:
                print("❌ Sem dados")
                continue
            
            df_papel = df_papel[['Close']].copy()
            df_papel.columns = [ticker]
            df_papel.index = pd.to_datetime(df_papel.index).date
            
            # Combinar com macro
            df_combined = pd.concat([df_papel, df_macro], axis=1).dropna()
            
            if len(df_combined) < 30:
                print("❌ Pouca sobreposição")
                continue
            
            # Retornos diários
            df_returns = df_combined.pct_change().dropna()
            
            # Calcular correlações SEM lag
            for col in df_macro.columns:
                if col in df_returns.columns:
                    corr = df_returns[ticker].corr(df_returns[col])
                    
                    todos_resultados.append({
                        'ticker': ticker,
                        'variavel': col,
                        'correlacao': corr,
                        'lag': 0,
                        'abs_corr': abs(corr)
                    })
            
            # Calcular correlações COM lag (macro antecipa papel)
            for col in df_macro.columns:
                if col in df_returns.columns:
                    for lag in range(1, 4):  # Lag 1, 2, 3 dias
                        corr = df_returns[col].corr(df_returns[ticker].shift(-lag))
                        
                        todos_resultados.append({
                            'ticker': ticker,
                            'variavel': col,
                            'correlacao': corr,
                            'lag': lag,
                            'abs_corr': abs(corr)
                        })
            
            print("✓")
        
        except Exception as e:
            print(f"❌ Erro: {e}")
            continue
    
    if not todos_resultados:
        print("\n❌ Nenhum resultado obtido!")
        return
    
    # Criar DataFrame consolidado
    df_resultados = pd.DataFrame(todos_resultados)
    
    # Salvar TUDO
    pasta = "resultados_correlacoes/batch"
    Path(pasta).mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(f"{pasta}/todos_papeis_vs_macro_completo.csv", index=False)
    
    # ========================================================================
    # ANÁLISE 1: TOP CORRELAÇÕES ABSOLUTAS (sem lag)
    # ========================================================================
    print(f"\n{'='*70}")
    print("📊 TOP 20 CORRELAÇÕES MAIS FORTES (sem lag)")
    print(f"{'='*70}\n")
    
    sem_lag = df_resultados[df_resultados['lag'] == 0].copy()
    top_sem_lag = sem_lag.nlargest(20, 'abs_corr')
    
    print(top_sem_lag[['ticker', 'variavel', 'correlacao']].to_string(index=False))
    top_sem_lag.to_csv(f"{pasta}/top_correlacoes_sem_lag.csv", index=False)
    
    # ========================================================================
    # ANÁLISE 2: TOP CORRELAÇÕES COM LAG (antecipação)
    # ========================================================================
    print(f"\n{'='*70}")
    print("🔮 TOP 20 CORRELAÇÕES COM LAG (antecipação)")
    print(f"{'='*70}\n")
    
    com_lag = df_resultados[df_resultados['lag'] > 0].copy()
    top_com_lag = com_lag.nlargest(20, 'abs_corr')
    
    print(top_com_lag[['ticker', 'variavel', 'correlacao', 'lag']].to_string(index=False))
    top_com_lag.to_csv(f"{pasta}/top_correlacoes_com_lag.csv", index=False)
    
    # ========================================================================
    # ANÁLISE 3: EXPORTADORES vs IMPORTADORES (USD/BRL)
    # ========================================================================
    print(f"\n{'='*70}")
    print("💵 SENSIBILIDADE AO DÓLAR (USD/BRL)")
    print(f"{'='*70}\n")
    
    usd_corr = df_resultados[
        (df_resultados['variavel'] == 'usd_brl') & 
        (df_resultados['lag'] == 0)
    ].sort_values('correlacao', ascending=False)
    
    print("🟢 EXPORTADORES (correlação positiva):\n")
    exportadores = usd_corr[usd_corr['correlacao'] > 0.3].head(10)
    print(exportadores[['ticker', 'correlacao']].to_string(index=False))
    
    print("\n🔴 IMPORTADORES (correlação negativa):\n")
    importadores = usd_corr[usd_corr['correlacao'] < -0.3].head(10)
    print(importadores[['ticker', 'correlacao']].to_string(index=False))
    
    usd_corr.to_csv(f"{pasta}/sensibilidade_dolar.csv", index=False)
    
    # ========================================================================
    # ANÁLISE 4: PETRÓLEO (WTI)
    # ========================================================================
    print(f"\n{'='*70}")
    print("🛢️  SENSIBILIDADE AO PETRÓLEO (WTI)")
    print(f"{'='*70}\n")
    
    wti_corr = df_resultados[
        (df_resultados['variavel'] == 'wti') & 
        (df_resultados['lag'] == 0)
    ].sort_values('correlacao', ascending=False).head(10)
    
    print(wti_corr[['ticker', 'correlacao']].to_string(index=False))
    wti_corr.to_csv(f"{pasta}/sensibilidade_petroleo.csv", index=False)
    
    # ========================================================================
    # ANÁLISE 5: S&P 500 (risco global)
    # ========================================================================
    print(f"\n{'='*70}")
    print("🌍 SENSIBILIDADE AO S&P 500 (risco global)")
    print(f"{'='*70}\n")
    
    sp_corr = df_resultados[
        (df_resultados['variavel'] == 'sp500') & 
        (df_resultados['lag'] == 0)
    ].sort_values('correlacao', ascending=False).head(10)
    
    print(sp_corr[['ticker', 'correlacao']].to_string(index=False))
    sp_corr.to_csv(f"{pasta}/sensibilidade_sp500.csv", index=False)
    
    # ========================================================================
    # ANÁLISE 6: VIX (medo)
    # ========================================================================
    print(f"\n{'='*70}")
    print("😱 SENSIBILIDADE AO VIX (medo)")
    print(f"{'='*70}\n")
    
    vix_corr = df_resultados[
        (df_resultados['variavel'] == 'vix') & 
        (df_resultados['lag'] == 0)
    ].sort_values('correlacao', ascending=True).head(10)  # Mais negativo = mais sensível ao medo
    
    print(vix_corr[['ticker', 'correlacao']].to_string(index=False))
    vix_corr.to_csv(f"{pasta}/sensibilidade_vix.csv", index=False)
    
    # ========================================================================
    # ANÁLISE 7: RANKING DE ANTECIPAÇÃO (melhor lag por papel)
    # ========================================================================
    print(f"\n{'='*70}")
    print("⏰ RANKING: MELHORES SINAIS DE ANTECIPAÇÃO (com lag)")
    print(f"{'='*70}\n")
    
    # Para cada papel, pegar a MELHOR correlação com lag
    melhores_lag = com_lag.loc[com_lag.groupby('ticker')['abs_corr'].idxmax()]
    melhores_lag = melhores_lag.sort_values('abs_corr', ascending=False).head(15)
    
    print(melhores_lag[['ticker', 'variavel', 'correlacao', 'lag']].to_string(index=False))
    melhores_lag.to_csv(f"{pasta}/melhores_sinais_antecipacao.csv", index=False)
    
    # ========================================================================
    # ANÁLISE 8: RESUMO EXECUTIVO POR PAPEL
    # ========================================================================
    print(f"\n{'='*70}")
    print("📋 GERANDO RESUMO EXECUTIVO POR PAPEL...")
    print(f"{'='*70}\n")
    
    resumo_papeis = []
    
    for ticker in UNIVERSO_LIQUIDO:
        dados_ticker = df_resultados[df_resultados['ticker'] == ticker]
        
        if dados_ticker.empty:
            continue
        
        # Sem lag
        sem_lag_ticker = dados_ticker[dados_ticker['lag'] == 0]
        melhor_sem_lag = sem_lag_ticker.loc[sem_lag_ticker['abs_corr'].idxmax()] if not sem_lag_ticker.empty else None
        
        # Com lag
        com_lag_ticker = dados_ticker[dados_ticker['lag'] > 0]
        melhor_com_lag = com_lag_ticker.loc[com_lag_ticker['abs_corr'].idxmax()] if not com_lag_ticker.empty else None
        
        # Pegar correlações específicas
        usd = dados_ticker[(dados_ticker['variavel'] == 'usd_brl') & (dados_ticker['lag'] == 0)]['correlacao'].values
        wti = dados_ticker[(dados_ticker['variavel'] == 'wti') & (dados_ticker['lag'] == 0)]['correlacao'].values
        sp = dados_ticker[(dados_ticker['variavel'] == 'sp500') & (dados_ticker['lag'] == 0)]['correlacao'].values
        vix = dados_ticker[(dados_ticker['variavel'] == 'vix') & (dados_ticker['lag'] == 0)]['correlacao'].values
        
        resumo_papeis.append({
            'ticker': ticker,
            'melhor_corr_sem_lag': melhor_sem_lag['correlacao'] if melhor_sem_lag is not None else np.nan,
            'melhor_var_sem_lag': melhor_sem_lag['variavel'] if melhor_sem_lag is not None else '',
            'melhor_corr_com_lag': melhor_com_lag['correlacao'] if melhor_com_lag is not None else np.nan,
            'melhor_var_com_lag': melhor_com_lag['variavel'] if melhor_com_lag is not None else '',
            'melhor_lag_dias': melhor_com_lag['lag'] if melhor_com_lag is not None else np.nan,
            'corr_usd_brl': usd[0] if len(usd) > 0 else np.nan,
            'corr_wti': wti[0] if len(wti) > 0 else np.nan,
            'corr_sp500': sp[0] if len(sp) > 0 else np.nan,
            'corr_vix': vix[0] if len(vix) > 0 else np.nan,
        })
    
    df_resumo = pd.DataFrame(resumo_papeis)
    df_resumo.to_csv(f"{pasta}/resumo_executivo_por_papel.csv", index=False)
    
    print("✓ Resumo executivo salvo!\n")
    
    # ========================================================================
    # RELATÓRIO FINAL
    # ========================================================================
    print(f"\n{'='*70}")
    print("✅ ANÁLISE BATCH CONCLUÍDA!")
    print(f"{'='*70}\n")
    
    print(f"📊 Resultados processados:")
    print(f"   • {len(df_resultados)} correlações calculadas")
    print(f"   • {df_resultados['ticker'].nunique()} papéis analisados")
    print(f"   • {df_resultados['variavel'].nunique()} variáveis macro")
    print(f"\n📁 Arquivos gerados em: {pasta}/\n")
    
    print("📄 Arquivos disponíveis:")
    print("   1. todos_papeis_vs_macro_completo.csv - Dataset completo")
    print("   2. top_correlacoes_sem_lag.csv - Top 20 sem lag")
    print("   3. top_correlacoes_com_lag.csv - Top 20 com antecipação")
    print("   4. sensibilidade_dolar.csv - Exportadores vs Importadores")
    print("   5. sensibilidade_petroleo.csv - Correlação com WTI")
    print("   6. sensibilidade_sp500.csv - Correlação com S&P")
    print("   7. sensibilidade_vix.csv - Sensibilidade ao medo")
    print("   8. melhores_sinais_antecipacao.csv - Melhores lags por papel")
    print("   9. resumo_executivo_por_papel.csv - Visão consolidada\n")
    
    print(f"{'='*70}\n")


def detectar_oportunidades_pairs(data_inicio: date, data_fim: date):
    """
    Detecta oportunidades de pairs trading
    Pares com alta correlação que estão descorrelacionados
    """
    print(f"\n{'='*70}")
    print("DETECÇÃO DE OPORTUNIDADES - PAIRS TRADING")
    print(f"{'='*70}\n")
    
    # Calcular matriz de correlação
    corr_matrix = matriz_correlacao_papeis(
        [f"{t}.SA" for t in UNIVERSO_LIQUIDO],
        data_inicio,
        data_fim
    )
    
    if corr_matrix.empty:
        print("❌ Não foi possível calcular correlações")
        return
    
    # Buscar pares com correlação > 0.70
    print("Buscando pares altamente correlacionados (> 0.70)...\n")
    
    pares_fortes = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            ticker1 = corr_matrix.index[i]
            ticker2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            
            if corr > 0.70:
                pares_fortes.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'correlacao': corr
                })
    
    if not pares_fortes:
        print("❌ Nenhum par com correlação > 0.70 encontrado")
        return
    
    print(f"✓ Encontrados {len(pares_fortes)} pares\n")
    
    # Para cada par, verificar descorrelações recentes
    print(f"{'='*70}")
    print("DESCORRELAÇÕES RECENTES (últimos 5 dias)")
    print(f"{'='*70}\n")
    
    oportunidades = []
    
    for par in pares_fortes[:10]:  # Top 10 pares
        ticker1 = par['ticker1']
        ticker2 = par['ticker2']
        corr_hist = par['correlacao']
        
        # Buscar dados recentes
        try:
            df1 = yf.Ticker(f"{ticker1}.SA").history(period="1mo")['Close']
            df2 = yf.Ticker(f"{ticker2}.SA").history(period="1mo")['Close']
            
            df = pd.DataFrame({ticker1: df1, ticker2: df2}).dropna()
            
            if len(df) < 5:
                continue
            
            # Retornos últimos 5 dias
            ret1_5d = df[ticker1].pct_change(5).iloc[-1]
            ret2_5d = df[ticker2].pct_change(5).iloc[-1]
            
            # Spread esperado vs real
            spread_esperado = ret1_5d * corr_hist
            spread_real = ret2_5d
            spread_diff = spread_real - spread_esperado
            
            # Z-score (simplificado)
            z = spread_diff / 0.02  # Assumindo std = 2%
            
            if abs(z) > 1.5:  # Descorrelação significativa
                oportunidades.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'correlacao_hist': corr_hist,
                    'ret1_5d': ret1_5d,
                    'ret2_5d': ret2_5d,
                    'spread_diff': spread_diff,
                    'z_score': z,
                    'acao': 'Long ' + ticker1 + ' / Short ' + ticker2 if z < 0 else 'Short ' + ticker1 + ' / Long ' + ticker2
                })
        
        except Exception:
            continue
    
    if oportunidades:
        df_opor = pd.DataFrame(oportunidades).sort_values('z_score', key=abs, ascending=False)
        
        print(df_opor[['ticker1', 'ticker2', 'ret1_5d', 'ret2_5d', 'z_score', 'acao']].to_string(index=False))
        
        # Salvar
        pasta = "resultados_correlacoes/oportunidades"
        Path(pasta).mkdir(parents=True, exist_ok=True)
        df_opor.to_csv(f"{pasta}/pairs_descorrelacionados.csv", index=False)
        
        print(f"\n✓ {len(df_opor)} oportunidades detectadas!")
        print(f"✓ Resultados salvos em: {pasta}/")
    else:
        print("❌ Nenhuma oportunidade detectada no momento")


def analisar_lideranca_entre_papeis(data_inicio: date, data_fim: date):
    """
    Analisa se algum papel ANTECIPA movimentos de outros (lag analysis)
    
    Busca relações do tipo:
    - PETR4 hoje → VALE3 amanhã
    - ITUB4 hoje → BBDC4 amanhã
    - WEGE3 hoje → SUZB3 amanhã
    """
    print(f"\n{'='*70}")
    print("🔮 LIDERANÇA ENTRE PAPÉIS (Análise com LAG)")
    print(f"{'='*70}\n")
    
    # Buscar dados de todos os papéis
    print(f"📊 Buscando dados de {len(UNIVERSO_LIQUIDO)} papéis...\n")
    
    dados = {}
    for i, ticker in enumerate(UNIVERSO_LIQUIDO, 1):
        print(f"[{i}/{len(UNIVERSO_LIQUIDO)}] {ticker}...", end=" ", flush=True)
        
        try:
            t = yf.Ticker(f"{ticker}.SA")
            df = t.history(start=data_inicio, end=data_fim)
            
            if not df.empty and len(df) >= 60:
                df.index = pd.to_datetime(df.index).date
                dados[ticker] = df['Close']
                print("✓")
            else:
                print("❌ Poucos dados")
        
        except Exception as e:
            print(f"❌ Erro: {e}")
    
    if len(dados) < 5:
        print("\n❌ Poucos papéis carregados!")
        return
    
    # Criar DataFrame com todos os closes
    df_closes = pd.DataFrame(dados)
    
    # Calcular retornos diários
    df_returns = df_closes.pct_change().dropna()
    
    print(f"\n✓ {len(df_returns)} dias de dados carregados")
    print(f"✓ {len(df_returns.columns)} papéis processados\n")
    
    # Calcular correlações com LAG
    print(f"{'='*70}")
    print("🔍 CALCULANDO CORRELAÇÕES COM LAG (1, 2, 3 dias)...")
    print(f"{'='*70}\n")
    
    resultados = []
    
    for ticker_lider in df_returns.columns:
        for ticker_seguidor in df_returns.columns:
            if ticker_lider == ticker_seguidor:
                continue
            
            # Testar lags 1, 2, 3 dias
            for lag in [1, 2, 3]:
                # ticker_lider HOJE vs ticker_seguidor daqui a 'lag' dias
                corr = df_returns[ticker_lider].corr(df_returns[ticker_seguidor].shift(-lag))
                
                if not pd.isna(corr):
                    resultados.append({
                        'lider': ticker_lider,
                        'seguidor': ticker_seguidor,
                        'lag': lag,
                        'correlacao': corr,
                        'abs_corr': abs(corr)
                    })
    
    df_lag = pd.DataFrame(resultados)
    
    # Salvar
    pasta = "resultados_correlacoes/lideranca_papeis"
    Path(pasta).mkdir(parents=True, exist_ok=True)
    df_lag.to_csv(f"{pasta}/correlacoes_com_lag_completo.csv", index=False)
    
    # ========================================================================
    # ANÁLISE 1: TOP LIDERANÇAS (correlação > 0.40)
    # ========================================================================
    print(f"\n{'='*70}")
    print("🏆 TOP 20 RELAÇÕES DE LIDERANÇA (correlação > 0.40)")
    print(f"{'='*70}\n")
    
    top_lideranca = df_lag[df_lag['abs_corr'] > 0.40].nlargest(20, 'abs_corr')
    
    if not top_lideranca.empty:
        print(top_lideranca[['lider', 'seguidor', 'correlacao', 'lag']].to_string(index=False))
        top_lideranca.to_csv(f"{pasta}/top_liderancas.csv", index=False)
    else:
        print("❌ Nenhuma liderança forte encontrada (todas < 0.40)")
    
    # ========================================================================
    # ANÁLISE 2: SETORES (clusters que movem juntos com lag)
    # ========================================================================
    print(f"\n{'='*70}")
    print("🏦 BANCOS: Quem lidera quem?")
    print(f"{'='*70}\n")
    
    bancos = ['ITUB4', 'BBDC4', 'BBAS3', 'SANB11']
    df_bancos = df_lag[
        (df_lag['lider'].isin(bancos)) &
        (df_lag['seguidor'].isin(bancos))
    ].nlargest(10, 'abs_corr')
    
    if not df_bancos.empty:
        print(df_bancos[['lider', 'seguidor', 'correlacao', 'lag']].to_string(index=False))
    else:
        print("❌ Sem correlações significativas")
    
    # ========================================================================
    # ANÁLISE 3: COMMODITIES
    # ========================================================================
    print(f"\n{'='*70}")
    print("⛏️  COMMODITIES/SIDERÚRGICAS: Quem lidera quem?")
    print(f"{'='*70}\n")
    
    commodities = ['PETR4', 'VALE3', 'CSNA3', 'GGBR4', 'PRIO3']
    df_commodities = df_lag[
        (df_lag['lider'].isin(commodities)) &
        (df_lag['seguidor'].isin(commodities))
    ].nlargest(10, 'abs_corr')
    
    if not df_commodities.empty:
        print(df_commodities[['lider', 'seguidor', 'correlacao', 'lag']].to_string(index=False))
    else:
        print("❌ Sem correlações significativas")
    
    # ========================================================================
    # ANÁLISE 4: VAREJO
    # ========================================================================
    print(f"\n{'='*70}")
    print("🛒 VAREJO: Quem lidera quem?")
    print(f"{'='*70}\n")
    
    varejo = ['LREN3', 'MGLU3', 'VIIA3', 'RENT3']
    df_varejo = df_lag[
        (df_lag['lider'].isin(varejo)) &
        (df_lag['seguidor'].isin(varejo))
    ].nlargest(10, 'abs_corr')
    
    if not df_varejo.empty:
        print(df_varejo[['lider', 'seguidor', 'correlacao', 'lag']].to_string(index=False))
    else:
        print("❌ Sem correlações significativas")
    
    # ========================================================================
    # ANÁLISE 5: MELHORES SINAIS POR PAPEL SEGUIDOR
    # ========================================================================
    print(f"\n{'='*70}")
    print("📊 MELHOR SINAL DE ANTECIPAÇÃO POR PAPEL")
    print(f"{'='*70}\n")
    
    melhores_sinais = []
    
    for seguidor in df_returns.columns:
        subset = df_lag[df_lag['seguidor'] == seguidor]
        
        if not subset.empty:
            melhor = subset.loc[subset['abs_corr'].idxmax()]
            
            if melhor['abs_corr'] > 0.30:  # Threshold mínimo
                melhores_sinais.append({
                    'papel': seguidor,
                    'antecipado_por': melhor['lider'],
                    'correlacao': melhor['correlacao'],
                    'lag': melhor['lag']
                })
    
    if melhores_sinais:
        df_melhores = pd.DataFrame(melhores_sinais).sort_values('correlacao', key=abs, ascending=False)
        print(df_melhores.to_string(index=False))
        df_melhores.to_csv(f"{pasta}/melhores_sinais_por_papel.csv", index=False)
    else:
        print("❌ Nenhum sinal forte encontrado")
    
    # ========================================================================
    # RELATÓRIO FINAL
    # ========================================================================
    print(f"\n{'='*70}")
    print("✅ ANÁLISE DE LIDERANÇA CONCLUÍDA!")
    print(f"{'='*70}\n")
    
    print(f"📊 Estatísticas:")
    print(f"   • {len(df_lag)} correlações calculadas")
    print(f"   • {len(df_lag[df_lag['abs_corr'] > 0.40])} lideranças fortes (>0.40)")
    print(f"   • {len(df_lag[df_lag['abs_corr'] > 0.50])} lideranças muito fortes (>0.50)")
    
    print(f"\n📁 Arquivos salvos em: {pasta}/")
    print(f"   1. correlacoes_com_lag_completo.csv - Dataset completo")
    print(f"   2. top_liderancas.csv - Top 20 relações")
    print(f"   3. melhores_sinais_por_papel.csv - Melhor sinal por papel\n")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Análise de Correlações - Papéis, Câmbio, Índices"
    )
    
    parser.add_argument(
        "--analise-completa",
        action="store_true",
        help="Análise completa: matriz de correlações entre todos os papéis"
    )
    
    parser.add_argument(
        "--batch-todos-vs-macro",
        action="store_true",
        help="🔥 BATCH: Analisa TODOS os 27 papéis vs macro (consolidado e priorizado)"
    )
    
    parser.add_argument(
        "--papel-vs-macro",
        type=str,
        help="Analisa correlação de UM papel vs variáveis macro (ex: PETR4)"
    )
    
    parser.add_argument(
        "--lideranca-sp500",
        action="store_true",
        help="Testa se S&P 500 antecipa Ibovespa (lag analysis)"
    )
    
    parser.add_argument(
        "--lideranca-papeis",
        action="store_true",
        help="🔥 Analisa se papéis antecipam outros (ex: PETR4 hoje → VALE3 amanhã)"
    )
    
    parser.add_argument(
        "--detectar-pairs",
        action="store_true",
        help="Detecta oportunidades de pairs trading (descorrelações)"
    )
    
    parser.add_argument(
        "--dias",
        type=int,
        default=365*2,
        help="Dias de histórico (default: 730 = 2 anos)"
    )
    
    args = parser.parse_args()
    
    # Datas
    data_fim = date.today()
    data_inicio = data_fim - timedelta(days=args.dias)
    
    # Executar análises
    if args.analise_completa:
        analisar_correlacoes_completo(UNIVERSO_LIQUIDO, data_inicio, data_fim)
    
    elif args.batch_todos_vs_macro:
        analisar_todos_papeis_vs_macro(data_inicio, data_fim)
    
    elif args.papel_vs_macro:
        analisar_correlacao_papel_vs_macro(args.papel_vs_macro.upper(), data_inicio, data_fim)
    
    elif args.lideranca_sp500:
        analisar_lideranca_sp500_ibov(data_inicio, data_fim)
    
    elif args.lideranca_papeis:
        analisar_lideranca_entre_papeis(data_inicio, data_fim)
    
    elif args.detectar_pairs:
        detectar_oportunidades_pairs(data_inicio, data_fim)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()