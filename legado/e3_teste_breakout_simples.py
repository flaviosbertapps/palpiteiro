"""
Diagnóstico de Breakout - Estrutura Correta

Loop por PAPEL primeiro (mais eficiente):
1. Baixa dados do papel
2. Testa todas estratégias
3. Separa trades por ano
"""

import sys
sys.path.insert(0, '/mnt/user-data/uploads')

import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

# Importar do arquivo principal
exec(open('e3_screening_regras.py').read())


def diagnosticar_breakout_simples(data_base: date = None):
    """
    Estrutura SIMPLES e CORRETA:
    - Loop por papel (baixa 1×)
    - Para cada papel: testa todas estratégias
    - Separa trades por ano no final
    """
    if data_base is None:
        data_base = date.today()
    
    tickers = carregar_universo_completo()
    ano_atual = data_base.year
    anos = list(range(ano_atual - 4, ano_atual + 1))
    
    estrategias = {
        "1. BÁSICO": ESTRATEGIA_BREAKOUT_BASICO,
        "2. + VOLUME": ESTRATEGIA_BREAKOUT_VOLUME,
        "3. + VOLUME + ADX": ESTRATEGIA_BREAKOUT_VOLUME_ADX,
        "4. + VOLUME + ADX + MM50": ESTRATEGIA_BREAKOUT_COMPLETO_SEM_REGIME,
    }
    
    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO BREAKOUT - Estrutura Simples")
    print(f"{len(tickers)} papéis × {len(estrategias)} estratégias × {len(anos)} anos")
    print(f"{'='*70}\n")
    
    # Resultados: {estratégia: {ano: [trades]}}
    resultados = {nome: {ano: [] for ano in anos} for nome in estrategias.keys()}
    
    # LOOP PRINCIPAL: POR PAPEL
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        try:
            # Preparar dados completos (1× por papel)
            df = preparar_df(ticker, date(ano_atual, 12, 31))
            
            if df.empty or len(df) < 30:
                print("dados insuficientes")
                continue
            
            # Testar cada estratégia
            for nome_estrategia, estrategia in estrategias.items():
                # Backtest completo
                trades = backtest_estrategia(ticker, df, estrategia)
                
                if trades.empty:
                    continue
                
                # Separar por ano
                trades["data_entrada"] = pd.to_datetime(trades["data_entrada"])
                
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
        
        except Exception as e:
            print(f"ERRO: {e}")
    
    # RESUMO
    print(f"\n{'='*70}")
    print("📊 RESULTADOS")
    print(f"{'='*70}\n")
    
    metricas_finais = {}
    
    for nome_estrategia in estrategias.keys():
        print(f"\n{'='*60}")
        print(f"{nome_estrategia}")
        print(f"{'='*60}")
        
        todos_trades = []
        
        for ano in anos:
            lista_trades = resultados[nome_estrategia][ano]
            
            if not lista_trades:
                print(f"  Ano {ano}: 0 trades")
                continue
            
            df_ano = pd.concat(lista_trades, ignore_index=True)
            todos_trades.append(df_ano)
            
            n = len(df_ano)
            acertos = (df_ano["retorno_%"] > 0).sum()
            taxa = acertos / n if n > 0 else 0
            
            print(f"  Ano {ano}: {n} trades | {taxa:.1%} acerto")
        
        # Consolidar
        if todos_trades:
            df_total = pd.concat(todos_trades, ignore_index=True)
            n_total = len(df_total)
            acertos_total = (df_total["retorno_%"] > 0).sum()
            taxa_media = acertos_total / n_total if n_total > 0 else 0
            ret_medio = df_total["retorno_%"].mean()
            
            metricas_finais[nome_estrategia] = {
                "total": n_total,
                "taxa": taxa_media,
                "retorno": ret_medio,
                "ops_ano": round(n_total / len(anos), 1),
            }
        else:
            metricas_finais[nome_estrategia] = {
                "total": 0,
                "taxa": 0,
                "retorno": 0,
                "ops_ano": 0,
            }
    
    # Análise final
    print(f"\n{'='*70}")
    print("💡 DIAGNÓSTICO")
    print(f"{'='*70}\n")
    
    basico = metricas_finais.get("1. BÁSICO", {}).get("total", 0)
    
    if basico == 0:
        print("❌ BREAKOUT não funciona na B3")
    else:
        print(f"✓ BREAKOUT gera sinais: {basico} trades totais")
        
        for nome, metricas in metricas_finais.items():
            if metricas["total"] > 0:
                print(f"\n{nome}:")
                print(f"  Total: {metricas['total']} trades")
                print(f"  Taxa: {metricas['taxa']:.1%}")
                print(f"  Retorno médio: {metricas['retorno']:.2f}%")
                print(f"  Ops/ano: {metricas['ops_ano']}")
    
    return metricas_finais


if __name__ == "__main__":
    diagnosticar_breakout_simples()