"""
DIAGNÓSTICO COMPLETO - Mean Reversion Light
============================================

Execute este script no SEU ambiente (onde você tem preprocessamento.py e dados reais).

Comandos:
    python diagnostico_mean_reversion_final.py PETR4.SA 2025
"""

import sys
import argparse
from datetime import date
from pathlib import Path

# Adicionar paths conforme necessário
# sys.path.insert(0, '/caminho/para/seu/projeto')

import pandas as pd
import numpy as np
import preprocessamento as pp
import estrategias.mean_reversion_light as mrl

CFG = {
    "dias_historico": 400,
}

def _val(row, col, default=np.nan):
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)

def preparar_df(ticker, data_base):
    """Preparar DataFrame exatamente como o screening faz"""
    df = pp.montar_dataset(ticker, dias_total_historico=CFG["dias_historico"], data_base=data_base)
    df = pp.adicionar_features_sniper(df)
    
    # Retornos
    for n in [1, 3, 5, 10, 20]:
        col = f"ret_{n}d"
        if col not in df.columns:
            df[col] = df["close"].pct_change(n)
    
    # MMs do Ibovespa
    if "close_ibov" in df.columns:
        df["mm50_ibov"]  = df["close_ibov"].rolling(50).mean()
        df["mm200_ibov"] = df["close_ibov"].rolling(200).mean()
    
    # MMs do papel
    df["mm50"] = df["close"].rolling(50).mean()
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Ticker a testar (ex: PETR4.SA)")
    parser.add_argument("ano", type=int, help="Ano a analisar (ex: 2025)")
    args = parser.parse_args()
    
    ticker = args.ticker
    ano = args.ano
    data_base = date(ano, 12, 31)
    
    print("\n" + "="*70)
    print(f"DIAGNÓSTICO: {ticker} em {ano}")
    print("="*70 + "\n")
    
    # Preparar dados
    print("1. Preparando dados...")
    df = preparar_df(ticker, data_base)
    
    # Filtrar ano
    df_ano = df[(df.index >= date(ano, 1, 1)) & (df.index <= date(ano, 12, 31))]
    
    print(f"   Total de dias no ano: {len(df_ano)}")
    print(f"   Primeiros 5 dias: {df_ano.index[:5].tolist()}")
    print(f"   Últimos 5 dias: {df_ano.index[-5:].tolist()}\n")
    
    # Verificar colunas
    print("2. Verificando colunas necessárias...")
    colunas_necessarias = ['rsi', 'ret_5d', 'close', 'mm50', 'close_ibov', 'mm200_ibov', 'vol_relativo', 'ADX_14']
    
    for col in colunas_necessarias:
        existe = col in df_ano.columns
        if not existe:
            print(f"   ✗ {col} FALTANDO!")
        else:
            n_nulos = df_ano[col].isna().sum()
            print(f"   ✓ {col} (nulos: {n_nulos})")
    print()
    
    # Contar sinais
    print("3. Detectando sinais...\n")
    p = mrl.PARAMETROS
    sinais = []
    
    for idx, row in df_ano.iterrows():
        try:
            disparou = mrl.fn_sinal(row, p)
            if disparou:
                sinais.append({
                    'data': idx,
                    'rsi': _val(row, 'rsi'),
                    'ret_5d': _val(row, 'ret_5d') * 100,
                    'close': _val(row, 'close'),
                    'mm50': _val(row, 'mm50'),
                    'close_ibov': _val(row, 'close_ibov'),
                    'mm200_ibov': _val(row, 'mm200_ibov'),
                    'vol_relativo': _val(row, 'vol_relativo'),
                    'ADX_14': _val(row, 'ADX_14'),
                })
        except Exception as e:
            print(f"   ✗ ERRO em {idx.date()}: {e}")
    
    # Relatório
    print(f"\n{'='*70}")
    print(f"RESULTADOS PARA {ticker} EM {ano}:")
    print(f"{'='*70}\n")
    print(f"Total de sinais detectados: {len(sinais)}\n")
    
    if len(sinais) == 0:
        print("❌ NENHUM SINAL DETECTADO!")
        print("\nPossíveis causas:")
        print("  1. Ibovespa não estava acima da MM200 durante o ano")
        print("  2. Papel nunca ficou sobrevendido (RSI < 40)")
        print("  3. Papel estava abaixo da MM50")
        print("  4. Dados faltando (verifique colunas acima)")
        
        # Estatísticas do ano
        print("\n" + "-"*70)
        print("ESTATÍSTICAS DO ANO:")
        print(f"  Ibov acima MM200: {(df_ano['close_ibov'] > df_ano['mm200_ibov']).sum()} dias ({(df_ano['close_ibov'] > df_ano['mm200_ibov']).sum() / len(df_ano) * 100:.1f}%)")
        print(f"  Papel acima MM50: {(df_ano['close'] > df_ano['mm50']).sum()} dias ({(df_ano['close'] > df_ano['mm50']).sum() / len(df_ano) * 100:.1f}%)")
        print(f"  RSI < 40: {(df_ano['rsi'] < 40).sum()} dias ({(df_ano['rsi'] < 40).sum() / len(df_ano) * 100:.1f}%)")
        print(f"  Queda > 3% em 5d: {(df_ano['ret_5d'] < -0.03).sum()} dias ({(df_ano['ret_5d'] < -0.03).sum() / len(df_ano) * 100:.1f}%)")
        
    else:
        print(f"✓ {len(sinais)} sinal(is) encontrado(s):\n")
        
        for i, s in enumerate(sinais, 1):
            print(f"  [{i}] {s['data'].date()}")
            print(f"      RSI: {s['rsi']:.1f} | ret_5d: {s['ret_5d']:.2f}%")
            print(f"      close: {s['close']:.2f} > mm50: {s['mm50']:.2f} ✓")
            print(f"      ibov: {s['close_ibov']:.0f} > mm200: {s['mm200_ibov']:.0f} ✓")
            print(f"      vol_rel: {s['vol_relativo']:.2f} | ADX: {s['ADX_14']:.1f}")
            print()
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()