"""
Análise de Interseção: Por que nenhum sinal dispara?
"""

import sys
import argparse
from datetime import date

import pandas as pd
import numpy as np
import preprocessamento as pp
import estrategias.mean_reversion_light as mrl

CFG = {"dias_historico": 400}

def _val(row, col, default=np.nan):
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)

def preparar_df(ticker, data_base):
    df = pp.montar_dataset(ticker, dias_total_historico=CFG["dias_historico"], data_base=data_base)
    df = pp.adicionar_features_sniper(df)
    
    for n in [1, 3, 5, 10, 20]:
        col = f"ret_{n}d"
        if col not in df.columns:
            df[col] = df["close"].pct_change(n)
    
    if "close_ibov" in df.columns:
        df["mm50_ibov"]  = df["close_ibov"].rolling(50).mean()
        df["mm200_ibov"] = df["close_ibov"].rolling(200).mean()
    
    df["mm50"] = df["close"].rolling(50).mean()
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Ticker (ex: PETR4.SA)")
    parser.add_argument("ano", type=int, help="Ano (ex: 2025)")
    args = parser.parse_args()
    
    ticker = args.ticker
    ano = args.ano
    data_base = date(ano, 12, 31)
    
    print("\n" + "="*70)
    print(f"ANÁLISE DE INTERSEÇÃO: {ticker} em {ano}")
    print("="*70 + "\n")
    
    df = preparar_df(ticker, data_base)
    df_ano = df[(df.index >= date(ano, 1, 1)) & (df.index <= date(ano, 12, 31))]
    
    p = mrl.PARAMETROS
    
    # Criar flags para cada condição
    df_ano['cond1_rsi_min'] = df_ano['rsi'] >= p['rsi_min']
    df_ano['cond2_rsi_max'] = df_ano['rsi'] <= p['rsi_max']
    df_ano['cond3_queda'] = df_ano['ret_5d'] <= p['queda_5d_min']
    df_ano['cond4_mm50'] = df_ano['close'] > df_ano['mm50']
    df_ano['cond5_regime'] = df_ano['close_ibov'] > df_ano['mm200_ibov']
    df_ano['cond6_vol'] = df_ano['vol_relativo'] >= p['vol_relativo_min']
    df_ano['cond7_adx'] = df_ano['ADX_14'] <= p['adx_max']
    
    # Todas as condições
    df_ano['todas'] = (
        df_ano['cond1_rsi_min'] & 
        df_ano['cond2_rsi_max'] & 
        df_ano['cond3_queda'] & 
        df_ano['cond4_mm50'] & 
        df_ano['cond5_regime'] & 
        df_ano['cond6_vol'] & 
        df_ano['cond7_adx']
    )
    
    # Estatísticas
    print("CONDIÇÕES INDIVIDUAIS:")
    print(f"  1. RSI >= {p['rsi_min']}: {df_ano['cond1_rsi_min'].sum()} dias ({df_ano['cond1_rsi_min'].sum()/len(df_ano)*100:.1f}%)")
    print(f"  2. RSI <= {p['rsi_max']}: {df_ano['cond2_rsi_max'].sum()} dias ({df_ano['cond2_rsi_max'].sum()/len(df_ano)*100:.1f}%)")
    print(f"  3. Queda >= {p['queda_5d_min']*100:.0f}%: {df_ano['cond3_queda'].sum()} dias ({df_ano['cond3_queda'].sum()/len(df_ano)*100:.1f}%)")
    print(f"  4. Close > MM50: {df_ano['cond4_mm50'].sum()} dias ({df_ano['cond4_mm50'].sum()/len(df_ano)*100:.1f}%)")
    print(f"  5. Ibov > MM200: {df_ano['cond5_regime'].sum()} dias ({df_ano['cond5_regime'].sum()/len(df_ano)*100:.1f}%)")
    print(f"  6. Vol >= {p['vol_relativo_min']}: {df_ano['cond6_vol'].sum()} dias ({df_ano['cond6_vol'].sum()/len(df_ano)*100:.1f}%)")
    print(f"  7. ADX <= {p['adx_max']}: {df_ano['cond7_adx'].sum()} dias ({df_ano['cond7_adx'].sum()/len(df_ano)*100:.1f}%)")
    print()
    
    # Interseções críticas
    print("INTERSEÇÕES CRÍTICAS (condições sobrevendidas):")
    sobrevenda = df_ano['cond1_rsi_min'] & df_ano['cond2_rsi_max'] & df_ano['cond3_queda']
    print(f"  RSI 20-40 + Queda > 3%: {sobrevenda.sum()} dias ({sobrevenda.sum()/len(df_ano)*100:.1f}%)")
    
    sobrevenda_mm50 = sobrevenda & df_ano['cond4_mm50']
    print(f"  ↳ + Close > MM50: {sobrevenda_mm50.sum()} dias ({sobrevenda_mm50.sum()/len(df_ano)*100:.1f}%) ← GARGALO!")
    
    sobrevenda_regime = sobrevenda & df_ano['cond5_regime']
    print(f"  ↳ + Ibov > MM200: {sobrevenda_regime.sum()} dias ({sobrevenda_regime.sum()/len(df_ano)*100:.1f}%)")
    
    sobrevenda_mm50_regime = sobrevenda & df_ano['cond4_mm50'] & df_ano['cond5_regime']
    print(f"  ↳ + Close > MM50 + Ibov > MM200: {sobrevenda_mm50_regime.sum()} dias ({sobrevenda_mm50_regime.sum()/len(df_ano)*100:.1f}%)")
    
    # Todas
    print(f"\n  TODAS AS CONDIÇÕES: {df_ano['todas'].sum()} dias ({df_ano['todas'].sum()/len(df_ano)*100:.1f}%)")
    print()
    
    # Mostrar dias que quase passaram
    print("="*70)
    print("DIAS QUE FALHARAM POR POUCO (sobrevendidos mas fora da MM50):")
    print("="*70)
    
    quase = df_ano[
        sobrevenda & 
        df_ano['cond5_regime'] & 
        df_ano['cond6_vol'] & 
        df_ano['cond7_adx'] & 
        ~df_ano['cond4_mm50']  # Falhou só por MM50
    ].head(10)
    
    if len(quase) > 0:
        print("\nPrimeiros 10 dias que falharam APENAS por Close < MM50:\n")
        for idx, row in quase.iterrows():
            dist_mm50 = (row['close'] - row['mm50']) / row['mm50'] * 100
            print(f"  {idx}")
            print(f"    RSI: {row['rsi']:.1f} | ret_5d: {row['ret_5d']*100:.2f}%")
            print(f"    Close: {row['close']:.2f} vs MM50: {row['mm50']:.2f} ({dist_mm50:+.2f}%)")
            print()
    else:
        print("\nNenhum dia falhou apenas por MM50.")
    
    # Mostrar o contrário: dias acima MM50 mas não sobrevendidos
    print("="*70)
    print("DIAS ACIMA MM50 + REGIME OK MAS SEM SOBREVENDA:")
    print("="*70)
    
    acima_mm50 = df_ano[
        df_ano['cond4_mm50'] & 
        df_ano['cond5_regime'] & 
        ~sobrevenda
    ].head(10)
    
    if len(acima_mm50) > 0:
        print("\nPrimeiros 10 dias acima MM50 + regime mas SEM sobrevenda:\n")
        for idx, row in acima_mm50.iterrows():
            print(f"  {idx}")
            print(f"    RSI: {row['rsi']:.1f} | ret_5d: {row['ret_5d']*100:.2f}%")
            print(f"    Close: {row['close']:.2f} vs MM50: {row['mm50']:.2f}")
            print()
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()