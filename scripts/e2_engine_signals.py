import pandas as pd
import numpy as np

# --- 1. ESTRATÉGIA DE TENDÊNCIA (AGRESSIVA) ---
def strategy_trend(df):
    last = df.iloc[-1]
    
    # Critérios
    c1 = last['adj_close'] > last['mm200_close']
    c2 = last['ADX_14'] > 25  # Ajustado para 25 (conforme seu comentário original)
    c3 = last['DMP_14'] > last['DMN_14']
    c4 = last['adj_close'] > last['mm20']
    c5 = (df['mm20'].diff(3).iloc[-1]) > 0
    c6 = last['vol_relativo'] > 1.2

    df['entry_signal'] = (c1 & c2 & c3 & c4 & c5 & c6)
    
    checklist = {
        'Preço > MM200': (last['adj_close'], last['mm200_close'], '>'),
        'ADX Força': (last['ADX_14'], 25, '>'),
        'DMI+ > DMI-': (last['DMP_14'], last['DMN_14'], '>'),
        'Preço > MM20': (last['adj_close'], last['mm20'], '>'),
        'MM20 Inclinada': (df['mm20'].diff(3).iloc[-1], 0, '>'),
        'Vol. Relativo': (last['vol_relativo'], 1.2, '>')
    }
    return df, checklist

# --- 2. ESTRATÉGIA DE RETORNO À MÉDIA (SNIPER) ---
def strategy_mean_reversion(df):
    last = df.iloc[-1]
    limit_band = last['BBL_20_2.0'] * 1.015
    
    c1 = last['rsi'] < 38
    c2 = last['adj_close'] <= limit_band
    
    df['entry_signal'] = (c1 & c2)
    
    checklist = {
        'RSI (Sobrevenda)': (last['rsi'], 38, '<'),
        'Preço na Banda Inf.': (last['adj_close'], limit_band, '<=')
    }
    return df, checklist

# --- 3. ESTRATÉGIA DE BREAKOUT (IMPULSO) ---
def strategy_breakout(df):
    last = df.iloc[-1]
    adx_diff = df['ADX_14'].diff().iloc[-1]
    
    c1 = last['adj_close'] > last['BBU_20_2.0']
    c2 = adx_diff > 0
    
    df['entry_signal'] = (c1 & c2)
    
    checklist = {
        'Rompimento Banda Sup.': (last['adj_close'], last['BBU_20_2.0'], '>'),
        'Inclin. ADX (Momento)': (adx_diff, 0, '>')
    }
    return df, checklist

# --- 4. ESTRATÉGIA DE MOMENTUM SWING (VAPT-VUPT) ---
def strategy_momentum_swing(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    c1 = last['mm5'] > prev['mm5']
    c2 = (last['adj_close'] > last['mm5']) and (prev['adj_close'] <= prev['mm5'])
    c3 = last['vol_relativo'] > 1.0 # Adicionado como filtro de segurança
    
    df['entry_signal'] = (c1 & c2 & c3)
    
    checklist = {
        'MM5 Virada pra Cima': (last['mm5'], prev['mm5'], '>'),
        'Cruzamento Preço/MM5': (last['adj_close'], last['mm5'], '>'),
        'Filtro Volume': (last['vol_relativo'], 1.0, '>')
    }
    return df, checklist

# --- 5. ESTRATÉGIA IFR2 QUANT ---
def strategy_ifr2_quant(df):   
    
    if 'mm200_close' not in df.columns:
        df['mm200_close'] = df['adj_close'].rolling(window=200).mean()
    
    last = df.iloc[-1]
    c1 = last['adj_close'] > last['mm200_close']
    c2 = last['ifr2'] < 10
    
    df['entry_signal'] = (c1 & c2)
    
    checklist = {
        'Filtro MM200': (last['adj_close'], last['mm200_close'], '>'),
        'IFR2 Extremo': (last['ifr2'], 10, '<')
    }
    return df, checklist