import pandas as pd
import numpy as np

# --- 1. ESTRATÉGIA DE TENDÊNCIA (AGRESSIVA) ---
def strategy_trend(df):
    """
    Estratégia focada em capturar tendências reais na B3,
    minimizando o ruído político e a volatilidade de curto prazo.
    """
    # --- 1. Filtro de Regime de Mercado (O "Não entre em furada") ---
    # Só compramos se o papel estiver acima da média de 200 dias.
    # Na B3, se está abaixo da 200, é "bear market" ou problema estrutural.
    df['filter_long_term'] = df['adj_close'] > df['mm200_close']
    
    # --- 2. Força do Movimento (O Filtro de Lateralização) ---
    # ADX > 25 na B3 é o divisor de águas entre tendência e bagunça.
    trend_strength = df['ADX_14'] > 20 
    bullish_direction = df['DMP_14'] > df['DMN_14']
    
    # --- 3. Gatilho de Entrada (Médio Prazo) ---
    # Trocamos a MM5 (ruído) pela MM21 (tendência de swing trade).
    # O preço deve estar acima da média e a média deve estar apontando para cima.
    above_trend_average = df['adj_close'] > df['mm20']
    mm20_rising = df['mm20'].diff(3) > 0 # Média subindo nos últimos 3 dias
    
    # --- 4. Confirmação de Fluxo (Volume) ---
    # O volume deve ser pelo menos 20% maior que a média recente.
    # Sem volume, o rompimento na B3 costuma ser falso.
    volume_confirm = df['vol_relativo'] > 1.2
    
    # --- 5. O Sinal Final ---
    df['entry_signal'] = (
        df['filter_long_term'] & 
        trend_strength & 
        bullish_direction & 
        above_trend_average & 
        mm20_rising &
        volume_confirm
    ).astype(int)
    
    return df

# --- 2. ESTRATÉGIA DE RETORNO À MÉDIA (SNIPER) ---
def strategy_mean_reversion(df):
    # Filtro de RSI mais sensível
    oversold = df['rsi'] < 38 
    # Compra na "picada" abaixo da banda inferior
    near_bottom_band = df['adj_close'] <= (df['BBL_20_2.0'] * 1.015)
    
    df['entry_signal'] = (oversold & near_bottom_band).astype(int)
    return df

# --- 3. ESTRATÉGIA DE BREAKOUT (IMPULSO) ---
def strategy_breakout(df):
    # Rompimento da banda superior com ADX subindo
    breakout_upper = df['adj_close'] > df['BBU_20_2.0']
    momentum = df['ADX_14'].diff() > 0 # ADX ganhando inclinação
    
    df['entry_signal'] = (breakout_upper & momentum).astype(int)
    return df

# --- 4. ESTRATÉGIA DE MOMENTUM SWING (VAPT-VUPT) ---
def strategy_momentum_swing(df):
    # Foco total na MM5 e fechamento anterior
    mm5_upturn = df['mm5'] > df['mm5'].shift(1)
    price_crossing = (df['adj_close'] > df['mm5']) & (df['adj_close'].shift(1) <= df['mm5'].shift(1))
    
    df['entry_signal'] = (mm5_upturn & price_crossing).astype(int)
    return df

def strategy_ifr2_quant(df):
    """
    Estratégia Quant de IFR2 com cálculo interno do indicador.
    """
    # 1. Cálculo do IFR de 2 períodos (RSI 2)
    change = df['adj_close'].diff()
    gain = change.mask(change < 0, 0)
    loss = -change.mask(change > 0, 0)

    # Usamos a média aritmética simples (SMA) para janelas muito curtas como 2
    avg_gain = gain.rolling(window=2).mean()
    avg_loss = loss.rolling(window=2).mean()

    # Evita divisão por zero
    rs = avg_gain / avg_loss
    df['ifr2'] = 100 - (100 / (1 + rs))
    
    # 2. Filtro de tendência de longo prazo (Média de 200 dias)
    # Certifique-se que mm200_close existe ou calcule-a aqui
    if 'mm200_close' not in df.columns:
        df['mm200_close'] = df['adj_close'].rolling(window=200).mean()
        
    trend_filter = df['adj_close'] > df['mm200_close']
    
    # 3. Gatilho de Entrada: Sobrevenda extrema (IFR2 < 10)
    oversold = df['ifr2'] < 10
    
    # 4. Gerar sinal
    df['entry_signal'] = (trend_filter & oversold).astype(int)
    
    return df