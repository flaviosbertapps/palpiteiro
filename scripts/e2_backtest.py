import pandas as pd
import numpy as np

def run_backtest(df, atr_multiplier=2.0, max_days=15):
    trades = []
    in_position = False
    entry_price = 0
    stop_loss = 0
    days_held = 0
    
    for i in range(len(df)):
        current_row = df.iloc[i]
        
        # Lógica de ENTRADA
        if not in_position and current_row['entry_signal'] == 1:
            in_position = True
            entry_price = current_row['adj_close']
            # Stop inicial baseado no ATR
            stop_loss = entry_price - (current_row['atr'] * atr_multiplier)
            days_held = 0
            entry_date = df.index[i]
            continue
            
        # Lógica de MONITORAMENTO (Durante a operação)
        if in_position:
            days_held += 1
            
            # Atualiza o Trailing Stop (Sobe o stop, mas nunca desce)
            new_stop = current_row['adj_close'] - (current_row['atr'] * atr_multiplier)
            if new_stop > stop_loss:
                stop_loss = new_stop
            
            # Condições de SAÍDA
            hit_stop = current_row['low'] <= stop_loss
            time_limit = days_held >= max_days
            
            if hit_stop or time_limit:
                exit_price = stop_loss if hit_stop else current_row['adj_close']
                profit_pct = (exit_price / entry_price) - 1
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'reason': 'Stop' if hit_stop else 'Time'
                })
                in_position = False
                
    return pd.DataFrame(trades)


def run_backtest_with_target(df, take_profit_pct=0.04, stop_loss_atr=2.0, max_days=10):
    # 1. Garantir que não há NaNs nos indicadores críticos
    df = df.dropna(subset=['atr', 'open', 'high', 'low', 'adj_close']).copy()
    df = df.sort_index(ascending=True)
    
    trades = []
    in_position = False
    
    for i in range(len(df) - 1):
        # --- BUSCA DE SINAL ---
        if not in_position:
            if df.iloc[i]['entry_signal'] == 1:
                # Entrada na ABERTURA do dia seguinte (T+1)
                entry_row = df.iloc[i+1]
                entry_price = entry_row['open']
                
                # Cálculo de Alvo e Stop
                target_price = entry_price * (1 + take_profit_pct)
                stop_loss = entry_price - (entry_row['atr'] * stop_loss_atr)
                
                # Verificação de Sanidade: Se o stop for inválido, não entra
                if np.isnan(stop_loss): continue
                
                entry_date = df.index[i+1]
                days_held = 0
                in_position = True
                continue # Pula para o dia seguinte para começar a monitorar

        # --- GESTÃO DA POSIÇÃO ---
        else:
            days_held += 1
            current_row = df.iloc[i]
            
            # PRIORIDADE 1: Verificar se o Stop Loss foi atingido primeiro (Conservador)
            hit_stop = current_row['low'] <= stop_loss
            
            # PRIORIDADE 2: Verificar Take Profit
            hit_target = current_row['high'] >= target_price
            
            # PRIORIDADE 3: Tempo limite
            time_limit = days_held >= max_days
            
            if hit_stop or hit_target or time_limit:
                # Se atingiu stop e target no mesmo dia, assume-se STOP
                if hit_stop:
                    exit_price = stop_loss
                    reason = 'Stop Loss'
                elif hit_target:
                    exit_price = target_price
                    reason = 'Take Profit'
                else:
                    exit_price = current_row['adj_close']
                    reason = 'Time Out'
                
                profit_pct = (exit_price / entry_price) - 1
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'profit_pct': profit_pct,
                    'reason': reason,
                    'days': days_held
                })
                in_position = False
                
    return pd.DataFrame(trades)


import numpy as np
import pandas as pd

def run_backtest_trend_following(df, stop_loss_atr=2.0, trailing_atr=2.5):
    """
    Versão Bulletproof: Remove o alvo fixo e implementa Trailing Stop.
    """
    df = df.dropna(subset=['atr', 'open', 'high', 'low', 'adj_close']).copy()
    df = df.sort_index(ascending=True)
    
    trades = []
    in_position = False
    
    for i in range(len(df) - 1):
        if not in_position:
            if df.iloc[i]['entry_signal'] == 1:
                # Entrada na ABERTURA do dia seguinte
                entry_row = df.iloc[i+1]
                entry_price = entry_row['open']
                entry_date = df.index[i+1]
                
                # Stop Inicial (Hard Stop)
                current_stop = entry_price - (entry_row['atr'] * stop_loss_atr)
                
                if np.isnan(current_stop): continue
                
                days_held = 0
                max_price_seen = entry_price
                in_position = True
                continue

        else:
            days_held += 1
            current_row = df.iloc[i]
            
            # Atualiza o preço máximo para o Trailing Stop
            if current_row['high'] > max_price_seen:
                max_price_seen = current_row['high']
            
            # TRAILING STOP: O stop sobe, mas nunca desce.
            # Ele protege o lucro conforme a tendência avança.
            dynamic_stop = max_price_seen - (current_row['atr'] * trailing_atr)
            current_stop = max(current_stop, dynamic_stop)
            
            # Verificação de Saída: Apenas Stop (Inicial ou Trailing)
            # Removemos o Take Profit de 4% e o Time Out.
            hit_stop = current_row['low'] <= current_stop
            
            if hit_stop:
                exit_price = current_stop
                profit_pct = (exit_price / entry_price) - 1
                profit_pct -= 0.002 # Subtrai 0.2% fixo de taxas/slippage total
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'profit_pct': profit_pct,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'days': days_held,
                    'reason': 'Trailing Stop'
                })
                in_position = False
                
    return pd.DataFrame(trades)