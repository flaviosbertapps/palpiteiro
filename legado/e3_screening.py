import pandas as pd
from datetime import date
import preprocessamento as pp  # Seu arquivo anexo

# Configurações
LISTA_PAPEIS = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'RENT3.SA', 'WEGE3.SA'] # Adicione mais
BASE_DATE = date.today()

def estrategia_trend_following(df):
    """Estratégia A: Trend Following (Médias + ADX)"""
    # Pega a última linha (hoje/ontem)
    last_row = df.iloc[-1]
    
    condicao_media = last_row['mm5'] > last_row['mm20']
    condicao_tendencia = last_row['adx_14'] > 25
    condicao_momentum = last_row['dmp_14'] > last_row['dmn_14']
    
    if condicao_media and condicao_tendencia and condicao_momentum:
        return "COMPRA (Tendência de Alta Forte)"
    return None

def estrategia_mean_reversion(df):
    """Estratégia B: IFR2 Estendido"""
    last_row = df.iloc[-1]
    
    # IFR2 < 10 indica sobrevenda extrema
    if last_row['ifr2'] < 10:
        return "COMPRA (Sobrevenda - IFR2)"
    return None

def executar_screening():
    print(f"--- Iniciando Screening Diário: {BASE_DATE} ---")
    oportunidades = []

    for ticker in LISTA_PAPEIS:
        try:
            print(f"Analisando {ticker}...")
            # 1. Monta o dataset usando seu script (500 dias para médias longas)
            df = pp.montar_dataset(ticker, data_base=BASE_DATE, dias_total_historico=500)
            
            # 2. Adiciona as features sniper do seu script
            df = pp.adicionar_features_sniper(df)
            
            if df.empty:
                continue

            # 3. Aplica os filtros das estratégias
            sinal_trend = estrategia_trend_following(df)
            sinal_mean = estrategia_mean_reversion(df)

            if sinal_trend:
                oportunidades.append({'ticker': ticker, 'estrategia': 'Trend Following', 'sinal': sinal_trend})
            if sinal_mean:
                oportunidades.append({'ticker': ticker, 'estrategia': 'Mean Reversion', 'sinal': sinal_mean})

        except Exception as e:
            print(f"Erro ao processar {ticker}: {e}")

    # Exibe Resultados
    if oportunidades:
        print("\n=== OPORTUNIDADES IDENTIFICADAS ===")
        print(pd.DataFrame(oportunidades))
    else:
        print("\nNenhuma operação recomendada para hoje.")

if __name__ == "__main__":
    executar_screening()