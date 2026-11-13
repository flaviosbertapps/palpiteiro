import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import pandas_ta as ta
import requests
import io
import os
import time
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, date
import random

import utils

CAMINHO_DADOS = '/palpiteiro/dados'


def verificar_divergencia_historica_e_alertar(ticker, df_referencia, df_comparacao,
                                             num_datas_amostra=5, tolerancia_pct=1.0,
                                             contexto_verificacao=""): # Novo: para log
    """Compara amostras e alerta. Retorna True se divergência detectada."""
    if df_referencia is None or df_referencia.empty or df_comparacao is None or df_comparacao.empty:
        return False

    df_referencia['data'] = pd.to_datetime(df_referencia['data']).dt.date
    df_comparacao['data'] = pd.to_datetime(df_comparacao['data']).dt.date

    df_merged = pd.merge(
        df_referencia[['data', 'close']].rename(columns={'close': 'close_ref'}).dropna(),
        df_comparacao[['data', 'close']].rename(columns={'close': 'close_comp'}).dropna(),
        on='data', how='inner'
    )

    if df_merged.empty:
        #print(f"[{ticker}] {contexto_verificacao}: Nenhuma data comum para verificação de divergência.")
        return False

    num_amostras = min(num_datas_amostra, len(df_merged))
    if num_amostras == 0: return False
    amostras = df_merged.sample(n=num_amostras, random_state=1) # random_state para consistência

    divergencia_flag = False
    detalhes_email = [f"Verificação de divergência ({contexto_verificacao}) para {ticker} (amostras):"]

    for _, row in amostras.iterrows():
        data_v, close_r, close_c = row['data'], row['close_ref'], row['close_comp']
        diff = abs((close_c - close_r) / close_r) * 100 if close_r != 0 else (0 if close_c == 0 else float('inf'))
        if diff > tolerancia_pct:
            divergencia_flag = True
            msg = (f"  - Data {data_v.strftime('%Y-%m-%d')}: Ref={close_r:.2f}, "
                   f"Comp={close_c:.2f} -> DIVERGÊNCIA ({diff:.2f}%)")
            print(f"[{ticker}] {contexto_verificacao}: {msg}")
            detalhes_email.append(msg)

    if divergencia_flag:
        print(f"ALERTA: Divergência significativa ({contexto_verificacao}) detectada para {ticker}! Modelo pode estar obsoleto.")
        corpo_alerta = (
            f"Foi detectada uma divergência ({contexto_verificacao}) nos dados históricos para {ticker}."
            # ... (resto do corpo do e-mail como antes) ...
            f"\nDetalhes das amostras (tolerância > {tolerancia_pct:.2f}%):\n" + "\n".join(detalhes_email)
        )        
        utils.enviar_email_gmail_com_app_password(                
            f"ALERTA URGENTE ({contexto_verificacao}): Dados Históricos Alterados para {ticker}?",
            corpo_texto=corpo_alerta
        )
        return True
    else:
        print(f"[{ticker}] {contexto_verificacao}: Nenhuma divergência significativa nas amostras (tolerância {tolerancia_pct:.2f}%).")
        return False


def processar_df_baixado(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """
    Processa o DataFrame baixado do yfinance, mantendo todas as colunas de preço/volume
    e ajustando a coluna de data.

    Colunas esperadas do yfinance: Open, High, Low, Close, Adj Close, Volume
    Coluna de saída de data: 'data' (tipo datetime.date)
    Nomes de coluna de saída: minúsculos ('open', 'high', 'low', 'close', 'adj_close', 'volume')
    """
    if df_bruto is None or df_bruto.empty:
        # Retorna um DataFrame vazio com as colunas esperadas se a entrada for vazia
        colunas_esperadas_saida = ['data', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        return pd.DataFrame(columns=colunas_esperadas_saida)

    df = df_bruto.copy() # Trabalhar com uma cópia para evitar modificar o DataFrame original

    # 1. Resetar o índice para transformar 'Date' (índice do yfinance) em uma coluna
    df = df.reset_index()

    # 2. Renomear a coluna de data e converter para datetime.date
    # O yfinance retorna o índice como 'Date' ou às vezes 'Datetime'
    coluna_data_original = None
    if 'Date' in df.columns:
        coluna_data_original = 'Date'
    elif 'Datetime' in df.columns: # Algumas versões/chamadas do yf podem usar 'Datetime'
        coluna_data_original = 'Datetime'
    
    if coluna_data_original:
        df.rename(columns={coluna_data_original: 'data'}, inplace=True)
        # Converte para datetime, remove informações de fuso horário (se houver),
        # normaliza para meia-noite e então pega apenas a parte da data.
        df['data'] = pd.to_datetime(df['data']).dt.normalize().dt.date
    else:
        # Se não encontrar 'Date' ou 'Datetime', algo está errado com os dados brutos
        # ou yfinance mudou seu formato de output.
        print("AVISO: Coluna 'Date' ou 'Datetime' não encontrada no DataFrame bruto. A coluna de data pode não ser processada.")
        # Mesmo assim, tentaremos processar as outras colunas.

    # 3. Renomear colunas de preço/volume para minúsculas
    colunas_rename_map = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close', # 'Adj Close' é o nome comum do Yahoo
        'Volume': 'volume'
    }
    df.columns = df.columns.get_level_values(0)
    df.rename(columns=colunas_rename_map, inplace=True)

    # 4. Selecionar e reordenar as colunas de interesse (opcional, mas bom para consistência)
    colunas_finais = ['data'] # Começa com data
    #print (df.columns)
    # Adiciona as colunas OHLCV se existirem, na ordem desejada
    for col_esperada in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
        if col_esperada in df.columns:
            colunas_finais.append(col_esperada)
    
    # Se alguma coluna esperada não foi encontrada no df, ela simplesmente não será incluída
    df = df[colunas_finais]


    # 5. Remover linhas onde o 'close' é NaN (geralmente indica dias sem negociação que yfinance pode retornar)
    # Ou onde 'data' pode ter ficado NaT após conversões (improvável aqui, mas seguro)
    df.dropna(subset=['data', 'close'], inplace=True)
    
    # Se, após tudo, o DataFrame ficar vazio, retorna um com as colunas esperadas
    if df.empty:
        colunas_esperadas_saida = ['data', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        return pd.DataFrame(columns=colunas_esperadas_saida)

    # print(f'Tipos do df processado (saída): \n{df.dtypes}') # Para depuração
    # print(f'Head do df processado (saída): \n{df.head(2)}')
    return df


def baixar_dados_preco_cache_avancado(ticker, inicio_solicitado_obj: date, fim_solicitado_obj: date,
                                    forcar_atualizacao_total_cache=False,
                                    verificar_divergencia_periodica=True,
                                    dias_antiguidade_cache_para_verificar=7,
                                    dias_sobreposicao_verificacao=30,
                                    num_amostras_verificacao=3,
                                    tolerancia_divergencia_pct=1.0,
                                    dias_atualizacao_recente=5): # <<< NOVO PARÂMETRO
    """
    Baixa dados de preço do yfinance, usando um cache por papel.
    Sempre atualiza os últimos 'dias_atualizacao_recente' para garantir dados de fechamento.
    """
    pasta_dados = CAMINHO_DADOS
    os.makedirs(pasta_dados, exist_ok=True)
    nome_arquivo_cache = f"{ticker.replace('.SA', '')}.csv"
    caminho_arquivo_cache = os.path.join(pasta_dados, nome_arquivo_cache)

    if not isinstance(inicio_solicitado_obj, date) or not isinstance(fim_solicitado_obj, date):
        raise TypeError("inicio_solicitado_obj e fim_solicitado_obj devem ser objetos datetime.date.")

    df_cache_disco = None
    if os.path.exists(caminho_arquivo_cache):
        try:
            df_cache_disco = pd.read_csv(caminho_arquivo_cache, parse_dates=['data'])
            if not df_cache_disco.empty:
                df_cache_disco['data'] = pd.to_datetime(df_cache_disco['data']).dt.date
        except Exception as e:
            print(f"[{ticker}] Erro ao ler cache {caminho_arquivo_cache}: {e}. Tratando como se não existisse.")
            df_cache_disco = None

    df_para_filtrar = df_cache_disco
    necessidade_download_total = forcar_atualizacao_total_cache

    if not forcar_atualizacao_total_cache:
        if df_cache_disco is None or df_cache_disco.empty:
            #print(f"[{ticker}] Cache não existe ou está vazio. Download completo necessário.")
            necessidade_download_total = True
        else:
            data_min_cache = df_cache_disco['data'].min()
            data_max_cache = df_cache_disco['data'].max()
            #print(f"[{ticker}] Cache existente cobre de {data_min_cache} a {data_max_cache}.")

            if data_min_cache <= inicio_solicitado_obj and data_max_cache >= fim_solicitado_obj:
                #print(f"[{ticker}] Cache cobre período solicitado ({inicio_solicitado_obj} a {fim_solicitado_obj}).")
                
                # <<< ALTERAÇÃO PRINCIPAL INICIA AQUI >>>
                #print(f"[{ticker}] Garantindo atualização dos últimos {dias_atualizacao_recente} dias do cache...")
                # Define o período para a pequena atualização
                data_inicio_att_recente = max(data_min_cache, data_max_cache - timedelta(days=dias_atualizacao_recente * 2)) # Pega uma janela um pouco maior para garantir dias úteis
                data_fim_att_recente_yf = data_max_cache + timedelta(days=1)

                try:
                    df_att_recente_bruto = yf.download(ticker, start=data_inicio_att_recente, end=data_fim_att_recente_yf, progress=False, auto_adjust=False, actions=True)
                    df_att_recente_proc = processar_df_baixado(df_att_recente_bruto) # Sua função que retorna todas as colunas

                    if not df_att_recente_proc.empty:
                        # Atualiza o df_cache_disco em memória com os dados mais recentes
                        # Mantém os dados antigos, exceto para as datas que foram re-baixadas
                        df_cache_disco = df_cache_disco.set_index('data')
                        df_att_recente_proc = df_att_recente_proc.set_index('data')
                        
                        df_cache_disco.update(df_att_recente_proc) # Atualiza as linhas correspondentes
                        df_cache_disco = df_cache_disco.reset_index() # Retorna 'data' a ser uma coluna
                        df_para_filtrar = df_cache_disco # Usa o DF atualizado em memória
                        
                        print(f"[{ticker}] Últimos dias atualizados em memória com sucesso.")
                    else:
                        #print(f"[{ticker}] AVISO: Nenhuma informação retornada na atualização dos últimos dias.")
                        # Procede usando o cache do disco como está
                        df_para_filtrar = df_cache_disco
                except Exception as e_att_recente:
                    #print(f"[{ticker}] Erro ao tentar atualizar os últimos dias do cache: {e_att_recente}. Usando cache como está.")
                    df_para_filtrar = df_cache_disco
                
                # <<< ALTERAÇÃO PRINCIPAL TERMINA AQUI >>>

                # A verificação periódica de divergência pode ocorrer aqui, se desejado.
                # Se uma divergência for detectada, ela setará 'necessidade_download_total = True'
                # ... (lógica de verificação periódica como na versão anterior) ...

            else: # Cache não cobre o período
                #print(f"[{ticker}] Cache NÃO cobre período solicitado. Download total necessário.")
                necessidade_download_total = True

    # Se, após todas as verificações, um download total for necessário
    if necessidade_download_total:
        #print(f"[{ticker}] Iniciando download de dados (lógica de atualização/complemento)...")
        data_inicio_dl = date(2020, 1, 1) # Ou sua data de início padrão
        data_fim_dl_yf = date.today() + timedelta(days=5) # Garante que pega dados futuros se fim de semana

        #print(f"[{ticker}] Período de download yfinance: {data_inicio_dl} a {data_fim_dl_yf - timedelta(days=1)}")
        try:
            df_dados_frescos_brutos = yf.download(ticker, start=data_inicio_dl, end=data_fim_dl_yf, progress=False, auto_adjust=False, actions=True)
            df_processado_fresco = processar_df_baixado(df_dados_frescos_brutos)

            if not df_processado_fresco.empty:
                if df_cache_disco is not None and not df_cache_disco.empty:
                    #print(f"[{ticker}] Realizando verificação de divergência entre cache antigo e dados frescos...")
                    verificar_divergencia_historica_e_alertar(
                        ticker, df_cache_disco, df_processado_fresco,
                        num_amostras_verificacao, tolerancia_divergencia_pct, "Pós-Download Total"
                    )
                
                df_processado_fresco.to_csv(caminho_arquivo_cache, index=False)
                #print(f"[{ticker}] Cache principal ATUALIZADO com dados frescos: {caminho_arquivo_cache}")
                df_para_filtrar = df_processado_fresco.copy()
            else:
                print(f"[{ticker}] AVISO: Nenhum dado retornado pelo yfinance para o download.")
                df_para_filtrar = df_cache_disco if df_cache_disco is not None else pd.DataFrame(columns=['data', 'close'])
                #if df_para_filtrar is not None and not df_para_filtrar.empty:
                    #print(f"[{ticker}] Usando cache do disco (se existia) devido à falha no download.")

        except Exception as e_download:
            print(f"[{ticker}] Erro durante o download ou processamento: {e_download}")
            df_para_filtrar = df_cache_disco if df_cache_disco is not None else pd.DataFrame(columns=['data', 'close'])
            #if df_para_filtrar is not None and not df_para_filtrar.empty:
                 #print(f"[{ticker}] Usando cache do disco (se existia) devido ao erro no download.")


    # --- Filtragem Final ---
    if df_para_filtrar is None or df_para_filtrar.empty:
        print(f"[{ticker}] AVISO FINAL: Nenhum dado disponível para {ticker}.")
        return pd.DataFrame() # Retorna DF vazio com colunas default se processar_df_baixado não for chamado

    df_para_filtrar['data'] = pd.to_datetime(df_para_filtrar['data']).dt.date

    resultado = df_para_filtrar[
        (df_para_filtrar['data'] >= inicio_solicitado_obj) &
        (df_para_filtrar['data'] <= fim_solicitado_obj)
    ].copy()


    # A verificação de sanidade deve ser feita aqui, no resultado final
    if not resultado.empty:
        pct_changes = resultado['close'].pct_change().abs()
        if (pct_changes > 0.3).any(): # Threshold de 30% de variação diária
            print(f"AVISO: Valores extremos detectados em {ticker}. USO NÃO RECOMENDADO ATÉ VERIFICAÇÃO.")
            raise ValueError(f"Valores extremos detectados em {ticker}. Verifique os dados baixados.")
            
    return resultado



def calcular_indicadores(df):
    df['pct_change'] = df['close'].pct_change()
    df['vol'] = df['pct_change'].rolling(10).std()
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['mm200_close'] = df['close'].rolling(window=200).mean()
    df['close_norm_mm200'] = df['close'] / df['mm200_close']

    # MACD
    macd = ta.macd(df['close'])
    if isinstance(macd, pd.DataFrame):
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
    else:
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan

    df.ta.adx(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    # Para a feature de regime de volatilidade
    df.ta.atr(length=5, append=True)  # Cria 'ATRr_5'
    df.ta.atr(length=50, append=True) # Cria 'ATRr_50'
    # Renomear para nomes mais simples se desejar
    df.rename(columns={'ATRr_5': 'atr_5', 'ATRr_50': 'atr_50'}, inplace=True)

    # indicadores de volume
    df_datetimeindex = df.copy()    
    df_datetimeindex.index = pd.to_datetime(df_datetimeindex['data'])

    df_datetimeindex['obv'] = ta.obv(df_datetimeindex['close'], df_datetimeindex['volume'])
    df_datetimeindex['vwap'] = ta.vwap(df_datetimeindex['high'], df_datetimeindex['low'], df_datetimeindex['close'], df_datetimeindex['volume'])
    df_datetimeindex['cmf'] = ta.cmf(df_datetimeindex['high'], df_datetimeindex['low'], df_datetimeindex['close'], df_datetimeindex['volume'], length=20)
    #df_datetimeindex['mfi'] = ta.mfi(df_datetimeindex['high'], df_datetimeindex['low'], df_datetimeindex['close'], df_datetimeindex['volume'], length=14)

    df_datetimeindex.index = df.index
    df['obv'] = df_datetimeindex['obv']
    df['vwap'] = df_datetimeindex['vwap']
    df['cmf'] = df_datetimeindex['cmf']

    # True Range (TR) e Average True Range (ATR)
    atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
    tr = ta.true_range(high=df['high'], low=df['low'], close=df['close'])

    df['tr'] = tr
    df['atr'] = atr    
    df['atr_pct'] = df['atr'] / df['close']
    df['tr_pct'] = df['tr'] / df['close']

    df['mm5'] = df['close'].rolling(window=5).mean()
    df['mm20'] = df['close'].rolling(window=20).mean()    



    return df

def adicionar_features_sniper(df):
    """
    Cria features de momentum e contexto relativo para modelos de árvore (XGBoost/LGBM).
    
    IMPORTANTE: Esta função DEVE ser aplicada no DataFrame COMPLETO (sem filtros de datas/linhas),
    pois usa cálculos de janela (diff, rolling) que exigem continuidade temporal.
    """
    # Evita modificar o original
    df = df.copy()
    
    # 1. Momentum do Preço e Médias (Distância Relativa)
    if 'mm20' in df.columns:
        df['dist_mm20'] = (df['close'] - df['mm20']) / df['mm20']
    
    # 2. Velocidade dos Osciladores (Delta)
    if 'rsi' in df.columns:
        # CORRETO: diff() aqui assume que o índice é sequencial (dias consecutivos)
        df['delta_rsi'] = df['rsi'].diff()
        df['delta_rsi_3d'] = df['rsi'].diff(3) 
    
    if 'adx_14' in df.columns:
        df['delta_adx'] = df['adx_14'].diff()
        
    # 3. Fluxo Institucional (Volume Relativo)
    if 'volume' in df.columns:
        df['vol_relativo'] = df['volume'] / df['volume'].rolling(20).mean()
        if 'obv' in df.columns:
            df['obv_trend_5d'] = df['obv'].diff(5)

    # 4. Volatilidade Relativa
    if 'atr' in df.columns:
        df['atr_relativo'] = df['atr'] / df['atr'].rolling(20).mean()

    # Cálculo interno do IFR2
    change = df['adj_close'].diff()
    gain = change.mask(change < 0, 0)
    loss = -change.mask(change > 0, 0)
    avg_gain = gain.rolling(window=2).mean()
    avg_loss = loss.rolling(window=2).mean()
    rs = avg_gain / avg_loss
    df['ifr2'] = 100 - (100 / (1 + rs))

    return df

# Carregar variáveis macro
def carregar_dados_macro(inicio, fim, pasta_cache=CAMINHO_DADOS):
    """
    Baixa (ou carrega do cache) os dados da SELIC, IPCA e dólar comercial do SGS (Banco Central).

    Parâmetros:
        inicio (datetime): Data de início.
        fim (datetime): Data de término.
        pasta_cache (str): Pasta onde os arquivos de cache serão salvos.

    Retorna:
        DataFrame diário com colunas 'selic', 'ipca' e 'dolar'.
    """
    # Formata datas para API (dd/mm/yyyy)
    dt_inicio_sgs = inicio.strftime('%d/%m/%Y')
    dt_fim_sgs = fim.strftime('%d/%m/%Y')

    # Nome do arquivo com base nas datas (em formato yyyymmdd para ordenação e unicidade)
    nome_arquivo = f'macro_{inicio.strftime("%Y%m%d")}_{fim.strftime("%Y%m%d")}.csv'
    caminho_arquivo = os.path.join(pasta_cache, nome_arquivo)

    # Cria pasta se não existir
    os.makedirs(pasta_cache, exist_ok=True)

    # Se cache existir, carrega e retorna
    if os.path.exists(caminho_arquivo):
        print(f'✅ Usando cache local: {caminho_arquivo}')
        df_macro = pd.read_csv(caminho_arquivo, index_col=0)
        df_macro['data'] = pd.to_datetime(df_macro['data'], format='%Y-%m-%d')
        df_macro.set_index('data', inplace=True)
        return df_macro

    print(f'⬇️ Baixando dados macro de {dt_inicio_sgs} a {dt_fim_sgs}...')

    def baixar_serie(codigo_serie, nome_coluna):
        print(f'Baixando série {codigo_serie} ({nome_coluna})...')
        url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=csv&dataInicial={dt_inicio_sgs}&dataFinal={dt_fim_sgs}'
        r = requests.get(url)
        r.encoding = 'utf-8'
        df = pd.read_csv(io.StringIO(r.text), sep=';')
        df['data'] = pd.to_datetime(df['data'], dayfirst=True).dt.date
        df[nome_coluna] = df['valor'].str.replace(',', '.').astype(float)
        return df[['data', nome_coluna]].set_index('data')

    # Baixar séries
    df_selic = baixar_serie(432, 'selic')
    df_ipca = baixar_serie(433, 'ipca')
    df_dolar = baixar_serie(1, 'dolar')

    # Merge dos dados
    df_macro = df_selic.merge(df_ipca, on='data', how='outer') \
                       .merge(df_dolar, on='data', how='outer') \
                       .sort_values('data')
    
    # Reindexar para dias úteis e aplicar ffill
    datas_diarias = pd.date_range(start=inicio, end=fim, freq='B')
    df_macro = df_macro.reindex(datas_diarias).ffill()

    # Salvar no cache
    df_macro.reset_index().rename(columns={'index': 'data'}).to_csv(caminho_arquivo)    
    print(f'💾 Dados salvos em cache: {caminho_arquivo}')

    return df_macro


def calcular_target(df, x_pct=0.03, dias_futuro=5):
    """
    Adiciona a coluna 'Target' ao DataFrame com base na valorização futura
    em relação ao fechamento do próprio dia.

    Parâmetros:
        df (DataFrame): deve conter a coluna 'close' e estar ordenado por data
        x_pct (float): percentual mínimo de valorização ou desvalorização (caso negativo)
        dias_futuro (int): quantos dias até atingir o alvo

    Retorna:
        df (DataFrame): com a coluna 'Target' adicionada
    """
    close = df['close'].values
    target = []

    for i in range(len(close)):
        if i + dias_futuro >= len(close):
            target.append(np.nan)
        else:
            preco_base = close[i]
            futuros = close[i + 1:i + 1 + dias_futuro]
            if (x_pct > 0):
                houve_valorizacao = any(f > preco_base * (1 + x_pct) for f in futuros)
            else:
                houve_valorizacao = any(f < preco_base * (1 + x_pct) for f in futuros)
            target.append(1 if houve_valorizacao else 0)

    df['Target'] = target
    return df

import pandas as pd
import numpy as np
from datetime import date, timedelta # Para o exemplo de uso

def calcular_target_com_stop(df: pd.DataFrame,
                                     valorizacao_alvo_pct: float = 0.03,
                                     dias_horizonte_alvo: int = 5,
                                     stop_loss_pct: float = None):
    """
    Adiciona a coluna 'Target' ao DataFrame.
    Target é 1 se o preço intraday (high/low) atingir 'valorizacao_alvo_pct'
    dentro de 'dias_horizonte_alvo', ANTES que o preço intraday atinja 'stop_loss_pct'.
    """
    if stop_loss_pct is not None and stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct deve ser um valor positivo.")

    # Agora high e low são sempre necessários se o alvo ou o stop existirem
    colunas_necessarias = ['close', 'high', 'low']
    if not all(col in df.columns for col in colunas_necessarias):
        raise ValueError(f"DataFrame deve conter as colunas: {', '.join(colunas_necessarias)}")

    # Usar .to_numpy() é um pouco mais moderno e pode ser mais rápido que .values
    close_prices = df['close'].to_numpy()
    high_prices = df['high'].to_numpy()
    low_prices = df['low'].to_numpy()
    
    n_linhas = len(df)
    target_col = np.full(n_linhas, np.nan) # Começa com NaN para linhas que não podem ser calculadas
    alvo_de_alta = valorizacao_alvo_pct > 0

    for i in range(n_linhas - dias_horizonte_alvo): # Loop mais eficiente, evita if dentro do loop
        preco_base = close_prices[i]
        if pd.isna(preco_base):
            continue

        nivel_alvo_calculado = preco_base * (1 + valorizacao_alvo_pct)
        
        nivel_stop_calculado = np.nan
        if stop_loss_pct is not None:
            if alvo_de_alta:
                nivel_stop_calculado = preco_base * (1 - stop_loss_pct)
            else: # Alvo de baixa
                nivel_stop_calculado = preco_base * (1 + stop_loss_pct)

        target_atingido = False
        stop_atingido = False

        # Verifica os próximos 'dias_horizonte_alvo' dias
        for k in range(1, dias_horizonte_alvo + 1):
            idx_futuro = i + k
            
            preco_high_futuro = high_prices[idx_futuro]
            preco_low_futuro = low_prices[idx_futuro]
            
            if pd.isna(preco_high_futuro) or pd.isna(preco_low_futuro):
                break # Interrompe a verificação para este 'i' se houver NaNs

            # 1. Verificar se o STOP foi atingido intraday
            if stop_loss_pct is not None:
                if alvo_de_alta and preco_low_futuro <= nivel_stop_calculado:
                    stop_atingido = True
                    break # Stop atingido, sai do loop k
                elif not alvo_de_alta and preco_high_futuro >= nivel_stop_calculado:
                    stop_atingido = True
                    break # Stop atingido

            # 2. Se o STOP NÃO foi atingido no dia k, verificar se o ALVO foi atingido intraday
            # <<< MUDANÇA PRINCIPAL AQUI >>>
            if alvo_de_alta:
                if preco_high_futuro >= nivel_alvo_calculado: # Usa a MÁXIMA para verificar alvo de alta
                    target_atingido = True
                    break # Alvo atingido, sai do loop k
            else: # Alvo de baixa
                if preco_low_futuro <= nivel_alvo_calculado: # Usa a MÍNIMA para verificar alvo de baixa
                    target_atingido = True
                    break # Alvo atingido

        # Definir o target final para o dia 'i'
        if target_atingido and not stop_atingido:
            target_col[i] = 1
        else: # Se stop foi atingido, ou se nem alvo nem stop foram atingidos no horizonte
            target_col[i] = 0
            
    df_com_target = df.copy()
    df_com_target['Target'] = target_col
    return df_com_target


def calcular_target_dinamico(df: pd.DataFrame,
                             multiplo_atr_alvo: float = 2.0,
                             dias_horizonte_alvo: int = 5,
                             multiplo_atr_stop: float = 1.3):
    """
    Adiciona a coluna 'Target' ao DataFrame usando alvos dinâmicos baseados no ATR.
    
    Target é 1 se o preço atingir (Close + multiplo_atr_alvo * ATR)
    dentro de 'dias_horizonte_alvo', ANTES de atingir (Close - multiplo_atr_stop * ATR).
    
    Parâmetros:
    - df: DataFrame contendo as colunas 'close', 'high', 'low' e 'atr'.
    - multiplo_atr_alvo: Quantas vezes o ATR o preço deve subir para ser considerado lucro.
    - dias_horizonte_alvo: Janela de tempo para o trade acontecer.
    - multiplo_atr_stop: Quantas vezes o ATR o preço pode cair antes de ser stopado.
    """
    
    # Validação básica
    if multiplo_atr_stop is not None and multiplo_atr_stop <= 0:
        raise ValueError("multiplo_atr_stop deve ser um valor positivo.")

    colunas_necessarias = ['close', 'high', 'low', 'atr']
    if not all(col in df.columns for col in colunas_necessarias):
        raise ValueError(f"DataFrame deve conter as colunas: {', '.join(colunas_necessarias)}")

    # Converter para numpy arrays para performance
    close_prices = df['close'].to_numpy()
    high_prices = df['high'].to_numpy()
    low_prices = df['low'].to_numpy()
    atr_values = df['atr'].to_numpy()
    
    n_linhas = len(df)
    target_col = np.full(n_linhas, np.nan) # Inicializa com NaN

    # Loop principal (vetorizado onde possível seria melhor, mas o loop com break é eficiente para horizon)
    for i in range(n_linhas - dias_horizonte_alvo):
        preco_entrada = close_prices[i]
        atr_atual = atr_values[i]
        
        # Pular se houver dados faltando no momento da entrada
        if pd.isna(preco_entrada) or pd.isna(atr_atual):
            continue

        # Definir níveis de Alvo e Stop baseados na volatilidade do dia 'i'
        nivel_alvo = preco_entrada + (atr_atual * multiplo_atr_alvo)
        nivel_stop = preco_entrada - (atr_atual * multiplo_atr_stop)

        target_atingido = False
        stop_atingido = False

        # Verifica o que acontece primeiro nos próximos dias
        for k in range(1, dias_horizonte_alvo + 1):
            idx_futuro = i + k
            
            preco_high_futuro = high_prices[idx_futuro]
            preco_low_futuro = low_prices[idx_futuro]
            
            # Se faltar dado futuro, não podemos determinar o resultado
            if pd.isna(preco_high_futuro) or pd.isna(preco_low_futuro):
                break 

            # 1. Verificar STOP (prioridade: se tocou no stop, o trade morreu)
            # Assumimos o pior caso: se o Low tocou no stop, fomos stopados.
            if preco_low_futuro <= nivel_stop:
                stop_atingido = True
                break # Sai do loop k, trade encerrado com perda

            # 2. Verificar ALVO
            # Se não foi stopado, verifica se a máxima tocou no alvo
            if preco_high_futuro >= nivel_alvo:
                target_atingido = True
                break # Sai do loop k, trade encerrado com ganho

        # Definir o target final (1 = Sucesso, 0 = Fracasso/Stop/Lateral)
        if target_atingido and not stop_atingido:
            target_col[i] = 1
        else:
            # Aqui 0 inclui tanto ser stopado quanto o tempo acabar sem atingir o alvo
            target_col[i] = 0
            
    df_com_target = df.copy()
    df_com_target['Target'] = target_col
    
    # Adicionar metadados úteis para análise posterior (opcional, mas recomendado)
    df_com_target['meta_alvo_preco'] = df_com_target['close'] + (df_com_target['atr'] * multiplo_atr_alvo)
    df_com_target['meta_stop_preco'] = df_com_target['close'] - (df_com_target['atr'] * multiplo_atr_stop)
    
    return df_com_target


# Função principal
def montar_dataset(papel, data_base, dias_total_historico=500, x_pct=0.03, dias_futuro=5):
    """
    Monta o dataset final para o modelo, combinando preços, indicadores e variáveis externas.

    Parâmetros:
    - papel: string com o código do papel (ex: 'PETR4.SA')
    - data_base: string ou datetime (ex: '2024-12-01'); último dia com dados disponíveis
    - dias_total_historico (int): quantidade de dias úteis de histórico a carregar (padrão: 500 dias).
    - x_pct (float): percentual mínimo de valorização para definir o target (padrão: 0.03)
    - dias_futuro (int): quantos dias à frente olhar para definir o target (padrão: 5 dias).
    """
    # Estima data inicial considerando apenas dias úteis (com margem de segurança)
    data_inicio = data_base - timedelta(days=int(dias_total_historico * 1.5))    
    
    df_preco = baixar_dados_preco_cache_avancado(ticker=papel, inicio_solicitado_obj=data_inicio, fim_solicitado_obj=data_base)
    df_preco = calcular_indicadores(df_preco)
    df_preco.index = pd.to_datetime(df_preco.index)


    # Carregar dados macro
    inicio_macro = (data_inicio - timedelta(days=365))
    fim_macro = data_base + timedelta(days=1)  # Para incluir o último dia
    df_macro = carregar_dados_macro(inicio_macro, fim_macro)            
    df_macro = df_macro[df_macro.index <= pd.to_datetime(data_base)]    

    time.sleep(2)
    df_bvsp = baixar_dados_preco_cache_avancado(ticker='^BVSP', inicio_solicitado_obj=data_inicio, fim_solicitado_obj=data_base)
    df_bova11 = baixar_dados_preco_cache_avancado(ticker='BOVA11.SA', inicio_solicitado_obj=data_inicio, fim_solicitado_obj=data_base)
    
    # dados de preços vem do BVSP, volume do BOVA11 para evitar atraso na atualização
    df_ibov_preco = df_bvsp[['data','open', 'high', 'low', 'close', 'adj_close']]
    df_ibov_volume = df_bova11[['data','volume']]
    df_ibov = pd.merge(df_ibov_preco, df_ibov_volume, on='data', how='outer')

    # Resetar índices para merge
    df_preco = df_preco
    df_ibov = df_ibov
    df_macro = df_macro.reset_index(names='data')
    df_macro['data'] = df_macro['data'].dt.date

    # Merge por data
    df_final = pd.merge(df_preco, df_macro, on='data', how='left')
    df_final = pd.merge(df_final, df_ibov, on='data', how='left', suffixes=('', '_ibov'))
    df_final = df_final.set_index('data').sort_index()    

    df_final = verificar_e_buscar_papeis_adicionais(df_final, papel)

    return df_final

def verificar_e_buscar_papeis_adicionais(df, ticker):
    dct_tickers = {
        'ITUB4.SA': ['BBAS3.SA', 'BBDC4.SA', 'SANB11.SA']
        , 'MRFG3.SA': ['BRFS3.SA', 'BEEF3.SA'] # 'JBSS3.SA'
    }

    lista_papeis_adicionais = dct_tickers.get(ticker, [])
    if lista_papeis_adicionais:
        print(f"Buscando dados adicionais para {ticker}: {lista_papeis_adicionais}")
        df = buscar_dados_papeis_adicionais(df, lista_papeis_adicionais)

    return df


def buscar_dados_papeis_adicionais(df, papeis_adicionais: list):
    """
    Busca dados adicionais de preços e indicadores para uma lista de papéis adicionais.
    Retorna um DataFrame com os dados combinados.

    Parâmetros:
    - df: DataFrame principal com os dados do papel base.
    - papeis_adicionais: lista de strings com os códigos dos papéis adicionais (ex: ['PETR3.SA', 'VALE3.SA']).

    Retorna:
    - DataFrame com os dados combinados do papel base e dos papéis adicionais.
    """
    for papel in papeis_adicionais:
        inicio_solicitado = df.index.min()
        fim_solicitado = df.index.max()

        print(type(inicio_solicitado), type(fim_solicitado))
        print(inicio_solicitado, fim_solicitado)

        df_adicional = baixar_dados_preco_cache_avancado(ticker=papel, inicio_solicitado_obj=df.index.min(), fim_solicitado_obj=df.index.max()+timedelta(days=1))
        df_adicional = calcular_indicadores(df_adicional)            

        df_adicional.rename(columns=lambda x: f"adc_{x}_{papel.replace('.SA', '')}" if x != 'data' else 'data', inplace=True)
        df = pd.merge(df, df_adicional, on='data', how='left')
        df.set_index('data', inplace=True)

    return df

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_X_y_agregado(df: pd.DataFrame, dias_entrada: int, dropna: bool = True, tipo_drop_na: str = 'linhas', papel: str = None) -> tuple:
    """
    Prepara os dataframes X e y para o modelo, usando uma janela deslizante de 'dias_entrada'.
    Usa colunas normalizadas pela MM200 (ex: 'close_norm_mm200') para features de preço,
    e assume que outros indicadores já estão calculados no df de entrada.
    """

    def inclinacao_trend(serie: pd.Series) -> float:
        serie_limpa = serie.dropna()
        if len(serie_limpa) < 2 or len(serie_limpa) < len(serie) * 0.7: return np.nan
        y_val = serie_limpa.values.reshape(-1, 1)
        x_val = np.arange(len(serie_limpa)).reshape(-1, 1)
        try:
            modelo = LinearRegression().fit(x_val, y_val)
            return modelo.coef_[0][0]
        except ValueError: return np.nan


    def retorno_desde_ultima_mudanca_sinal(series: pd.Series) -> float:
        if len(series) < 2: return np.nan
        returns = series.pct_change()
        sinais = np.sign(returns)
        mudancas = sinais.diff().fillna(0) != 0
        if not mudancas.any(): return np.nan
        idx_ultima_mudanca_real = mudancas[mudancas].index[-1]
        try:
            pos_na_serie_original = series.index.get_indexer_for([idx_ultima_mudanca_real])[0]
        except (KeyError, IndexError): return np.nan
        valor_inicial = series.iloc[pos_na_serie_original]
        valor_final = series.iloc[-1]
        if pd.isna(valor_inicial) or pd.isna(valor_final) or valor_inicial == 0: return np.nan
        return (valor_final / valor_inicial) - 1

    X_rows, y_list, indices_X = [], [], []
    soma_volume_janelas_target_1 = []

    # --- 1. Identificar papéis adicionais e suas colunas de interesse ---
    colunas_df = df.columns.tolist()
    map_papeis_adicionais = {}
    colunas_base_adc_interesse = [
        'close_norm_mm200', 'pct_change', 'rsi', 'atr_pct', 'macd_hist', 'volume',
        'ADX_14', 'DMP_14', 'DMN_14', 'BBB_20_2.0', 'BBP_20_2.0'
    ]
    for col in colunas_df:
        if col.startswith("adc_"):
            parts = col.split('_')
            if len(parts) >= 3:
                nome_papel_adc_sufixo = parts[-1]
                coluna_base_nome = "_".join(parts[1:-1])
                if coluna_base_nome in colunas_base_adc_interesse:
                    if nome_papel_adc_sufixo not in map_papeis_adicionais:
                        map_papeis_adicionais[nome_papel_adc_sufixo] = {}
                    map_papeis_adicionais[nome_papel_adc_sufixo][coluna_base_nome] = col

    # --- 2. Loop Principal sobre as Janelas ---
    for i in range(dias_entrada - 1, len(df)):
        janela = df.iloc[i - dias_entrada + 1:i + 1].copy()
        linha = {}

        # --- 3. Features do Papel Principal ---
        
        # 3a. Features de Preço (usando dados normalizados pela MM200)
        close_norm_principal_series = janela.get('close_norm_mm200')
        if close_norm_principal_series is not None and not close_norm_principal_series.empty:
            close_norm_principal_ultimo = close_norm_principal_series.iloc[-1]
            linha['close_norm_mean'] = close_norm_principal_series.mean()
            linha['close_norm_std'] = close_norm_principal_series.std()
            linha['close_norm_max'] = close_norm_principal_series.max()
            linha['close_norm_min'] = close_norm_principal_series.min()
            linha['close_norm_trend'] = inclinacao_trend(close_norm_principal_series)
            linha['close_norm_ultimo'] = close_norm_principal_ultimo
            if not pd.isna(close_norm_principal_series.iloc[0]) and close_norm_principal_series.iloc[0] != 0:
                linha['retorno_acumulado_norm'] = (close_norm_principal_ultimo / close_norm_principal_series.iloc[0]) - 1

        if 'close' in janela.columns:
            linha['retorno_acumulado_ultimos_5_dias'] = janela['close'].iloc[-1] / janela['close'].iloc[-5] - 1 if len(janela) >= 5 else np.nan

        # 3b. Features de Retorno (usando pct_change)
        pct_change_principal_series = janela.get('pct_change')
        if pct_change_principal_series is not None and not pct_change_principal_series.empty:
            linha['retorno_mean'] = pct_change_principal_series.mean()
            linha['retorno_std'] = pct_change_principal_series.std()
            linha['dias_alta_pct'] = (pct_change_principal_series > 0).mean()
            linha['dias_queda_pct'] = (pct_change_principal_series < 0).mean()
            pos_returns = pct_change_principal_series[pct_change_principal_series > 0]
            neg_returns = pct_change_principal_series[pct_change_principal_series < 0]
            linha['retorno_medio_pos'] = pos_returns.mean() if not pos_returns.empty else 0
            linha['retorno_medio_neg'] = neg_returns.mean() if not neg_returns.empty else 0
            
            
        
        # 3c. Features de Volume e Interação Volume-Preço
        if 'volume' in janela.columns:
            volume_series = janela['volume']
            linha['volume_mean'] = volume_series.mean()
            linha['volume_std'] = volume_series.std()
            linha['volume_trend'] = inclinacao_trend(volume_series)
            if not pd.isna(linha['volume_mean']) and linha['volume_mean'] > 0:
                linha['volume_ult_vs_mean'] = volume_series.iloc[-1] / linha['volume_mean'] - 1
            if pct_change_principal_series is not None and len(janela) > 1:
                ret_vol_df = pd.DataFrame({'retorno': pct_change_principal_series, 'volume': volume_series}).dropna()
                if len(ret_vol_df) >= 5:
                    linha['correl_retorno_volume'] = ret_vol_df['retorno'].corr(ret_vol_df['volume'])
        
        # 3d. Indicadores Técnicos
        colunas_indicadores_agg = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'obv', 'mfi', 'cmf', 'atr_pct']
        for indicador in colunas_indicadores_agg:
            if indicador in janela.columns:
                ind_series = janela[indicador]
                linha[f'{indicador}_mean'] = ind_series.mean()
                linha[f'{indicador}_std'] = ind_series.std()
                linha[f'{indicador}_ultimo'] = ind_series.iloc[-1] if not ind_series.empty else np.nan
                linha[f'{indicador}_trend'] = inclinacao_trend(ind_series)
        
        # 3e. Features de Regime/Contexto
        if 'ADX_14' in janela.columns:
            adx_series = janela['ADX_14']
            linha['adx_ultimo'] = adx_series.iloc[-1] if not adx_series.empty else np.nan
            linha['adx_trend'] = inclinacao_trend(adx_series)
            linha['tendencia_forte_adx25'] = 1 if linha.get('adx_ultimo', 0) > 25 else 0
        if 'BBB_20_2.0' in janela.columns:
            linha['bbands_largura_pct_ultimo'] = janela['BBB_20_2.0'].iloc[-1] if not janela['BBB_20_2.0'].empty else np.nan
            linha['bbands_largura_pct_trend'] = inclinacao_trend(janela['BBB_20_2.0'])
        if 'BBP_20_2.0' in janela.columns:
            linha['bbands_posicao_pct_ultimo'] = janela['BBP_20_2.0'].iloc[-1] if not janela['BBP_20_2.0'].empty else np.nan
            linha['bbands_exaustao_compra'] = 1 if linha.get('bbands_posicao_pct_ultimo', 0.5) > 1.0 else 0
            linha['bbands_exaustao_venda'] = 1 if linha.get('bbands_posicao_pct_ultimo', 0.5) < 0.0 else 0

        # --- 4. Features de Papéis Adicionais ---
        lista_close_norm_adicionais_ultimos = []
        for nome_adc, cols_adc_map in map_papeis_adicionais.items():
            col_close_norm_adc = cols_adc_map.get('close_norm_mm200')
            if col_close_norm_adc and col_close_norm_adc in janela.columns:
                close_norm_series_adc = janela[col_close_norm_adc]
                linha[f'adc_close_norm_ultimo_{nome_adc}'] = close_norm_series_adc.iloc[-1] if not close_norm_series_adc.empty else np.nan
                linha[f'adc_close_norm_ultimo_vs_principal_{nome_adc}'] = (close_norm_series_adc.iloc[-1] / linha.get('close_norm_ultimo', np.nan)) - 1 if not pd.isna(linha.get('close_norm_ultimo', np.nan)) and not pd.isna(close_norm_series_adc.iloc[-1]) and close_norm_series_adc.iloc[-1] != 0 else np.nan
                linha[f'adc_close_norm_trend_{nome_adc}'] = inclinacao_trend(close_norm_series_adc)
                if not close_norm_series_adc.empty:
                    lista_close_norm_adicionais_ultimos.append(close_norm_series_adc.iloc[-1])
            col_pct_change_adc = cols_adc_map.get('pct_change')
            if pct_change_principal_series is not None and not pct_change_principal_series.empty and \
               col_pct_change_adc and col_pct_change_adc in janela.columns and len(janela) > 1:
                df_ret_comb_adc = pd.DataFrame({'principal': pct_change_principal_series, 'adicional': janela[col_pct_change_adc]}).dropna()
                if len(df_ret_comb_adc) >= 5:
                    linha[f'adc_correl_com_principal_{nome_adc}'] = df_ret_comb_adc['principal'].corr(df_ret_comb_adc['adicional'])
            col_adx_adc = cols_adc_map.get('ADX_14')
            if col_adx_adc and col_adx_adc in janela.columns:
                linha[f'adc_adx_ultimo_{nome_adc}'] = janela[col_adx_adc].iloc[-1] if not janela[col_adx_adc].empty else np.nan

        # --- 5. Features Agregadas dos Adicionais ---
        if len(lista_close_norm_adicionais_ultimos)>0:
            linha['adc_close_norm_mean_geral_ultimo'] = np.nanmean(lista_close_norm_adicionais_ultimos)
            if 'close_norm_mm200' in janela.columns and not janela['close_norm_mm200'].empty:
                close_norm_ult = janela['close_norm_mm200'].iloc[-1]
                adc_mean_ult = linha['adc_close_norm_mean_geral_ultimo']
                if not pd.isna(close_norm_ult) and not pd.isna(adc_mean_ult) and adc_mean_ult != 0:
                    linha['close_norm_vs_adc_norm_mean_geral'] = (close_norm_ult / adc_mean_ult) - 1

        # --- 6. Macro e Ibovespa ---
        for macro in ['selic', 'ipca', 'dolar']:
            if macro in df.columns:
                serie_macro_completa = df[macro].iloc[:i + 1]
                if macro in janela.columns and not janela[macro].empty:
                    linha[f'{macro}_ultimo'] = janela[macro].iloc[-1]
                if macro == 'dolar':
                     linha[f'{macro}_trend'] = inclinacao_trend(janela[macro]) if macro in janela.columns else np.nan
                     linha[f'{macro}_std'] = janela[macro].std() if macro in janela.columns else np.nan
                     linha[f'retorno_desde_mudanca_{macro}'] = retorno_desde_ultima_mudanca_sinal(serie_macro_completa)
                if macro in ['selic', 'ipca']:
                    valor_atual = janela[macro].iloc[-1]     
                    serie_sem_atual = serie_macro_completa[serie_macro_completa != valor_atual]
                    ultimo_valor = serie_sem_atual.iloc[-1] if not serie_sem_atual.empty else np.nan
                    linha[f'{macro}_comparacao_anterior'] = np.sign(valor_atual - ultimo_valor) if not pd.isna(ultimo_valor) else 0
                    if not pd.isna(ultimo_valor):
                        dias_desde_mudanca = (janela.index[-1] - serie_sem_atual.index[-1]).days
                        linha[f'{macro}_dias_desde_mudanca'] = dias_desde_mudanca
                    else:
                        dias_desde_mudanca = (janela.index[-1] - serie_macro_completa.index[0]).days
                        linha[f'{macro}_dias_desde_mudanca'] = dias_desde_mudanca

        if 'close_ibov' in janela.columns:
            ibov_series_janela = janela['close_ibov']
            ibov_series_completa = df['close_ibov'].iloc[:i + 1]
            linha['ibov_trend'] = inclinacao_trend(ibov_series_janela)
            linha['ibov_std'] = ibov_series_janela.std()
            linha['retorno_desde_mudanca_ibov'] = retorno_desde_ultima_mudanca_sinal(ibov_series_completa)
            if not ibov_series_janela.empty:
                linha['ibov_amplitude'] = ibov_series_janela.max() - ibov_series_janela.min()
            if pct_change_principal_series is not None and not pct_change_principal_series.empty and len(janela) > 1:
                ret_principal_limpo = pct_change_principal_series.dropna()
                ret_ibov_limpo = ibov_series_janela.pct_change().dropna()
                df_ret_comb_ibov = pd.DataFrame({'principal': ret_principal_limpo, 'ibov': ret_ibov_limpo}).dropna()
                if len(df_ret_comb_ibov) >= 5:
                    linha['correl_papel_ibov'] = df_ret_comb_ibov['principal'].corr(df_ret_comb_ibov['ibov'])
                    cov = df_ret_comb_ibov['principal'].cov(df_ret_comb_ibov['ibov'])
                    var_ibov = df_ret_comb_ibov['ibov'].var()
                    beta_val = (cov / var_ibov) if var_ibov != 0 else np.nan
                    linha['beta_papel_ibov'] = beta_val
                    alpha_val = (df_ret_comb_ibov['principal'] - (beta_val * df_ret_comb_ibov['ibov'])).mean() if not pd.isna(beta_val) else (df_ret_comb_ibov['principal'] - df_ret_comb_ibov['ibov']).mean()
                    linha['alpha_papel_vs_ibov_mean'] = alpha_val

         # 1. Feature de Regime de Tendência (baseado no ADX)
        # Pega o último valor do ADX, DMP e DMN na janela. Usa 0 se a coluna não existir ou for NaN.
        adx_ultimo = janela.get('ADX_14', pd.Series([0])).iloc[-1]
        dmp_ultimo = janela.get('DMP_14', pd.Series([0])).iloc[-1]
        dmn_ultimo = janela.get('DMN_14', pd.Series([0])).iloc[-1]

        # Atribui 1 para tendência de alta forte, -1 para baixa forte, 0 para lateral/indefinido.
        if adx_ultimo > 25 and dmp_ultimo > dmn_ultimo:
            linha['regime_tendencia'] = 1  # Tendência de Alta Forte
        elif adx_ultimo > 25 and dmn_ultimo > dmp_ultimo:
            linha['regime_tendencia'] = -1 # Tendência de Baixa Forte
        else:
            linha['regime_tendencia'] = 0  # Mercado Lateral ou Indefinido

        # 2. Feature de Regime de Volatilidade (baseado na tendência da largura das Bandas de Bollinger)
        # Pega a série da largura das bandas. Se não existir, cria uma série vazia.
        bb_width_series = janela.get('BBB_20_2.0', pd.Series(dtype=float))
        
        # Compara a média dos últimos 3 dias com a média da janela inteira para ver a tendência recente.
        if len(bb_width_series.dropna()) > 5: # Precisa de dados mínimos para a comparação fazer sentido
            media_recente_vol = bb_width_series.tail(3).mean()
            media_geral_vol = bb_width_series.mean()
            
            # Se a média recente é significativamente maior/menor que a média geral, indica uma mudança.
            if media_recente_vol > media_geral_vol * 1.1: # 10% maior
                linha['regime_volatilidade'] = 1  # Expansão de Volatilidade
            elif media_recente_vol < media_geral_vol * 0.9: # 10% menor
                linha['regime_volatilidade'] = -1 # Contração de Volatilidade
            else:
                linha['regime_volatilidade'] = 0  # Volatilidade Estável
        else:
            linha['regime_volatilidade'] = 0 # Default se não houver dados suficientes


        # --- 8. Alvo e Construção de X e y ---
        if 'Target' in df.columns:
            alvo = df.iloc[i]['Target']
            X_rows.append(linha)
            y_list.append(alvo)
            indices_X.append(df.index[i])
            if alvo == 1 and 'volume' in janela.columns:
                soma_volume_janelas_target_1.append(janela['volume'].sum())

    X_df = pd.DataFrame(X_rows, index=indices_X)
    if X_df.empty:
        y_series = pd.Series(dtype='float64')
    else:
        y_series = pd.Series(y_list, index=X_df.index)

    X_df = features_temporais(X_df)

    if dropna:
        if not X_df.empty:
            if tipo_drop_na == 'linhas':
                len_antes = len(X_df)
                X_df.dropna(axis=0, how='any', inplace=True)
                y_series = y_series.loc[X_df.index]
                if len(X_df) < len_antes:
                    print(f"INFO: {len_antes - len(X_df)} linha(s) removidas de X e y devido a NaNs.")
            elif tipo_drop_na == 'colunas':
                cols_antes = X_df.shape[1]
                X_df.dropna(axis=1, how='any', inplace=True)
                if X_df.shape[1] < cols_antes:
                    print(f"INFO: {cols_antes - X_df.shape[1]} coluna(s) removidas de X por NaNs.")

    return X_df, y_series


def preparar_features_especificas_por_papel(linha, papel, janela, df, i):
    if (papel=='BRAP4.SA'):
        return

import sklearn.preprocessing

def features_temporais(df):
    """
    Adiciona features temporais ao DataFrame.
    """

    df_feriados = pd.read_csv('feriados.csv', parse_dates=['dt'])
    
    # transforma o índice em datetime
    idx_dt = pd.to_datetime(df.index)
    #df.index = pd.to_datetime(df.index)
    dia_semana = idx_dt.dayofweek
    #one-hot encoding do dia da semana
    dias_semana_dummies = pd.get_dummies(dia_semana, prefix='dia_semana', dtype=int)
    dias_semana_dummies.index = df.index 

    # concatena as colunas de dia da semana ao DataFrame
    
    df = pd.concat([df, dias_semana_dummies], axis=1)
    

    df['feriado'] = idx_dt.isin(df_feriados['dt']).astype(int)
    # verifica se o dia seguinte é fim de semana ou feriado
    df['proximo_dia_util'] = np.where((dia_semana==4) | (idx_dt + pd.Timedelta(days=1)).isin(df_feriados['dt']),False,True)

   
    return df