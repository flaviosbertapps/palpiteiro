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

def baixar_dados_preco(ticker, inicio, fim):
    """
    Baixa dados de preço do yfinance, salvando em cache para evitar excesso de requisições.
    
    Parâmetros:
    - ticker: str, código do papel (ex: "PETR4")
    - inicio: str, data de início no formato "YYYY-MM-DD"
    - fim: str, data de fim no formato "YYYY-MM-DD"
    - pasta_cache: str, nome da pasta onde salvar os CSVs

    Retorna:
    - DataFrame com colunas ['data', 'close']
    """
    # Criar pasta se não existir
    pasta_dados = '../dados'
    os.makedirs(pasta_dados, exist_ok=True)

    # Nome do arquivo cache
    nome_arquivo = f"{ticker}_{inicio}_{fim}.csv"
    caminho_arquivo = os.path.join(pasta_dados, nome_arquivo)

    # Verifica se já existe o CSV
    if os.path.exists(caminho_arquivo):
        print(f"Carregando dados de cache: {caminho_arquivo}")
        df = pd.read_csv(caminho_arquivo, parse_dates=['data'])
    else:
        print(f"Baixando dados do yfinance para {ticker} de {inicio} a {fim}...")
        
        ticker_yahoo = ticker

        #ajuste da data de término para pegar todos os dados
        data_termino_yf = fim + timedelta(days=1)
        df = yf.download(ticker_yahoo, start=inicio, end=data_termino_yf)
        if df.shape[0] == 0:
            raise ValueError(f"Dados vazios para o ticker {ticker_yahoo} entre {inicio} e {fim}.")

        df = df[['Close']].rename(columns={'Close': 'close'})
        df.index = df.index.tz_localize(None)
        df = df.reset_index()
        df.columns = df.columns.get_level_values(0)
        df.rename(columns={'Date': 'data'}, inplace=True)

        # Salvar no cache
        df.to_csv(caminho_arquivo, index=False)
        print(f"Dados salvos em cache: {caminho_arquivo}")

    return df


# Gerar indicadores técnicos
def calcular_indicadores(df):
    df['pct_change'] = df['close'].pct_change()
    df['vol'] = df['pct_change'].rolling(10).std()
    df['rsi'] = ta.rsi(df['close'], length=14)

    macd = ta.macd(df['close'])

    if isinstance(macd, pd.DataFrame):
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
    else:
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan

    return df

# Carregar variáveis macro
def carregar_dados_macro(inicio, fim, pasta_cache='../dados'):
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
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df[nome_coluna] = df['valor'].str.replace(',', '.').astype(float)
        return df[['data', nome_coluna]]

    # Baixar séries
    df_selic = baixar_serie(432, 'selic')
    df_ipca = baixar_serie(433, 'ipca')
    df_dolar = baixar_serie(1, 'dolar')

    # Merge dos dados
    df_macro = df_selic.merge(df_ipca, on='data', how='outer') \
                       .merge(df_dolar, on='data', how='outer') \
                       .sort_values('data')

    df_macro.set_index('data', inplace=True)

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
        x_pct (float): percentual mínimo de valorização
        dias_futuro (int): quantos dias à frente olhar

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
            houve_valorizacao = any(f > preco_base * (1 + x_pct) for f in futuros)
            target.append(1 if houve_valorizacao else 0)

    df['Target'] = target
    return df



# Função principal
def montar_dataset(papel, data_base, dias_total_historico=500, x_pct=0.03):
    """
    Monta o dataset final para o modelo, combinando preços, indicadores e variáveis externas.

    Parâmetros:
    - papel: string com o código do papel (ex: 'PETR4.SA')
    - data_base: string ou datetime (ex: '2024-12-01'); último dia com dados disponíveis
    - dias_total_historico (int): quantidade de dias úteis de histórico a carregar (padrão: 500 dias).
    """
    # Estima data inicial considerando apenas dias úteis (com margem de segurança)
    data_inicio = data_base - timedelta(days=int(dias_total_historico * 1.5))    
    
    df_preco = baixar_dados_preco(papel, data_inicio, data_base)
    df_preco = calcular_indicadores(df_preco)
    df_preco.index = pd.to_datetime(df_preco.index)


    # Carregar dados macro
    inicio_macro = (data_inicio - timedelta(days=365))
    fim_macro = data_base + timedelta(days=1)  # Para incluir o último dia
    df_macro = carregar_dados_macro(inicio_macro, fim_macro)            
    df_macro = df_macro[df_macro.index <= pd.to_datetime(data_base)]    

    time.sleep(2)
    df_ibov = baixar_dados_preco('^BVSP', data_inicio, data_base)

    # Resetar índices para merge
    df_preco = df_preco.reset_index().rename(columns={'Date': 'data'})
    df_ibov = df_ibov.reset_index().rename(columns={'Date': 'data'})
    df_macro = df_macro.reset_index().rename(columns={'index': 'data'})

    # Merge por data
    df_final = pd.merge(df_preco, df_macro, on='data', how='left')
    df_final = pd.merge(df_final, df_ibov, on='data', how='left', suffixes=('', '_ibov'))
    df_final = df_final.set_index('data').sort_index()

    # Calcula a variável Target
    df_final = calcular_target(df_final, x_pct=x_pct)

    return df_final


def preparar_X_y_agregado(df, dias_entrada, dropna=True):
    def inclinacao_trend(serie):
        y = serie.values.reshape(-1, 1)
        x = np.arange(len(serie)).reshape(-1, 1)
        if len(serie.dropna()) < len(serie) * 0.8:
            return np.nan
        modelo = LinearRegression().fit(x, y)
        return modelo.coef_[0][0]
    
    def retorno_desde_ultima_mudanca_sinal(series):
        returns = series.pct_change()
        sinais = np.sign(returns).ffill()
        mudancas = sinais.diff().fillna(0) != 0
        idx_ultima_mudanca = mudancas[::-1].idxmax()
        if idx_ultima_mudanca in series.index:
            valor_inicial = series.loc[idx_ultima_mudanca]
            valor_final = series.iloc[-1]
            return (valor_final / valor_inicial) - 1
        else:
            return np.nan

    X_rows, y = [], []

    for i in range(dias_entrada - 1, len(df)):
        janela = df.iloc[i - dias_entrada + 1:i + 1]
        data_inicial = janela.index[0]
        data_final = janela.index[-1]
        
        linha = {}

        if 'close' in janela.columns:
            linha['close_mean'] = janela['close'].mean()
            linha['close_std'] = janela['close'].std()
            linha['close_max'] = janela['close'].max()
            linha['close_min'] = janela['close'].min()
            linha['close_trend'] = inclinacao_trend(janela['close'])
            linha['retorno_desde_mudanca_close'] = retorno_desde_ultima_mudanca_sinal(janela['close'])
            linha['close_amplitude'] = janela['close'].max() - janela['close'].min()
            linha['close_dist_min'] = janela['close'].iloc[-1] - janela['close'].min()
            linha['close_ult_vs_mm'] = janela['close'].iloc[-1] - janela['close'].mean()

            

        if 'pct_change' in janela.columns:
            linha['retorno_mean'] = janela['pct_change'].mean()
            linha['retorno_std'] = janela['pct_change'].std()
            linha['retorno_acumulado'] = (janela['close'].iloc[-1] / janela['close'].iloc[0]) - 1
            linha['dias_alta_pct'] = (janela['pct_change'] > 0).mean()
            linha['dias_queda_pct'] = (janela['pct_change'] < 0).mean()
            pos_returns = janela['pct_change'][janela['pct_change'] > 0]
            neg_returns = janela['pct_change'][janela['pct_change'] < 0]
            linha['retorno_medio_pos'] = pos_returns.mean() if not pos_returns.empty else 0
            linha['retorno_medio_neg'] = neg_returns.mean() if not neg_returns.empty else 0

        if 'volume' in janela.columns:
            linha['volume_mean'] = janela['volume'].mean()
            linha['volume_std'] = janela['volume'].std()
            linha['volume_trend'] = inclinacao_trend(janela['volume'])

        for indicador in ['rsi', 'macd', 'macd_signal', 'macd_hist']:
            if indicador in janela.columns:
                linha[f'{indicador}_mean'] = janela[indicador].mean()
                linha[f'{indicador}_std'] = janela[indicador].std()
                linha[f'{indicador}_ultimo'] = janela[indicador].iloc[-1]

        # Indicadores de confiança
        if {'macd', 'macd_signal'}.issubset(janela.columns):
            macd_dist = (janela['macd'] - janela['macd_signal']).abs()
            linha['macd_conf_mean'] = macd_dist.mean()
            linha['macd_conf_ultimo'] = macd_dist.iloc[-1]

        if 'macd_hist' in janela.columns:
            linha['macd_hist_abs_mean'] = janela['macd_hist'].abs().mean()
            linha['macd_hist_abs_ultimo'] = abs(janela['macd_hist'].iloc[-1])

        if 'rsi' in janela.columns:
            linha['rsi_volatilidade'] = janela['rsi'].std()

        for macro in ['selic', 'ipca', 'dolar']:
            if macro in janela.columns:
                linha[f'{macro}_ultimo'] = janela[macro].iloc[-1]

        if 'dolar' in janela.columns:
            linha['dolar_trend'] = inclinacao_trend(janela['dolar'])
            linha['dolar_std'] = janela['dolar'].std()
            linha['retorno_desde_mudanca_dolar'] = retorno_desde_ultima_mudanca_sinal(janela['dolar'])

        if 'close_ibov' in janela.columns:
            linha['ibov_trend'] = inclinacao_trend(janela['close_ibov'])
            linha['ibov_std'] = janela['close_ibov'].std()
            linha['retorno_desde_mudanca_ibov'] = retorno_desde_ultima_mudanca_sinal(janela['close_ibov'])
            linha['ibov_amplitude'] = janela['close_ibov'].max() - janela['close_ibov'].min()


        alvo = df.iloc[i]['Target']
        X_rows.append(linha)
        y.append(alvo)

    X_df = pd.DataFrame(X_rows)
    y_series = pd.Series(y, index=X_df.index)

    if (dropna):
        X_df = X_df.dropna()
        y_series = y_series.loc[X_df.index]  # Garante alinhamento
            
    return X_df, y_series