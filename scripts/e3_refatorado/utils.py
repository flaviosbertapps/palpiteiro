import smtplib # Importado para a função de email
from email.mime.text import MIMEText # Importado
from email.mime.multipart import MIMEMultipart # Importado
import os
import mlflow #type: ignore

MEU_EMAIL_GMAIL = "flaviosbert.apps@gmail.com"
MINHA_SENHA_APP = "qshycrdzrmdqbgek"
EMAIL_PARA_RELATORIOS = "flaviobertaggia@gmail.com"

def enviar_email_gmail_com_app_password(assunto, corpo_texto=None, corpo_html=None):
    # ... (código da função de e-mail completo como na sua versão) ...
    if not corpo_texto and not corpo_html:
        print("ERRO: Pelo menos um corpo de mensagem (texto ou HTML) deve ser fornecido.")
        return False
    msg = MIMEMultipart('alternative')
    msg['From'] = MEU_EMAIL_GMAIL
    msg['To'] = EMAIL_PARA_RELATORIOS
    msg['Subject'] = assunto
    if corpo_texto:
        part_texto = MIMEText(corpo_texto, 'plain', 'utf-8')
        msg.attach(part_texto)
    if corpo_html:
        corpo_html_formatado = corpo_html.replace('\n', '<br>')
        part_html = MIMEText(corpo_html_formatado, 'html', 'utf-8')
        msg.attach(part_html)
    server = None
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(MEU_EMAIL_GMAIL, MINHA_SENHA_APP)
        print(f"Login no Gmail como {MEU_EMAIL_GMAIL} bem-sucedido.")
        server.sendmail(MEU_EMAIL_GMAIL, EMAIL_PARA_RELATORIOS, msg.as_string())
        print(f"E-mail enviado com sucesso para: {EMAIL_PARA_RELATORIOS}")
        return True
    except smtplib.SMTPAuthenticationError as e_auth:
        print(f"Erro de autenticação SMTP: {e_auth}")
        return False
    except smtplib.SMTPConnectError as e_conn:
        print(f"Erro de conexão SMTP: {e_conn}")
        return False
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao enviar o e-mail: {e}")
        return False
    finally:
        if server:
            server.quit()
            print("Conexão SMTP fechada.")




import sqlite3
from datetime import date
import pandas as pd # Usado no exemplo de uso

def extrair_float(texto):
    return float(texto.strip('%')) / 100 if isinstance(texto, str) else texto


def logar_sinal_compra_sqlite(
    caminho_bd: str,
    dados_sinal: dict
):
    """
    Loga os detalhes de um sinal de compra em um banco de dados SQLite.
    Cria a tabela 'sinais_compra' se ela não existir.

    Args:
        caminho_bd (str): O caminho completo para o arquivo do banco de dados SQLite (ex: 'meus_logs/trading_signals.db').
        dados_sinal (dict): Um dicionário contendo as informações do sinal.
                           Chaves esperadas: 'papel', 'probabilidade', 'threshold', 'decisao',
                           'data_medicao', 'fechamento', 'val_alvo', 'stop_loss', 'dias_alvo',
                           'prec_geral', 'prec_janelas', 'ultimo_atr_pct', etc.
    Returns:
        bool: True se o log foi bem-sucedido, False caso contrário.
    """
    conexao = None
    try:
        # 1. Conectar ao banco de dados (o arquivo será criado se não existir)
        conexao = sqlite3.connect(caminho_bd)
        cursor = conexao.cursor()

        # 2. Criar a tabela se ela não existir
        # Usamos tipos de dados SQLite: TEXT, REAL (para números de ponto flutuante), INTEGER, DATE
        # Adicionei a data de log (data_log) para saber quando o registro foi inserido.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sinais_compra (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_log DATE NOT NULL,
                data_medicao DATE,
                papel TEXT NOT NULL,
                decisao TEXT,
                probabilidade REAL,
                threshold REAL,
                fechamento REAL,
                val_alvo REAL,
                stop_loss REAL,
                dias_alvo INTEGER,                
                prec_janelas REAL,
                ultimo_atr_pct TEXT
            )
        """)

        # 3. Preparar os dados para inserção
        # Usar .get(chave, None) para evitar KeyError se alguma chave faltar no dicionário
        dados_para_inserir = (
            date.today(), # data_log
            dados_sinal.get('data_medicao', None),
            dados_sinal.get('papel', None),
            dados_sinal.get('decisao', None),
            extrair_float(dados_sinal.get('probabilidade', '0%')),
            extrair_float(dados_sinal.get('threshold', '0%')),
            dados_sinal.get('fechamento', None),
            extrair_float(dados_sinal.get('val_alvo', '0%')),
            extrair_float(dados_sinal.get('stop_loss', '0%')),
            int(dados_sinal.get('dias_alvo', None)) if dados_sinal.get('dias_alvo') is not None and str(dados_sinal.get('dias_alvo')).isdigit() else None,            
            extrair_float(dados_sinal.get('prec_janelas', '0%')),
            dados_sinal.get('ultimo_atr_pct', None)
        )

        # 4. Inserir um novo registro
        cursor.execute("""
            INSERT INTO sinais_compra (
                data_log, data_medicao, papel, decisao, probabilidade, threshold,
                fechamento, val_alvo, stop_loss, dias_alvo, prec_janelas, ultimo_atr_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, dados_para_inserir)

        # 5. Salvar (commit) as mudanças
        conexao.commit()
        print(f"INFO: Sinal para '{dados_sinal.get('papel')}' logado com sucesso no banco de dados '{caminho_bd}'.")
        return True

    except sqlite3.Error as e:
        print(f"ERRO SQLite: Não foi possível logar o sinal. Erro: {e}")
        # Opcional: fazer rollback em caso de erro
        if conexao:
            conexao.rollback()
        return False
    except (ValueError, TypeError) as e_conv:
        print(f"ERRO de conversão de tipo: Não foi possível preparar os dados para o SQLite. Erro: {e_conv}")
        print(f"Dados recebidos: {dados_sinal}")
        return False
    finally:
        # 6. Fechar a conexão
        if conexao:
            conexao.close()

# --- Exemplo de como você usaria esta função no seu script diário ---

if __name__ == '__main__':
    # Definir o caminho para o seu banco de dados
    DB_PATH = "logs/trading_signals_log.db"
    
    # Criar a pasta se não existir
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # Simular a lista 'dados_papeis_analisados_email' que seu script gera
    dados_papeis_analisados_exemplo = [
        {
            "papel": "PETR4",
            "probabilidade": "78.50%",
            "threshold": "75.00%",
            "decisao": "**COMPRAR**",
            "data_medicao": "2023-10-27",
            "prec_geral": "85.00%",
            "prec_janelas": "90.00%",
            "fechamento": "35.40",
            "val_alvo": "3.00%",
            "ultimo_atr_pct": "2.15",
            "stop_loss": "1.50%",
            "dias_alvo": 5
        },
        {
            "papel": "VALE3",
            "probabilidade": "65.20%",
            "threshold": "70.00%",
            "decisao": "Não Comprar",
            "data_medicao": "2023-10-27",
            "prec_geral": "80.00%",
            "prec_janelas": "82.00%",
            "fechamento": "68.90",
            "val_alvo": "2.50%",
            "ultimo_atr_pct": "1.98",
            "stop_loss": "1.25%",
            "dias_alvo": 7
        },
        {
            "papel": "ITUB4",
            "probabilidade": "92.10%",
            "threshold": "80.00%",
            "decisao": "**COMPRAR**",
            "data_medicao": "2023-10-27",
            "prec_geral": "95.00%",
            "prec_janelas": "98.00%",
            "fechamento": "28.15",
            "val_alvo": "2.00%",
            "ultimo_atr_pct": "1.55",
            "stop_loss": "1.00%",
            "dias_alvo": 5
        }
    ]

    print("--- Processando sinais e logando no SQLite ---")
    for sinal in dados_papeis_analisados_exemplo:
        # A lógica para decidir se loga ou não fica aqui.
        # Você quer logar apenas os sinais de COMPRA.
        if "COMPRAR" in sinal.get('decisao', ''):
            logar_sinal_compra_sqlite(DB_PATH, sinal)
        else:
            print(f"INFO: Sinal para '{sinal.get('papel')}' não é de compra. Não será logado.")

    print("\n--- Verificando o conteúdo do banco de dados ---")
    try:
        conn_verify = sqlite3.connect(DB_PATH)
        df_from_db = pd.read_sql_query("SELECT * FROM sinais_compra", conn_verify)
        conn_verify.close()
        
        print("Dados logados:")
        print(df_from_db)
    except Exception as e:
        print(f"Não foi possível ler o banco de dados para verificação. Erro: {e}")            






EXPERIMENT_NAME = "Modelo_B3"
CAMINHO_BASE_MLFLOW = r'C:\palpiteiro\mlruns'

def preparar_ambiente_mlflow():
    """
    Configura o ambiente MLflow para rastreamento de experimentos.
    Cria o diretório base se não existir e define o Tracking URI.
    Verifica se o experimento já existe, caso contrário, cria um novo.
    """
    # 1. Crie o diretório base se ele não existir
    os.makedirs(CAMINHO_BASE_MLFLOW, exist_ok=True)

    # 2. Processe o caminho ANTES de passá-lo para o f-string.
    caminho_processado = os.path.abspath(CAMINHO_BASE_MLFLOW).replace('\\', '/')

    # 3. Construa o Tracking URI.
    tracking_uri = f"file:///{caminho_processado}"

    # 4. Defina o Tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI definido para: {mlflow.get_tracking_uri()}")

    # 5. Verifique se o experimento existe. Se não, crie-o.
    #    A principal mudança é NÃO especificar o artifact_location na criação.
    #    Deixe o MLflow usar seu padrão, que geralmente é uma subpasta do tracking_uri.
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise AttributeError
        
        experiment_id = experiment.experiment_id
        print(f"Experimento '{EXPERIMENT_NAME}' já existe (ID: {experiment_id}).")
        print(f"  Artifact Location existente: {experiment.artifact_location}")

    except AttributeError:
        print(f"Experimento '{EXPERIMENT_NAME}' não encontrado. Criando agora...")
        # --- MUDANÇA CRUCIAL ---
        # NÃO passe artifact_location. Deixe o MLflow criar o padrão.
        # O padrão será um caminho relativo dentro do tracking_uri, que é o comportamento desejado.
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        
        # Vamos verificar o artifact_location que o MLflow criou por padrão
        novo_experimento = mlflow.get_experiment(experiment_id)
        print(f"Experimento '{EXPERIMENT_NAME}' criado com ID: {experiment_id}")
        print(f"  Artifact Location padrão criado pelo MLflow: {novo_experimento.artifact_location}")


    # 6. Defina o experimento que será usado para as próximas runs
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)