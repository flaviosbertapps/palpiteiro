import sys
import mlflow #type: ignore
import mlflow.pyfunc #type: ignore
import pandas as pd
import os
import re
from datetime import datetime, timedelta, date


import json
import traceback # Já estava lá, bom para logs

import preprocessamento as pp
import utils

MEU_EMAIL_GMAIL = "flaviosbert.apps@gmail.com"
MINHA_SENHA_APP = "qshycrdzrmdqbgek"
EMAIL_PARA_RELATORIOS = "flaviobertaggia@gmail.com"
PATH_LOGS = '../logs'
CAMINHO_BD = '/palpiteiro/dados/_bd_sinais.db'
CAMINHO_MLFLOW = 'file:/C:/palpiteiro/mlruns'

utils.preparar_ambiente_mlflow()
client = mlflow.tracking.MlflowClient()
alias_de_producao = "prd"


def obter_data_base():
    """
    Obtém a data base para a execução.
    Usa a data passada como argumento de linha de comando, ou a data atual se nenhum argumento for fornecido.
    """
    # sys.argv é uma lista de argumentos. sys.argv[0] é o nome do script.
    # Os argumentos do usuário começam em sys.argv[1].
    if len(sys.argv) > 1:
        data_str = sys.argv[1]
        try:
            # Tenta converter o argumento para um objeto de data
            data_base = datetime.strptime(data_str, "%Y-%m-%d").date()
            print(f"INFO: Executando previsão para a data fornecida como argumento: {data_base.strftime('%Y-%m-%d')}")
            return data_base
        except ValueError:
            print(f"ERRO: Formato de data inválido fornecido: '{data_str}'. Use YYYY-MM-DD.")
            print("INFO: Executando com a data atual como fallback.")
            return date.today()
    else:
        # Nenhum argumento de data fornecido, usa a data atual
        data_base_atual = date.today()
        print(f"INFO: Nenhum argumento de data fornecido. Executando para a data atual: {data_base_atual.strftime('%Y-%m-%d')}")
        return data_base_atual

# --- No início do seu fluxo principal ---
# Substitua a linha 'base = date.today()' pela chamada a esta função.
base = obter_data_base()

logs_completos_inferencia = []
dados_papeis_analisados_email = []

path_artefato_features = "model_config/features.txt"
path_artefato_df_precisoes = "model_config/precisoes.csv" # Mantenha .csv se você salva como CSV

data_execucao_script = date.today()
data_medicao_global = None

print(f"--- Iniciando script diário de previsão dinâmico ({data_execucao_script}) usando alias '{alias_de_producao}' ---")

registered_models = client.search_registered_models()

if not registered_models:
    print("Nenhum modelo registrado encontrado no MLflow Model Registry.")
    # Enviar e-mail se nenhum modelo for encontrado
    if MEU_EMAIL_GMAIL and MINHA_SENHA_APP and EMAIL_PARA_RELATORIOS:
        utils.enviar_email_gmail_com_app_password(            
            assunto=f"Relatório de Execução B3 ({data_execucao_script.strftime('%d/%m/%Y')}) - Sem Modelos",
            corpo_texto="Nenhum modelo foi encontrado no MLflow Model Registry para processamento."
        )
    exit() # Pode sair ou continuar, dependendo do comportamento desejado

print(f"Encontrados {len(registered_models)} modelos registrados. Verificando versões com alias '{alias_de_producao}'...")

# descartar modelos que não tem ALIAS de produção
modelos_producao = []
for rm in registered_models:    
    try:
        versao = client.get_model_version_by_alias(name=rm.name, alias=alias_de_producao)
        if versao:
            modelos_producao.append(versao)    
    except mlflow.exceptions.MlflowException as e:
        print(f"Erro ao obter a versao de producao do modelo {rm.name}: {e}")


for rm in modelos_producao:
    model_registry_name = rm.name
    # Inicializar variáveis para este loop de modelo/papel
    nome_papel = "N/A"
    version_info_with_alias = None
    run_data = None
    params_da_run = {}
    metrics_da_run = {}
    features_esperadas = []
    precisoes_dict = {}
    probabilidade_valorizacao_final = 0.0
    threshold_modelo_final = 0.75 # Valor padrão, pode ser sobrescrito
    decisao_final_str = "ERRO AO PROCESSAR"
    data_previsao_papel = "N/A"
    precision_val_geral_run = "N/A"
    precision_val_janelas_run = "N/A"
    preco_fechamento_dia = "N/A"        # <<< NOVO: Inicialização
    valorizacao_alvo_modelo = "N/A"   # <<< NOVO: Inicialização
    stop_loss_modelo = "N/A"
    dias_horizonte_modelo = "N/A"

    log_lines_papel_atual = [f"--- LOG DE INFERÊNCIA PARA MODELO: {model_registry_name} ---"]
    timestamp_inicio_papel = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines_papel_atual.append(f"[TIMESTAMP_INICIO_PROC]: {timestamp_inicio_papel}")
    log_lines_papel_atual.append(f"[INFO] Verificando modelo registrado: '{model_registry_name}'")

    try:
        model_uri_with_alias = f"models:/{model_registry_name}@{alias_de_producao}"
        log_lines_papel_atual.append(f"[INFO] Tentando carregar modelo de: {model_uri_with_alias}")

        try:
            version_info_with_alias = client.get_model_version_by_alias(name=model_registry_name, alias=alias_de_producao)
            log_lines_papel_atual.append(f"[INFO] Alias '{alias_de_producao}' encontrado. Versão {version_info_with_alias.version} (Run ID: {version_info_with_alias.run_id}).")
        except mlflow.exceptions.MlflowException as e_alias:
            msg_erro_alias = f"Alias '{alias_de_producao}' não encontrado para o modelo '{model_registry_name}'. Pulando. Detalhe: {e_alias}"
            log_lines_papel_atual.append(f"[ERRO] {msg_erro_alias}")
            print(msg_erro_alias)
            dados_papeis_analisados_email.append({ # Adicionar informações de erro ao e-mail
                "papel": model_registry_name.split('_')[1] if '_' in model_registry_name else model_registry_name, # Tentativa
                "probabilidade": "N/A", "threshold": "N/A", "decisao": "ERRO (Alias)",
                "data_medicao": "N/A", "prec_geral": "N/A", "prec_janelas": "N/A",
                "fechamento": "N/A", "val_alvo": "N/A" ,  "ultimo_atr_pct": "N/A"
                , 'stop_loss': 'N/A', 'dias_alvo': 'N/A'
            })
            logs_completos_inferencia.append("\n".join(log_lines_papel_atual))
            continue

        model = mlflow.sklearn.load_model(model_uri_with_alias)
        log_lines_papel_atual.append(f"[INFO] Modelo '{model_registry_name}' (versão {version_info_with_alias.version}) carregado.")

        run_id_da_versao_alias = version_info_with_alias.run_id
        try:
            run_data = client.get_run(run_id_da_versao_alias)
            log_lines_papel_atual.append(f"[INFO] Dados da Run {run_id_da_versao_alias} carregados.")
        except mlflow.exceptions.MlflowException as e_run:
            log_lines_papel_atual.append(f"[ERRO] Ao carregar dados da Run {run_id_da_versao_alias}: {e_run}.")

        if run_data:
            params_da_run = run_data.data.params
            metrics_da_run = run_data.data.metrics
            log_lines_papel_atual.append("[INFO] Parâmetros da Run Original do Modelo:")
            for key, value in params_da_run.items(): log_lines_papel_atual.append(f"[INFO]     {key}: {value}")
            log_lines_papel_atual.append("[INFO] Métricas da Run Original do Modelo:")
            for key, value in metrics_da_run.items(): log_lines_papel_atual.append(f"[INFO]     {key}: {value:.4f}")

            nome_papel = params_da_run.get("papel", "N/A")
            dias_janela_x_param = params_da_run.get("dias_janela_x")
            dias_janela_x = int(dias_janela_x_param) if dias_janela_x_param else 60 # Valor padrão para dias_janela
            threshold_param = params_da_run.get("threshold")
            if threshold_param: threshold_modelo_final = float(threshold_param)
            
            val_alvo_param = params_da_run.get("valorizacao") # Use o nome EXATO do parâmetro
            if val_alvo_param is not None:
                try:
                    valorizacao_alvo_modelo = float(val_alvo_param)
                except ValueError:
                    valorizacao_alvo_modelo = str(val_alvo_param)
            log_lines_papel_atual.append(f"[INFO] Parâmetro 'valorizacao' (alvo do modelo): {valorizacao_alvo_modelo}")

            dias_futuro_valorizacao_param = params_da_run.get("dias_futuro_valorizacao")
            if dias_futuro_valorizacao_param is not None:
                dias_horizonte_modelo = int(dias_futuro_valorizacao_param)
        
            stop_loss_param = params_da_run.get("stop_loss") # << Use o nome EXATO do parâmetro
            if stop_loss_param is not None:
                try:
                    # Formata como porcentagem para exibição
                    stop_loss_modelo = f"{float(stop_loss_param):.2%}"
                except ValueError:
                    stop_loss_modelo = str(stop_loss_param)


            dias_futuro_valorizacao_param = params_da_run.get("dias_futuro_valorizacao") # Use o nome EXATO do parâmetro
            if dias_futuro_valorizacao_param is not None:
                try:
                    dias_futuro_valorizacao_modelo = int(dias_futuro_valorizacao_param)
                except ValueError:
                    dias_futuro_valorizacao_modelo = str(dias_futuro_valorizacao_param)
            log_lines_papel_atual.append(f"[INFO] Parâmetro 'dias_futuro_valorizacao': {dias_futuro_valorizacao_modelo}")


            pv_geral = metrics_da_run.get("Precision_Val_Geral")
            if pv_geral is not None: precision_val_geral_run = f"{pv_geral:.2%}"
            pv_janelas = metrics_da_run.get("Precision_Val_Janelas")
            if pv_janelas is not None: precision_val_janelas_run = f"{pv_janelas:.2%}"

            try:
                local_path_df_precisoes = client.download_artifacts(run_id=run_id_da_versao_alias, path=path_artefato_df_precisoes)
                df_precisoes_carregado = pd.read_csv(local_path_df_precisoes) # Assume CSV
                log_lines_papel_atual.append(f"[INFO] DataFrame de precisões '{path_artefato_df_precisoes}' carregado.")
                if 'faixa_probabilidade' in df_precisoes_carregado.columns and 'precisao_empirica' in df_precisoes_carregado.columns:
                    for _, row in df_precisoes_carregado.iterrows():
                        precisoes_dict[str(row['faixa_probabilidade'])] = row['precisao_empirica']
                    log_lines_papel_atual.append("[INFO] Dicionário de precisões reconstruído.")
                else:
                    log_lines_papel_atual.append("[AVISO] Colunas esperadas não encontradas no DF de precisões.")
            except Exception as e_prec:
                log_lines_papel_atual.append(f"[AVISO] Falha ao processar artefato de precisões: {e_prec}")
                precisoes_dict = {}
        else: # Se run_data não pôde ser carregado
            log_lines_papel_atual.append("[AVISO] Não foi possível carregar dados da run. Tentando extrair nome do papel do registro.")
            if '_' in model_registry_name: # Fallback para nome do papel
                partes_nome_modelo = model_registry_name.split('_')
                if len(partes_nome_modelo) > 1: nome_papel = partes_nome_modelo[1]

        if nome_papel == "N/A":
            log_lines_papel_atual.append("[ERRO] Nome do papel não pôde ser determinado. Pulando.")
            dados_papeis_analisados_email.append({ # Adicionar informações de erro ao e-mail
                "papel": model_registry_name, "probabilidade": "N/A", "threshold": "N/A",
                "decisao": "ERRO (Nome Papel N/A)", "data_medicao": "N/A",
                "prec_geral": "N/A", "prec_janelas": "N/A",
                "fechamento": "N/A", "val_alvo": valorizacao_alvo_modelo,  "ultimo_atr_pct": "N/A"
                , 'stop_loss': 'N/A', 'dias_alvo': 'N/A'
            })
            logs_completos_inferencia.append("\n".join(log_lines_papel_atual))
            continue

        log_lines_papel_atual.append(f"[INFO] Papel Identificado para o Modelo: {nome_papel}")
        log_lines_papel_atual.append(f"[INFO] Threshold de Compra do Modelo: {threshold_modelo_final:.4f}")

        try:
            local_path_features_file = client.download_artifacts(run_id=run_id_da_versao_alias, path=path_artefato_features)
            with open(local_path_features_file, "r") as f:
                features_esperadas = [line.strip() for line in f if line.strip()]
            log_lines_papel_atual.append(f"[INFO] Features esperadas ({len(features_esperadas)}): {', '.join(features_esperadas)}")
        except Exception as e_feat:
            log_lines_papel_atual.append(f"[ERRO] Erro ao processar features: {e_feat}. Pulando.")
            dados_papeis_analisados_email.append({ # Adicionar informações de erro ao e-mail
                "papel": nome_papel, "probabilidade": "N/A", "threshold": f"{threshold_modelo_final:.2%}",
                "decisao": "ERRO (Features)", "data_medicao": "N/A",
                "prec_geral": precision_val_geral_run, "prec_janelas": precision_val_janelas_run,
                "fechamento": "N/A", "val_alvo": valorizacao_alvo_modelo, "ultimo_atr_pct": "N/A"
                , 'stop_loss': 'N/A', 'dias_alvo': 'N/A'
            })
            logs_completos_inferencia.append("\n".join(log_lines_papel_atual))
            continue


        ##### PREPARAÇÃO DOS DADOS PARA INFERÊNCIA #####

        log_lines_papel_atual.append(f"[INFO] Montando dataset para {nome_papel}...")
        df_base = pp.montar_dataset(nome_papel, dias_total_historico=100, data_base=base)
        log_lines_papel_atual.append(f"[DEBUG] Tail(3) do df montado:\n{df_base.tail(3).to_string()}")

        if df_base.empty or 'close' not in df_base.columns:
            msg_erro_df = "DataFrame de dados históricos vazio" if df_base.empty else "Coluna 'close' não encontrada no DataFrame"
            log_lines_papel_atual.append(f"[ERRO] {msg_erro_df} para {nome_papel}. Pulando.")
            dados_papeis_analisados_email.append({ # Adicionar informações de erro ao e-mail
                "papel": nome_papel, "probabilidade": "N/A", "threshold": f"{threshold_modelo_final:.2%}",
                "decisao": "ERRO (Dados Hist.)", "data_medicao": "N/A",
                "prec_geral": precision_val_geral_run, "prec_janelas": precision_val_janelas_run,
                "fechamento": "N/A", "val_alvo": valorizacao_alvo_modelo, "ultimo_atr_pct": ultimo_atr_pct_dia
                , 'stop_loss': 'N/A', 'dias_alvo': 'N/A'
            })
            logs_completos_inferencia.append("\n".join(log_lines_papel_atual))
            continue

        data_previsao_papel = df_base.index[-1].strftime('%Y-%m-%d')
        preco_fechamento_dia = f"{df_base['close'].iloc[-1]:.2f}"
        ultimo_atr_pct_dia = f"{100*df_base['atr_pct'].iloc[-1]:.2f}"
        if data_medicao_global is None: data_medicao_global = data_previsao_papel
        log_lines_papel_atual.append(f"[INFO] Data da última medição para {nome_papel}: {data_previsao_papel}, Fechamento: {preco_fechamento_dia}")

        print(dias_futuro_valorizacao_modelo)
        print(type(dias_futuro_valorizacao_modelo))

        df = pp.calcular_target(df_base
                                 , x_pct = valorizacao_alvo_modelo
                                 ,dias_futuro=int(dias_futuro_valorizacao_modelo))

        log_lines_papel_atual.append(f"[INFO] Preparando X e y com dias_janela={dias_janela_x}...")
        X, y = pp.preparar_X_y_agregado(df, dias_janela_x, dropna=False)

        if X.empty:
            log_lines_papel_atual.append(f"[ERRO] DataFrame X vazio após preparação para {nome_papel}. Pulando.")
            dados_papeis_analisados_email.append({ # Adicionar informações de erro ao e-mail
                "papel": nome_papel, "probabilidade": "N/A", "threshold": f"{threshold_modelo_final:.2%}",
                "decisao": "ERRO (Prep. X)", "data_medicao": data_previsao_papel,
                "prec_geral": precision_val_geral_run, "prec_janelas": precision_val_janelas_run,
                "fechamento": preco_fechamento_dia, "val_alvo": valorizacao_alvo_modelo, "ultimo_atr_pct": ultimo_atr_pct_dia
                , 'stop_loss': 'N/A', 'dias_alvo': 'N/A'
            })
            logs_completos_inferencia.append("\n".join(log_lines_papel_atual))
            continue

        if not all(feature in X.columns for feature in features_esperadas):
            missing_features = [f for f in features_esperadas if f not in X.columns]
            log_lines_papel_atual.append(f"[ERRO] Features esperadas não encontradas em X para {nome_papel}: {missing_features}. Pulando.")
            dados_papeis_analisados_email.append({ # Adicionar informações de erro ao e-mail
                "papel": nome_papel, "probabilidade": "N/A", "threshold": f"{threshold_modelo_final:.2%}",
                "decisao": f"ERRO (Faltam features: {', '.join(missing_features)})", "data_medicao": data_previsao_papel,
                "prec_geral": precision_val_geral_run, "prec_janelas": precision_val_janelas_run,
                "fechamento": preco_fechamento_dia, "val_alvo": valorizacao_alvo_modelo , "ultimo_atr_pct": ultimo_atr_pct_dia
                , 'stop_loss': 'N/A', 'dias_alvo': 'N/A'
            })
            logs_completos_inferencia.append("\n".join(log_lines_papel_atual))
            continue

        X_filtrado = X[features_esperadas]
        if list(X_filtrado.columns) != features_esperadas: # Verificação de ordem (opcional, mas bom)
            log_lines_papel_atual.append("[AVISO] Ordem das colunas em X_filtrado não bate com features_esperadas. Reordenando.")
            X_filtrado = X_filtrado[features_esperadas] # Garante a ordem

        #X_para_previsao = X_filtrado.loc[[X_filtrado.index[-1]]]
        X_para_previsao = X_filtrado.loc[base]
        log_lines_papel_atual.append(f"[INFO] Último registro de X preparado para previsão (shape {X_para_previsao.shape}):\n{X_para_previsao.to_string()}")

        if hasattr(model, 'predict_proba'):
            log_lines_papel_atual.append("[INFO] Modelo possui 'predict_proba'. Executando...")
            proba_array = model.predict_proba(X_para_previsao.values.reshape(1, -1))  # Reshape para garantir que seja 2D
            log_lines_papel_atual.append(f"[INFO] Array de Probabilidades retornado: {proba_array}")

            classes_do_modelo = model.classes_
            log_lines_papel_atual.append(f"[INFO] Classes do modelo: {classes_do_modelo}")
            try:
                indice_classe_1 = list(classes_do_modelo).index(1)
                probabilidade_valorizacao_final = proba_array[0][indice_classe_1]
                log_lines_papel_atual.append(f"[INFO] Probabilidade da Classe 1 (índice {indice_classe_1}): {probabilidade_valorizacao_final:.4f}")

                if probabilidade_valorizacao_final >= threshold_modelo_final:
                    decisao_final_str = "**COMPRAR**"
                else:
                    decisao_final_str = "Não Comprar"
                log_lines_papel_atual.append(f"[RESULTADO] Decisão: {decisao_final_str.replace('*','')} para {nome_papel} (Prob: {probabilidade_valorizacao_final:.4f}, Threshold: {threshold_modelo_final:.4f})")

            except (ValueError, IndexError) as e_proba:
                log_lines_papel_atual.append(f"[ERRO] Ao processar probabilidades para {nome_papel}: {e_proba}")
                decisao_final_str = "ERRO (Probabilidades)"
        else:
            log_lines_papel_atual.append(f"[AVISO] Modelo para {nome_papel} não possui 'predict_proba'.")
            decisao_final_str = "N/A (Sem predict_proba)"

        dados_do_papel = {
            "papel": nome_papel,
            "probabilidade": f"{probabilidade_valorizacao_final:.2%}" if "ERRO" not in decisao_final_str and "N/A" not in decisao_final_str else "N/A",
            "threshold": f"{threshold_modelo_final:.2%}",
            "decisao": decisao_final_str,
            "data_medicao": data_previsao_papel,
            "prec_geral": precision_val_geral_run,
            "prec_janelas": precision_val_janelas_run,
            "fechamento": preco_fechamento_dia,   
            "val_alvo": f"{valorizacao_alvo_modelo:.2%}" if isinstance(valorizacao_alvo_modelo, float) else valorizacao_alvo_modelo, # Formatar como %
            "ultimo_atr_pct": ultimo_atr_pct_dia,
            "stop_loss": stop_loss_modelo, 
            "dias_alvo": dias_horizonte_modelo 
        }

        # Adicionar dados para o e-mail (agora com todas as informações)
        dados_papeis_analisados_email.append(dados_do_papel)   

        # LOGA NO SQLITE SOMENTE SE FOR COMPRA
        if "COMPRAR" in dados_do_papel.get('decisao', ''):
            sucesso_log = utils.logar_sinal_compra_sqlite(CAMINHO_BD, dados_do_papel)
            if not sucesso_log:
                # Adicione aqui alguma lógica se o log falhar, se necessário
                print(f"AVISO: Falha ao logar o sinal de compra para {nome_papel} no banco de dados.")                

    except mlflow.exceptions.MlflowException as e_mlflow_outer:
        msg_erro_mlflow_outer = f"Erro MLflow geral ao processar modelo '{model_registry_name}': {e_mlflow_outer}"
        log_lines_papel_atual.append(f"[ERRO_FATAL] {msg_erro_mlflow_outer}")
        print(msg_erro_mlflow_outer)
        dados_papeis_analisados_email.append({ # Adicionar informações de erro ao e-mail
            "papel": nome_papel if nome_papel != "N/A" else model_registry_name, "probabilidade": "N/A",
            "threshold": "N/A", "decisao": "ERRO FATAL MLFLOW", "data_medicao": "N/A",
            "prec_geral": "N/A", "prec_janelas": "N/A", "fechamento": "N/A", "val_alvo": "N/A", "ultimo_atr_pct": "N/A"
        })
    except Exception as e_general_outer:
        msg_erro_geral_outer = f"Erro inesperado geral ao processar modelo '{model_registry_name}': {e_general_outer}"
        log_lines_papel_atual.append(f"[ERRO_FATAL] {msg_erro_geral_outer}")
        log_lines_papel_atual.append(f"[STACKTRACE] {traceback.format_exc()}")
        print(msg_erro_geral_outer)
        dados_papeis_analisados_email.append({ # Adicionar informações de erro ao e-mail
            "papel": nome_papel if nome_papel != "N/A" else model_registry_name, "probabilidade": "N/A",
            "threshold": "N/A", "decisao": "ERRO FATAL GERAL", "data_medicao": "N/A",
            "prec_geral": "N/A", "prec_janelas": "N/A", "fechamento": "N/A", "val_alvo": "N/A", "ultimo_atr_pct": "N/A"
            , 'stop_loss': 'N/A', 'dias_alvo': 'N/A'
        })
    finally:
        timestamp_fim_papel = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_lines_papel_atual.append(f"[TIMESTAMP_FIM_PROC]: {timestamp_fim_papel}")
        log_lines_papel_atual.append("--- FIM DO LOG DE INFERÊNCIA PARA ESTE MODELO/PAPEL ---")
        logs_completos_inferencia.append("\n".join(log_lines_papel_atual))


# --- Fim do Loop de Modelos ---

# Processar os logs
string_log_consolidado = "\n\n".join(logs_completos_inferencia)
# print(string_log_consolidado) # Opcional imprimir aqui
log_filename = f"{PATH_LOGS}\log_inferencias_{data_execucao_script.strftime('%Y%m%d')}.txt"
with open(log_filename, "w", encoding='utf-8') as f_log:
   f_log.write(string_log_consolidado)
print(f"Log completo salvo em: {log_filename}")


# --- Preparar e Enviar E-mail de Resumo ---

assunto_email_resumo = f"Relatório Diário de Análise de Papéis B3 - {data_execucao_script.strftime('%d/%m/%Y')}"
data_medicao_display = data_medicao_global if data_medicao_global else "N/A (nenhum papel processado com sucesso)"

# Corpo em Texto Puro (Com Novas Colunas)
corpo_txt_email = [f"Relatório de Análise de Papéis - Execução: {data_execucao_script.strftime('%d/%m/%Y')}"]
corpo_txt_email.append(f"Data da Medição de Referência (última cotação usada): {data_medicao_display}\n")
corpo_txt_email.append("Resumo das Análises:")
# Ajustar a largura e os cabeçalhos
header = (f"{'Papel':<8} | {'Fechamento':<10} | {'Probab.':<10} | {'Decisão':<15} | "
          f"{'Alvo':<10} | {'Stop':<10} | {'Prazo(d)':<8} | {'Threshold':<10} | "
           f"{'Prec.Janelas':<12}")
corpo_txt_email.append("-" * len(header))
corpo_txt_email.append(header)
corpo_txt_email.append("-" * len(header))
for item in dados_papeis_analisados_email:
    corpo_txt_email.append(
        f"{item.get('papel', 'N/A'):<8} | "
        f"{item.get('fechamento', 'N/A'):<10} | "
        f"{item.get('probabilidade', 'N/A'):<10} | "
        f"{item.get('decisao', 'N/A').replace('*',''):<15} | "
        f"{item.get('val_alvo', 'N/A'):<10} | "         # <<< NOVO
        f"{item.get('stop_loss', 'N/A'):<10} | "      # <<< NOVO
        f"{item.get('dias_alvo', 'N/A'):<8} | "       # <<< NOVO
        f"{item.get('threshold', 'N/A'):<10} | "        
        f"{item.get('prec_janelas', 'N/A'):<12}"
    )
corpo_txt_email.append("-" * len(header))
corpo_texto_final = "\n".join(corpo_txt_email)

# Corpo em HTML (Com Novas Colunas e Estilo de Linha Alternada)
html_rows = ""
# <<< NOVO: Adiciona um contador para as linhas da tabela >>>
for i, item in enumerate(dados_papeis_analisados_email):
    estilo_decisao = "color: green; font-weight: bold;" if "COMPRAR" in item.get('decisao', '') else \
                     ("color: red; font-weight: bold;" if "ERRO" in item.get('decisao', '') else "")
    
    # <<< NOVO: Define a cor de fundo com base se a linha é par ou ímpar >>>
    background_color = "#f9f9f9" if i % 2 == 0 else "#ffffff" # Cinza claro para linhas pares, branco para ímpares

    html_rows += f"""
    <tr style="background-color: {background_color};">
        <td style="border: 1px solid #ddd; padding: 8px;">{item.get('papel', 'N/A')}</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{item.get('fechamento', 'N/A')}</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{item.get('probabilidade', 'N/A')}</td>
        <td style="border: 1px solid #ddd; padding: 8px; {estilo_decisao}">{item.get('decisao', 'N/A')}</td>
        <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{item.get('val_alvo', 'N/A')}</td>        <!-- NOVO -->
        <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{item.get('stop_loss', 'N/A')}</td>      <!-- NOVO -->
        <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{item.get('dias_alvo', 'N/A')}</td>      <!-- NOVO -->
        <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{item.get('threshold', 'N/A')}</td>        
        <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{item.get('prec_janelas', 'N/A')}</td>
        <td style="border: 1px solid #ddd; padding: 8px;">{item.get('data_medicao', 'N/A')}</td>
    </tr>"""

corpo_html_final = f"""
<html>
    <head>
    <style>
        table {{ font-family: Arial, sans-serif; border-collapse: collapse; width: 100%; font-size: 0.9em; }}
        th, td {{ border: 1px solid #dddddd; text-align: left; padding: 6px; }}
        th {{ background-color: #f2f2f2; text-align: center; }}
        /* O estilo de linha alternada agora é aplicado diretamente no <tr>, mas podemos manter um default */
        tr:nth-child(even) {{ background-color: #f9f9f9; }} 
    </style>
    </head>
    <body>
    <h2>Relatório Diário de Análise de Papéis B3</h2>
    <p><strong>Data da Execução do Script:</strong> {data_execucao_script.strftime('%d/%m/%Y')}</p>
    <p><strong>Data da Medição de Referência (última cotação usada):</strong> {data_medicao_display}</p>
    <h3>Resumo das Análises:</h3>
    <table>
        <thead>
        <tr>
            <th>Papel</th>
            <th>Fechamento</th>
            <th>Probabilidade</th>
            <th>Decisão</th>
            <th>Alvo</th>         <!-- NOVO -->
            <th>Stop Loss</th>    <!-- NOVO -->
            <th>Prazo (d)</th>    <!-- NOVO -->
            <th>Threshold</th>            
            <th>Prec. Janelas</th>
            <th>Data Avaliação</th>
        </tr>
        </thead>
        <tbody>
        {html_rows}
        </tbody>
    </table>
    <hr>
    <p><em>Este é um e-mail automático. Para logs detalhados, consulte o arquivo: {log_filename}</em></p>
    </body>
</html>
"""

print(f"\nPreparando para enviar e-mail de resumo para {EMAIL_PARA_RELATORIOS}...")
sucesso_envio_resumo = utils.enviar_email_gmail_com_app_password(    
    assunto=assunto_email_resumo,
    corpo_texto=corpo_texto_final,
    corpo_html=corpo_html_final
)
if sucesso_envio_resumo:
    print("E-mail de resumo diário enviado com sucesso.")
else:
    print("Falha ao enviar e-mail de resumo diário.")


print("--- Fim do script diário de previsão dinâmico ---")

