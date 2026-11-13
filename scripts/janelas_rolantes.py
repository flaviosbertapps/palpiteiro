import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def avaliar_conjunto_de_features_walk_forward_rolante(
        X_treino_subset: pd.DataFrame,
        y_treino: pd.Series,
        modelo_template,
        tamanho_janela_treino: int = 126, # Aprox. 6 meses (21 * 6)
        tamanho_janela_validacao: int = 21  # Aprox. 1 mês
    ) -> dict:
    """
    Função auxiliar que avalia um ÚNICO conjunto de features usando
    walk-forward com JANELA ROLANTE (tamanho de treino fixo).
    Retorna a média das métricas.
    """
    # O passo de deslizamento será igual ao tamanho da janela de validação
    # para evitar sobreposição nos conjuntos de teste.
    passo_wf = tamanho_janela_validacao

    f1s, precisions, recalls = [], [], []
    num_janelas_executadas = 0

    X_wf_array = X_treino_subset.values
    y_wf_array = y_treino.values
    n_total_amostras = len(X_wf_array)
    
    # Loop de walk-forward com janela rolante
    for inicio_treino_idx in range(0, n_total_amostras, passo_wf):
        fim_treino_idx = inicio_treino_idx + tamanho_janela_treino
        inicio_validacao_idx = fim_treino_idx
        fim_validacao_idx = inicio_validacao_idx + tamanho_janela_validacao

        if fim_validacao_idx > n_total_amostras:
            break # Não há mais dados suficientes para uma janela completa
            
        num_janelas_executadas += 1

        X_treino_fold = X_wf_array[inicio_treino_idx : fim_treino_idx]
        y_treino_fold = y_wf_array[inicio_treino_idx : fim_treino_idx]
        X_teste_fold = X_wf_array[inicio_validacao_idx : fim_validacao_idx]
        y_teste_fold = y_wf_array[inicio_validacao_idx : fim_validacao_idx]
        
        # Pular se a janela de treino for muito pequena (acontece no início se o total de dados for menor que o tamanho da janela)
        # ou se não houver exemplos positivos para treinar
        if len(X_treino_fold) < tamanho_janela_treino or np.sum(y_treino_fold) == 0:
            continue
        
        modelo = clone(modelo_template)
        try:
            modelo.fit(X_treino_fold, y_treino_fold)
            y_pred = modelo.predict(X_teste_fold)
            
            f1s.append(f1_score(y_teste_fold, y_pred, zero_division=0))
            precisions.append(precision_score(y_teste_fold, y_pred, zero_division=0))
            recalls.append(recall_score(y_teste_fold, y_pred, zero_division=0))
        except Exception as e:
            # print(f"  Aviso: Erro no fit/predict do fold {num_janelas_executadas}: {e}")
            f1s.append(0); precisions.append(0); recalls.append(0)

    if not f1s: return {'f1_medio': 0, 'precisao_media': 0, 'recall_medio': 0}
    
    return {
        'f1_medio': np.mean(f1s),
        'precisao_media': np.mean(precisions),
        'recall_medio': np.mean(recalls)
    }


def encontrar_melhor_conjunto_features_incremental(
        X_super_treino, y_super_treino,
        grupos_de_features_candidatas,
        modelo_template,
        tamanho_janela_treino_wf: int = 126, # Passando os novos parâmetros
        tamanho_janela_validacao_wf: int = 21
    ):
    """
    Testa a adição de grupos de features um a um, usando a validação
    walk-forward com janela rolante.
    """
    print("\n--- Iniciando Busca Incremental pelo Melhor Conjunto de Features (Janela Rolante) ---")
    
    features_campeas = []
    melhor_score_ate_agora = -1
    resultados_busca = []
    
    grupos_para_testar = {'baseline': []}
    grupos_para_testar.update(grupos_de_features_candidatas)
    
    for nome_grupo, features_do_grupo in grupos_para_testar.items():
        print(f"\n>>> Testando adição do grupo: '{nome_grupo}'")
        
        features_atuais_teste = sorted(list(set(features_campeas + features_do_grupo)))
        features_atuais_teste = [f for f in features_atuais_teste if f in X_super_treino.columns]
        
        if not features_atuais_teste and nome_grupo != 'baseline':
             print(f"  Aviso: Nenhuma feature do grupo '{nome_grupo}' encontrada. Pulando.")
             continue

        X_subset = X_super_treino[features_atuais_teste] if features_atuais_teste else X_super_treino.iloc[:, :0]

        # Chamar a nova função de avaliação com janela rolante
        metricas_atuais = avaliar_conjunto_de_features_walk_forward_rolante(
            X_subset, y_super_treino, modelo_template,
            tamanho_janela_treino=tamanho_janela_treino_wf,
            tamanho_janela_validacao=tamanho_janela_validacao_wf
        )
        
        score_atual = metricas_atuais['f1_medio']
        print(f"  Conjunto com {len(features_atuais_teste)} features. F1-médio (WF Rolante): {score_atual:.4f}")

        if score_atual > melhor_score_ate_agora:
            print(f"  MELHORA! Score atual ({score_atual:.4f}) > Melhor anterior ({melhor_score_ate_agora:.4f}). Mantendo o grupo.")
            melhor_score_ate_agora = score_atual
            features_campeas = features_atuais_teste
            status = "Mantido"
        else:
            print(f"  Sem melhora. Score atual ({score_atual:.4f}) <= Melhor anterior ({melhor_score_ate_agora:.4f}). Descartando o grupo.")
            status = "Descartado"
            
        resultados_busca.append({
            'grupo_testado': nome_grupo,
            'f1_medio_com_grupo': score_atual,
            'status': status,
            'num_features_acumulado': len(features_campeas)
        })

    print("\n--- Conclusão da Busca Incremental ---")
    print(f"Melhor F1-score médio (WF Rolante) encontrado: {melhor_score_ate_agora:.4f}")
    print(f"Número de features no conjunto campeão: {len(features_campeas)}")
    print("Features Campeãs:")
    print(features_campeas)

    return features_campeas, pd.DataFrame(resultados_busca)

def gerador_de_janela_rolante_cv(n_amostras, tamanho_janela_treino, tamanho_janela_validacao, passo_wf):
    """
    (Gerador Auxiliar)
    Cria um gerador de índices de treino/validação para validação cruzada
    com janela rolante de tamanho fixo.
    """
    inicio_janela_treino = 0
    while True:
        fim_janela_treino = inicio_janela_treino + tamanho_janela_treino
        inicio_janela_validacao = fim_janela_treino
        fim_janela_validacao = inicio_janela_validacao + tamanho_janela_validacao
        
        if fim_janela_validacao > n_amostras:
            break
            
        indices_treino = np.arange(inicio_janela_treino, fim_janela_treino)
        indices_validacao = np.arange(inicio_janela_validacao, fim_janela_validacao)
        
        yield indices_treino, indices_validacao
        
        inicio_janela_treino += passo_wf


def otimizar_hiperparametros_logreg_com_janela_rolante(
        X_super_treino, y_super_treino,
        tamanho_janela_treino_wf: int = 126, # Aprox. 6 meses
        tamanho_janela_validacao_wf: int = 21,  # Aprox. 1 mês
        passo_wf: int = 21, # Desliza 1 mês
        pipeline_template=None
    ):
    """
    Encontra os melhores hiperparâmetros para a Regressão Logística usando
    GridSearchCV com uma validação cruzada de JANELA ROLANTE personalizada.
    """
    print(f"\n--- Iniciando Otimização de Hiperparâmetros com Janela Rolante ---")
    print(f"Janela de Treino: {tamanho_janela_treino_wf}, Janela de Validação: {tamanho_janela_validacao_wf}, Passo: {passo_wf}")
    
    # 1. Criar o gerador de validação cruzada personalizado
    cv_janela_rolante = gerador_de_janela_rolante_cv(
        n_amostras=len(X_super_treino),
        tamanho_janela_treino=tamanho_janela_treino_wf,
        tamanho_janela_validacao=tamanho_janela_validacao_wf,
        passo_wf=passo_wf
    )

    # 2. Definir o modelo base (template)
    if pipeline_template is None:
        # É altamente recomendado usar um pipeline para escalar os dados em cada fold
        pipeline_template = Pipeline([
            ('scaler', StandardScaler()),
            ('modelo', LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'))
        ])
    
    # 3. Definir o "grid" de parâmetros
    param_grid = [
        {
            'modelo__penalty': ['l1'],
            'modelo__solver': ['liblinear', 'saga'],
            'modelo__C': np.logspace(-3, 1, 5) # Ex: [0.001, 0.01, 0.1, 1, 10]
        },
        {
            'modelo__penalty': ['l2'],
            'modelo__solver': ['liblinear', 'lbfgs', 'saga'],
            'modelo__C': np.logspace(-3, 1, 5)
        }
    ]

    # 4. Definir a métrica de scoring
    scorer = make_scorer(f1_score) # Mantendo F1-score como no seu código

    # 5. Configurar e executar o GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline_template, # Usa o pipeline
        param_grid=param_grid,
        scoring=scorer,
        cv=cv_janela_rolante, # <<< USA O GERADOR DE JANELA ROLANTE PERSONALIZADO
        n_jobs=-1,
        verbose=1
    )

    # Limpar dados antes do fit para segurança
    X_limpo = X_super_treino.copy()
    y_limpo = y_super_treino.copy()
    X_limpo.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_limpo.fillna(X_limpo.mean(), inplace=True) # Preencher NaNs

    grid_search.fit(X_limpo, y_limpo)

    print("\n--- Resultados da Otimização ---")
    print(f"Melhor Score (F1-médio na CV rolante): {grid_search.best_score_:.4f}")
    print("Melhores Parâmetros Encontrados:")
    print(grid_search.best_params_)

    # O grid_search.best_estimator_ já é o modelo (pipeline) treinado com os melhores parâmetros
    # em TODO o X_super_treino.
    return grid_search.best_estimator_


def otimizar_threshold_com_walk_forward_rolante(
        X_super_train: pd.DataFrame,
        y_super_train: pd.Series,
        modelo_nao_treinado_factory, # Função que retorna uma nova instância do modelo
        tamanho_janela_treino: int, # Ex: 126 (aprox. 6 meses de dias úteis)
        tamanho_janela_validacao: int, # Ex: 21 (aprox. 1 mês de dias úteis)
        passo_wf: int, # Ex: 21 (desliza 1 mês a cada iteração)
        recall_minimo_desejado: float,
        thresholds_para_testar = np.arange(0.40, 0.61, 0.01), # Um range razoável para começar
        metrica_otimizacao = 'precision' # Nova: Métrica principal para escolha ('precision', 'f1', ou objeto scorer)
    ):
    """
    Realiza validação walk-forward com janela de treino ROLANTE e retreinamento,
    para encontrar o threshold de decisão ideal.

    Args:
        ... (outros args) ...
        metrica_otimizacao: String (ex: 'precision') ou objeto scorer (ex: make_scorer(fbeta_score, beta=0.5)).
                            Define qual métrica será maximizada na escolha do threshold.
    """
    print(f"\n--- Iniciando Otimização de Threshold com Walk-Forward Rolante ---")
    print(f"Janela de Treino: {tamanho_janela_treino} amostras, Janela de Validação: {tamanho_janela_validacao}, Passo: {passo_wf}")

    # Determinar a função de score a ser usada
    scorer_func = None
    nome_metrica = "custom_score"
    
    if callable(metrica_otimizacao):
        # Se for um objeto scorer (ex: make_scorer), extraímos a função interna se possível ou usamos wrapper
        # Scikit-learn scorers geralmente são chamados como scorer(estimator, X, y)
        # Aqui precisamos calcular manualmente baseados nas predições binárias
        pass # Trataremos isso dentro do loop de forma genérica se possível, ou simplificamos para métricas padrão
    elif isinstance(metrica_otimizacao, str):
        nome_metrica = metrica_otimizacao

    # Dicionário para acumular métricas para cada threshold em todos os folds
    resultados_agregados = {
        thresh: {'precisions': [], 'recalls': [], 'scores_otimizacao': []} for thresh in thresholds_para_testar
    }
    
    n_total_amostras = len(X_super_train)
    num_janelas_executadas = 0

    # O loop desliza o INÍCIO da janela de treino
    for inicio_treino_idx in range(0, n_total_amostras, passo_wf):
        fim_treino_idx = inicio_treino_idx + tamanho_janela_treino
        inicio_validacao_idx = fim_treino_idx
        fim_validacao_idx = inicio_validacao_idx + tamanho_janela_validacao

        # Se a janela de validação ultrapassar o final dos dados, paramos
        if fim_validacao_idx > n_total_amostras:
            break

        X_treino_janela = X_super_train.iloc[inicio_treino_idx : fim_treino_idx]
        y_treino_janela = y_super_train.iloc[inicio_treino_idx : fim_treino_idx]
        X_validacao_janela = X_super_train.iloc[inicio_validacao_idx : fim_validacao_idx]
        y_validacao_janela = y_super_train.iloc[inicio_validacao_idx : fim_validacao_idx]

        if X_treino_janela.empty or X_validacao_janela.empty or y_treino_janela.sum() == 0:
            print(f"  AVISO: Janela {num_janelas_executadas + 1} pulada (dados insuficientes ou sem classe positiva).")
            continue
            
        num_janelas_executadas += 1
        print(f"  Walk-Forward Janela {num_janelas_executadas}: Treinando com {len(X_treino_janela)}, Validando com {len(X_validacao_janela)}...")

        # 1. Instanciar e RETREINAR o modelo para esta janela de treino
        modelo_janela = modelo_nao_treinado_factory()
        try:
            modelo_janela.fit(X_treino_janela.values, y_treino_janela.values) 
        except Exception as e_fit:
            print(f"    ERRO ao treinar modelo na janela {num_janelas_executadas}: {e_fit}. Pulando esta janela.")
            continue

        # 2. Obter probabilidades na janela de validação
        try:
            probabilidades_val = modelo_janela.predict_proba(X_validacao_janela.values)[:, 1]
        except Exception as e_pred:
            print(f"    ERRO ao prever na janela {num_janelas_executadas}: {e_pred}. Pulando.")
            continue

        # 3. Calcular métricas para cada threshold e acumular
        y_val_real = y_validacao_janela.values
        
        for p_thresh in thresholds_para_testar:
            predicoes_binarias = (probabilidades_val >= p_thresh).astype(int)
            
            # Métricas Padrão
            precision = precision_score(y_val_real, predicoes_binarias, zero_division=0)
            recall = recall_score(y_val_real, predicoes_binarias, zero_division=0)
            
            # Métrica de Otimização Personalizada
            score_otim = 0.0
            if callable(metrica_otimizacao):
                # Se for um scorer do sklearn, ele geralmente espera (estimator, X, y)
                # Mas aqui já temos as predições. 
                # Vamos assumir que se é callable, é uma função de métrica simples like fbeta_score(y_true, y_pred)
                # SE você passar make_scorer object, ele é mais complexo.
                # Simplificação: Se for 'fbeta', calculamos manualmente.
                # Se o usuário passou um objeto complexo, tentamos usar _score_func se disponível
                if hasattr(metrica_otimizacao, '_score_func'):
                     kwargs = metrica_otimizacao._kwargs if hasattr(metrica_otimizacao, '_kwargs') else {}
                     score_otim = metrica_otimizacao._score_func(y_val_real, predicoes_binarias, **kwargs)
                else:
                     # Tenta chamar como função(y_true, y_pred)
                     try:
                        score_otim = metrica_otimizacao(y_val_real, predicoes_binarias)
                     except:
                        score_otim = precision # Fallback
            elif metrica_otimizacao == 'f1':
                score_otim = f1_score(y_val_real, predicoes_binarias, zero_division=0)
            elif metrica_otimizacao == 'precision':
                score_otim = precision
            else:
                score_otim = precision # Default fallback

            resultados_agregados[p_thresh]['precisions'].append(precision)
            resultados_agregados[p_thresh]['recalls'].append(recall)
            resultados_agregados[p_thresh]['scores_otimizacao'].append(score_otim)
                
    if num_janelas_executadas == 0:
        print("AVISO: Nenhuma janela de walk-forward foi executada.")
        return {'threshold_ideal': None, 'df_resultados_agregados': pd.DataFrame()}

    # Calcular médias e selecionar threshold ideal
    df_res_list = []
    for p_thresh, metrics_list in resultados_agregados.items():
        if metrics_list['precisions']:
            avg_precision = np.mean(metrics_list['precisions'])
            avg_recall = np.mean(metrics_list['recalls'])
            avg_score_otim = np.mean(metrics_list['scores_otimizacao'])
            std_score_otim = np.std(metrics_list['scores_otimizacao'])

            df_res_list.append({
                'threshold': p_thresh,
                'precisao_media': avg_precision,
                'recall_medio': avg_recall,
                'score_otimizacao_medio': avg_score_otim,
                'score_std': std_score_otim
            })
    
    df_resultados_agregados = pd.DataFrame(df_res_list)
    
    print("\n--- Resultados Agregados por Threshold (Walk-Forward no Super Treino) ---")
    if not df_resultados_agregados.empty:
        print(df_resultados_agregados.round(3))
    else:
        return {'threshold_ideal': None}

    # Selecionar o melhor threshold
    # 1. Filtra por recall mínimo (Hard Constraint)
    df_candidatos = df_resultados_agregados[df_resultados_agregados['recall_medio'] >= recall_minimo_desejado].copy()

    threshold_ideal_final = None
    precisao_otima_final = None
    
    if not df_candidatos.empty:
        # 2. Ordena pela métrica de otimização escolhida (ex: fbeta), depois por estabilidade (menor desvio)
        df_candidatos.sort_values(by=['score_otimizacao_medio', 'score_std'], ascending=[False, True], inplace=True)
        
        melhor_candidato = df_candidatos.iloc[0]
        threshold_ideal_final = melhor_candidato['threshold']
        precisao_otima_final = melhor_candidato['precisao_media']
        
        print(f"\n--- Threshold Ideal Selecionado ---")
        print(f"  Métrica de Otimização: {nome_metrica}")
        print(f"  Threshold: {threshold_ideal_final:.3f}")
        print(f"  Score Médio ({nome_metrica}): {melhor_candidato['score_otimizacao_medio']:.4f}")
        print(f"  Precisão Média: {precisao_otima_final:.2%}")
        print(f"  Recall Médio: {melhor_candidato['recall_medio']:.2%}")
    else:
        print(f"\nAVISO: Nenhum threshold atendeu ao recall mínimo de {recall_minimo_desejado:.2%}.")

    return {
        'threshold_ideal': threshold_ideal_final,
        'precisao_media_otima': precisao_otima_final,
        'df_resultados_agregados': df_resultados_agregados
    }


def backtest_final_walk_forward(
        modelo_template, # Um modelo NÃO treinado com os hiperparâmetros já otimizados
        X_super_treino: pd.DataFrame,
        y_super_treino: pd.Series,
        X_validacao_final: pd.DataFrame,
        y_validacao_final: pd.Series,
        tamanho_janela_treino: int, # Ex: 252 (aprox. 1 ano)
        passo_wf: int = 21 # Ex: 21 (retreina a cada 1 mês)
    ) -> dict:
    """
    Realiza um backtest final em um conjunto de validação (hold-out) usando
    uma estratégia de walk-forward com janela rolante e retreinamento.

    Args:
        modelo_template: Uma instância NÃO TREINADA do modelo final com hiperparâmetros otimizados.
        X_super_treino: O conjunto completo de features de treino.
        y_super_treino: Os alvos completos de treino.
        X_validacao_final: O conjunto de features de validação hold-out.
        y_validacao_final: Os alvos de validação hold-out.
        tamanho_janela_treino (int): Tamanho da janela de treino rolante.
        passo_wf (int): O número de amostras para avançar a cada retreinamento.

    Returns:
        dict: Um dicionário contendo as previsões de probabilidade, os alvos reais,
              e um DataFrame com as métricas calculadas em todo o período de validação.
    """
    print("\n--- Iniciando Backtest Final com Walk-Forward e Retreinamento ---")
    print(f"Janela de Treino: {tamanho_janela_treino}, Passo de Retreinamento: {passo_wf}")
    
    # Combinar os dados de treino e validação para facilitar o fatiamento
    X_completo = pd.concat([X_super_treino, X_validacao_final])
    y_completo = pd.concat([y_super_treino, y_validacao_final])

    lista_probas_out_of_sample = []
    lista_y_reais_out_of_sample = []
    indices_out_of_sample = []

    # O loop vai iterar sobre o conjunto de validação, com base no passo_wf
    n_validacao = len(X_validacao_final)
    offset_inicio_treino = len(X_super_treino) - tamanho_janela_treino

    for i in range(0, n_validacao, passo_wf):
        
        # --- Definir Janelas ---
        # Janela de treino: `tamanho_janela_treino` amostras terminando ANTES do início da janela de teste
        inicio_treino_idx = offset_inicio_treino + i
        fim_treino_idx = inicio_treino_idx + tamanho_janela_treino
        
        # Janela de teste: `passo_wf` amostras (ou menos, no último passo)
        inicio_teste_idx = fim_treino_idx
        fim_teste_idx = min(inicio_teste_idx + passo_wf, len(X_completo))

        if inicio_teste_idx >= fim_teste_idx: # Se não houver mais dados para testar
            break

        X_treino_janela = X_completo.iloc[inicio_treino_idx : fim_treino_idx]
        y_treino_janela = y_completo.iloc[inicio_treino_idx : fim_treino_idx]
        X_teste_janela = X_completo.iloc[inicio_teste_idx : fim_teste_idx]
        y_teste_janela = y_completo.iloc[inicio_teste_idx : fim_teste_idx]

        print(f"  Backtest Passo {i//passo_wf + 1}: Treinando com {len(X_treino_janela)} amostras [{X_treino_janela.index.min()} a {X_treino_janela.index.max()}], "
              f"Prevendo {len(X_teste_janela)} amostras [{X_teste_janela.index.min()} a {X_teste_janela.index.max()}]...")

        # --- Treinar e Prever ---
        modelo = clone(modelo_template)
        try:
            modelo.fit(X_treino_janela, y_treino_janela)
            probas_janela = modelo.predict_proba(X_teste_janela)[:, 1]
            
            # Armazenar resultados
            lista_probas_out_of_sample.extend(probas_janela)
            lista_y_reais_out_of_sample.extend(y_teste_janela.values)
            indices_out_of_sample.extend(y_teste_janela.index)
            
        except Exception as e:
            print(f"    ERRO no Passo {i//passo_wf + 1}: {e}. Pulando este passo.")
            # Preencher com previsões "neutras" (0.5) para manter o alinhamento
            lista_probas_out_of_sample.extend([0.5] * len(y_teste_janela))
            lista_y_reais_out_of_sample.extend(y_teste_janela.values)
            indices_out_of_sample.extend(y_teste_janela.index)

    # --- Consolidar e Avaliar Resultados Finais ---
    if not lista_probas_out_of_sample:
        print("AVISO: Nenhum resultado de backtest foi gerado.")
        return None

    probas_finais_val = pd.Series(lista_probas_out_of_sample, index=indices_out_of_sample)
    y_reais_finais_val = pd.Series(lista_y_reais_out_of_sample, index=indices_out_of_sample)

    # Calcular métricas em diferentes thresholds para análise
    resultados_thresholds = []
    for threshold in np.arange(0.40, 0.61, 0.01):
        y_pred = (probas_finais_val >= threshold).astype(int)
        
        prec = precision_score(y_reais_finais_val, y_pred, zero_division=0)
        rec = recall_score(y_reais_finais_val, y_pred, zero_division=0)
        f1 = f1_score(y_reais_finais_val, y_pred, zero_division=0)
        num_sinais = y_pred.sum()

        resultados_thresholds.append({
            'threshold': threshold,
            'precisao': prec,
            'recall': rec,
            'f1_score': f1,
            'num_sinais': num_sinais
        })
    
    df_metricas_finais = pd.DataFrame(resultados_thresholds)
    
    print("\n--- Resultados Finais do Backtest no Hold-Out ---")
    print(df_metricas_finais.round(3).to_string())
    
    return {
        'probas_validacao': probas_finais_val,
        'y_reais_validacao': y_reais_finais_val,
        'metricas_por_threshold': df_metricas_finais
    }



import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from collections import Counter
from itertools import combinations

def encontrar_interacoes_robustas_com_wf(
        X_super_treino: pd.DataFrame,
        y_super_treino: pd.Series,
        max_profundidade: int = 3, # Manter a árvore rasa para focar nas interações mais fortes
        min_amostras_folha: int = 20,
        tamanho_janela_treino: int = 252,
        passo_wf: int = 21 # Quanto a janela desliza a cada iteração
    ):
    """
    Usa um processo de walk-forward com árvores de decisão para encontrar
    pares de features que interagem de forma consistente e robusta ao longo do tempo.

    Args:
        X_super_treino: DataFrame completo de features para o "super" treino.
        y_super_treino: Série completa de alvos para o "super" treino.
        max_profundidade: Profundidade máxima das árvores de decisão.
        min_amostras_folha: Mínimo de amostras por folha para regularizar.
        tamanho_janela_treino: Número de amostras na janela de treino rolante.
        passo_wf: Número de amostras para deslizar a janela.

    Returns:
        pd.DataFrame: Um DataFrame com os pares de features que mais apareceram
                      juntos nos caminhos das árvores, ordenados por frequência.
    """
    print(f"\n--- Buscando Interações Robustas com Walk-Forward ---")
    print(f"Janela de Treino: {tamanho_janela_treino}, Passo: {passo_wf}")
    
    # Modelo template
    arvore_template = DecisionTreeClassifier(
        max_depth=max_profundidade,
        min_samples_leaf=min_amostras_folha,
        class_weight='balanced',
        random_state=42
    )

    n_total_amostras = len(X_super_treino)
    num_janelas_executadas = 0
    
    contador_de_pares = Counter() # Para contar a frequência de cada par de interação
    contador_features_individuais = Counter() # Para contar a importância individual

    # Loop de walk-forward com janela ROLANTE
    for inicio_treino_idx in range(0, n_total_amostras - tamanho_janela_treino + 1, passo_wf):
        fim_treino_idx = inicio_treino_idx + tamanho_janela_treino
        
        num_janelas_executadas += 1
        print(f"  WF Janela {num_janelas_executadas}: Treinando com {tamanho_janela_treino} amostras...")

        X_treino_janela = X_super_treino.iloc[inicio_treino_idx : fim_treino_idx]
        y_treino_janela = y_super_treino.iloc[inicio_treino_idx : fim_treino_idx]

        if X_treino_janela.empty or y_treino_janela.sum() < 2: # Pula se não houver positivos suficientes
            continue

        # Retreinar a árvore na janela atual
        arvore_janela = clone(arvore_template)
        arvore_janela.fit(X_treino_janela, y_treino_janela)

        # --- Extrair interações da árvore treinada ---
        tree_ = arvore_janela.tree_
        feature_names = X_treino_janela.columns.tolist()
        
        # Percorrer os nós da árvore para encontrar os caminhos
        node_stack = [(0, [])] # (node_id, caminho_de_features_ate_aqui)
        while len(node_stack) > 0:
            node_id, parent_path = node_stack.pop()

            # Se for um nó de decisão (não uma folha)
            if tree_.children_left[node_id] != tree_.children_right[node_id]:
                feature_index = tree_.feature[node_id]
                feature_name = feature_names[feature_index]
                
                # Contar a importância individual da feature
                contador_features_individuais[feature_name] += 1
                
                # Criar e contar os pares de interação
                # Um par é formado pela feature atual e todas as features no caminho até ela
                for parent_feature in parent_path:
                    # Ordenar o par para que (A, B) seja igual a (B, A)
                    par_ordenado = tuple(sorted((parent_feature, feature_name)))
                    contador_de_pares[par_ordenado] += 1
                
                # Adicionar nós filhos à pilha para continuar a busca
                current_path = parent_path + [feature_name]
                node_stack.append((tree_.children_left[node_id], current_path))
                node_stack.append((tree_.children_right[node_id], current_path))

    if num_janelas_executadas == 0:
        print("AVISO: Nenhuma janela de walk-forward foi executada.")
        return pd.DataFrame()

    print(f"\nAnálise concluída após {num_janelas_executadas} janelas de treino.")

    # Processar os resultados
    if not contador_de_pares:
        print("Nenhuma interação de pares foi encontrada consistentemente.")
        return pd.DataFrame()
        
    df_interacoes = pd.DataFrame(contador_de_pares.items(), columns=['par_features', 'frequencia'])
    df_interacoes[['feature_1', 'feature_2']] = pd.DataFrame(df_interacoes['par_features'].tolist(), index=df_interacoes.index)
    df_interacoes = df_interacoes[['feature_1', 'feature_2', 'frequencia']].sort_values(by='frequencia', ascending=False)
    
    return df_interacoes    