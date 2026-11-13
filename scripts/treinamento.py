from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow         # type: ignore
import mlflow.sklearn # type: ignore

import utils

EXPERIMENT_NAME = "Palpiteiro"
CAMINHO_BASE_MLFLOW = r'C:\palpiteiro\mlruns'

def comparar_auc_conjuntos(modelo, X_treino, y_treino, X_validacao, y_validacao, verboso=True):    

    precisao_media, recall_media, janelas_com_previsao, auc_media, pr_auc_media, f1_score = avaliar_modelo_janelas_deslizantes(modelo, X_treino, y_treino, threshold=0.5, window_size=30, probas=modelo.predict_proba(X_treino)[:, 1])
    roc_treino = auc_media
    pr_auc_treino = pr_auc_media
    f1_score_treino = f1_score

    precisao_media, recall_media, janelas_com_previsao, auc_media, pr_auc_media, f1_score = avaliar_modelo_janelas_deslizantes(modelo, X_validacao, y_validacao, threshold=0.5, window_size=30, probas=modelo.predict_proba(X_validacao)[:, 1])
    roc_validacao = auc_media
    pr_auc_validacao = pr_auc_media
    f1_score_validacao = f1_score
    
    diferenca = roc_treino - roc_validacao

    if verboso:
        print(f"AUC no treino: {roc_treino:.3f}")    
        print(f"AUC na validação: {roc_validacao:.3f}")
        print(f"Diferença AUC (Treino - Validação): {diferenca:.3f}")
        if diferenca > 0.1:
            print("PROVÁVEL OVERFITTING")

        print(f"PR AUC no treino: {pr_auc_treino:.3f}")
        print(f"PR AUC na validação: {pr_auc_validacao:.3f}")
        print(f"Diferença PR AUC (Treino - Validação): {pr_auc_treino - pr_auc_validacao:.3f}")

        print(f"F1 Score no treino: {f1_score_treino:.3f}")
        print(f"F1 Score na validação: {f1_score_validacao:.3f}")

        plotar_curvas_precision_recall(modelo, X_treino, y_treino, X_validacao, y_validacao)




    return roc_treino, roc_validacao, diferenca, pr_auc_treino, pr_auc_validacao, f1_score_treino, f1_score_validacao


def eliminar_features_correlacionadas(df, limite_correlacao=0.95, target=None):
    """
    Remove colunas altamente correlacionadas do DataFrame, mantendo as mais relevantes.

    Parâmetros:
    - df: DataFrame de entrada (X), somente com variáveis explicativas
    - limite_correlacao: limiar de correlação acima do qual duas features são consideradas redundantes
    - target: Series com a variável alvo (opcional). Se fornecido, ajuda a escolher qual feature manter

    Retorna:
    - df_filtrado: DataFrame sem as features eliminadas
    - colunas_removidas: lista das colunas eliminadas
    """

    matriz_corr = df.corr().abs()
    upper = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))

    colunas_removidas = set()

    for col in upper.columns:
        correlacoes_altas = upper[col][upper[col] > limite_correlacao].index.tolist()
        for outra_col in correlacoes_altas:
            if col in colunas_removidas or outra_col in colunas_removidas:
                continue

            if target is not None:
                corr_col = abs(np.corrcoef(df[col], target)[0, 1])
                corr_outra = abs(np.corrcoef(df[outra_col], target)[0, 1])
                if corr_col >= corr_outra:
                    colunas_removidas.add(outra_col)
                else:
                    colunas_removidas.add(col)
            else:
                colunas_removidas.add(outra_col)  # Mantém `col` por padrão

    df_filtrado = df.drop(columns=list(colunas_removidas))
    return df_filtrado, list(colunas_removidas)


def avaliar_modelo(modelo, X_val, y_val, threshold=0.5):
    """
    Avalia o modelo com base em um limiar ajustável.

    Parâmetros:
    - modelo: modelo já treinado
    - X_val: dados de validação
    - y_val: rótulos reais
    - threshold: limite mínimo de probabilidade para considerar classe 1
    """
    # Garante que X_val seja 2D
    if isinstance(X_val, pd.Series):
        X_val = X_val.to_frame()
    elif hasattr(X_val, 'ndim') and X_val.ndim == 1:
        X_val = X_val.values.reshape(-1, 1)

    probas = modelo.predict_proba(X_val)[:, 1]
    y_pred = (probas >= threshold).astype(int)

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)

    print(f"\nRelatório de Validação (threshold = {threshold}):")
    print(classification_report(y_val, y_pred, digits=2))    
    print("Precisão (classe 1):", precision)
    print("Recall (classe 1):", recall)

    return {
        'precisao': precision,
        'recall': recall
    }
def avaliar_modelo_janelas_deslizantes(modelo, X, y, threshold, window_size, probas):
    precisoes = []
    recalls = []
    aucs = []
    pr_aucs = []
    f1_scores = []
    janelas_com_previsao = 0

    for start in range(0, len(X) - window_size + 1):
        end = start + window_size
        X_window = X[start:end]
        y_window = y[start:end]
        y_probas = probas[start:end]

        y_pred = (y_probas >= threshold).astype(int)
        recall = recall_score(y_window, y_pred, zero_division=0)
        precision = precision_score(y_window, y_pred, zero_division=0)        
        f1 = f1_score(y_window, y_pred, zero_division=0)

        if len(np.unique(y_window)) > 1:
            auc = roc_auc_score(y_window, y_probas)
            pr_auc = average_precision_score(y_window, y_probas)
            aucs.append(auc)
            pr_aucs.append(pr_auc)            
            f1_scores.append(f1)

        num_positive_predictions = np.sum(y_pred)

        if num_positive_predictions > 0:
            precisoes.append(precision)
            
            janelas_com_previsao += 1

        recalls.append(recall)

    precisao_media = np.mean(precisoes) if precisoes else 0
    recall_media = np.mean(recalls) if recalls else 0
    auc_media = np.mean(aucs) if aucs else 0
    pr_auc_media = np.mean(pr_aucs) if pr_aucs else 0
    f1_score_medio = np.mean(f1_scores) if f1_scores else 0

    return precisao_media, recall_media, janelas_com_previsao, auc_media, pr_auc_media, f1_score_medio


def avaliar_modelo_janelas_deslizantes_thresholds(modelo, X_val, y_val, recall_minimo=None, precisao_minima=None, threshold_fixo=None, window_size=30):
    """
    Avalia o modelo usando uma janela deslizante de validação, com possibilidade de calibrar o threshold com base em um recall mínimo.

    Parâmetros:
    - modelo: modelo já treinado
    - X_val: dados de validação (array ou DataFrame)
    - y_val: rótulos reais (array ou Series)
    - threshold: limiar de decisão (usado se recall_minimo não for fornecido)
    - window_size: tamanho da janela deslizante
    - recall_minimo: se fornecido, calibra automaticamente o threshold para atingir esse mínimo de recall
    """

    probas_val = modelo.predict_proba(X_val)[:, 1]


    # apuração de precision e recall para todos os thresholds por janela deslizante    
    if (threshold_fixo is None):
        thresholds_testados = np.linspace(0.4, 0.9, 81)
    else:
        thresholds_testados = [threshold_fixo]

    all_fold_precisions = {thresh: [] for thresh in thresholds_testados}
    all_fold_recalls = {thresh: [] for thresh in thresholds_testados}
    all_fold_janelas = {thresh: [] for thresh in thresholds_testados}

    janelas_usadas_para_precisao = 0    

    for threshold in thresholds_testados:                
        precisao_media, recall_media, janelas_com_previsao, auc, pr_auc, f1_score = avaliar_modelo_janelas_deslizantes(modelo, X_val, y_val, threshold, window_size, probas_val)        
        if (threshold==0.5):
            print(f"\nThreshold {threshold:.2f} - AUC: {auc:.3f}")
            print(f"  Precisão média (em {janelas_com_previsao} janelas com previsão positiva): {precisao_media:.3f}")
            print(f"  Recall médio: {recall_media:.3f}")        

        if janelas_com_previsao > 0:      
            all_fold_precisions[threshold].append(precisao_media)            
            all_fold_recalls[threshold].append(recall_media)
            all_fold_janelas[threshold].append(janelas_com_previsao)


    # analisar os resultados por threshold
    resultados = []
    for threshold in thresholds_testados:
        if (len(all_fold_precisions[threshold]) > 0):
            avg_precision = np.mean(all_fold_precisions[threshold])
            avg_recall = np.mean(all_fold_recalls[threshold])
            num_janelas = np.sum(all_fold_janelas[threshold])
            resultados.append({
                'threshold': threshold,
                'precisao_media': avg_precision,
                'recall_media': avg_recall,
                'janelas_usadas_para_precisao': num_janelas
            })

    df_resultados = pd.DataFrame(resultados, columns=['threshold', 'precisao_media', 'recall_media', 'janelas_usadas_para_precisao'])

    # localizar result do thresh 0.5
    idx_05 = df_resultados[df_resultados['threshold'] == 0.5].index
    precisao_05 = df_resultados.loc[idx_05, 'precisao_media'].values[0] if not idx_05.empty else 0
    recall_05 = df_resultados.loc[idx_05, 'recall_media'].values[0] if not idx_05.empty else 0   

    if (recall_minimo is not None):
        print(f"\nCalibrando thresholds para recall mínimo de {recall_minimo:.2f}...")
        df_resultados = df_resultados.sort_values(by='precisao_media', ascending=False)

        #filtrar os resultados que atendem ao recall mínimo
        df_resultados = df_resultados[df_resultados['recall_media'] >= recall_minimo]

        if df_resultados.empty:
            print(f"Nenhum threshold atendeu ao recall mínimo de {recall_minimo:.2f}. Seguindo com 0.5")
            return {
                'precisao_media': precisao_05,
                'recall_media': recall_05,
                'threshold_ideal': 0.5
            }
        
    elif (precisao_minima is not None):
        print(f"\nCalibrando thresholds para precisão mínima de {precisao_minima:.2f}...")
        df_resultados = df_resultados.sort_values(by='recall_media', ascending=False)

        #filtrar os resultados que atendem à precisão mínima
        df_resultados = df_resultados[df_resultados['precisao_media'] >= precisao_minima]

        if df_resultados.empty:
            print(f"Nenhum threshold atendeu à precisão mínima de {precisao_minima:.2f}. Seguindo com 0.5")
            return {
                'precisao_media': precisao_05,
                'recall_media': recall_05,
                'threshold_ideal': 0.5
            }

    threshold_ideal = df_resultados.iloc[0]['threshold']
    precisao_ideal = df_resultados.iloc[0]['precisao_media']
    recall_ideal = df_resultados.iloc[0]['recall_media']
    janelas_usadas_para_precisao = df_resultados.iloc[0]['janelas_usadas_para_precisao']
    print(f"\nThreshold Ideal Selecionado: {threshold_ideal:.2f}")
    print(f"  Precisão Média Esperada: {precisao_ideal:.2%}")
    print(f"  Recall Médio Esperado: {recall_ideal:.2%}")
    print(f"  Janelas Usadas para Precisão: {janelas_usadas_para_precisao}")

    return {
        'precisao_media': precisao_ideal,
        'recall_media': recall_ideal,                 
        'threshold_ideal': threshold_ideal
    }



def avaliar_modelo_janela(modelo, X_val, y_val, threshold=0.5, window_size=30, recall_minimo=None):
    """
    Avalia o modelo usando uma janela deslizante de validação, com possibilidade de calibrar o threshold com base em um recall mínimo.

    Parâmetros:
    - modelo: modelo já treinado
    - X_val: dados de validação (array ou DataFrame)
    - y_val: rótulos reais (array ou Series)
    - threshold: limiar de decisão (usado se recall_minimo não for fornecido)
    - window_size: tamanho da janela deslizante
    - recall_minimo: se fornecido, calibra automaticamente o threshold para atingir esse mínimo de recall
    """

    probas_val = modelo.predict_proba(X_val)[:, 1]

    if recall_minimo is not None:
        thresholds_testados = np.linspace(0.1, 0.9, 81)
        melhor_threshold = threshold
        melhor_precisao = 0
        for t in thresholds_testados:
            y_pred = (probas_val >= t).astype(int)
            recall = recall_score(y_val, y_pred, zero_division=0)
            precisao = precision_score(y_val, y_pred, zero_division=0)
            if recall >= recall_minimo and precisao > melhor_precisao:
                melhor_precisao = precisao
                melhor_threshold = t
        threshold = melhor_threshold
        print(f"→ Threshold calibrado para recall ≥ {recall_minimo}: {threshold:.2f}")
    else:
        print(f"→ Usando threshold fixo: {threshold:.2f}")

    precisions = []
    recalls = []
    accs = []
    janelas_usadas_para_precisao = 0

    for start in range(0, len(X_val) - window_size + 1):
        end = start + window_size
        X_window = X_val[start:end]
        y_window = y_val[start:end]

        probas = modelo.predict_proba(X_window)[:, 1]
        y_pred = (probas >= threshold).astype(int)

        # Só computa precisão se houve pelo menos uma previsão positiva
        if y_pred.sum() > 0:
            precisions.append(precision_score(y_window, y_pred, zero_division=0))
            janelas_usadas_para_precisao += 1
        recalls.append(recall_score(y_window, y_pred, zero_division=0))
        accs.append(accuracy_score(y_window, y_pred))

    print(f"\nAvaliação com Janela Deslizante ({window_size} dias):")
    print(f"Precisão média (em {janelas_usadas_para_precisao} janelas com previsão positiva): {np.mean(precisions):.3f}" if precisions else "Nenhuma janela com previsão positiva.")
    print(f"Recall médio: {np.mean(recalls):.3f}")
    print(f"Acurácia média: {np.mean(accs):.3f}")

    return {
        'precisao_media': np.mean(precisions) if precisions else None,
        'recall_media': np.mean(recalls),        
        'todas_precisoes': precisions,
        'todas_recalls': recalls,        
        'threshold_utilizado': threshold
    }


def plotar_curva_precision_recall(modelo, X_val, y_val, titulo):
    """
    Plota a curva de precisão x recall para diversos thresholds.
    """
    probas = modelo.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probas)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label='Precisão', color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', color='green')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show()


def plotar_curvas_precision_recall(modelo, X_treino, y_treino, X_val, y_val):
    """
    Plota a curva de precisão x recall para diversos thresholds.
    """

    probas_treino = modelo.predict_proba(X_treino)[:, 1]
    precision_treino, recall_treino, thresholds_treino = precision_recall_curve(y_treino, probas_treino)

    probas_val = modelo.predict_proba(X_val)[:, 1]
    precision_val, recall_val, thresholds_val = precision_recall_curve(y_val, probas_val)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
    axes[0].plot(thresholds_treino, precision_treino[:-1], label='Precisão', color='blue')
    axes[0].plot(thresholds_treino, recall_treino[:-1], label='Recall', color='green')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title("Treino - Curva de Precisão x Recall")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(thresholds_val, precision_val[:-1], label='Precisão', color='blue')
    axes[1].plot(thresholds_val, recall_val[:-1], label='Recall', color='green')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].set_title("Validação - Curva de Precisão x Recall")
    axes[1].legend()
    axes[1].grid(True)

    plt.show()


def escolher_threshold_ideal(modelo, X_val, y_val, recall_minimo=0.2):
    probas = modelo.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probas)

    # Encontrar o maior threshold que satisfaça o recall mínimo com melhor precisão
    indices_validos = [i for i in range(len(thresholds)) if recalls[i] >= recall_minimo]
    if not indices_validos:
        return 0.5  # fallback

    melhor_indice = max(indices_validos, key=lambda i: precisions[i])
    return thresholds[melhor_indice]    


def dividir_treino_validacao(X, y, dias_validacao, dias_futuro):
    """
    Divide os dados em conjuntos de treino e validação.
    - dias_futuro (int): quantos dias à frente olhar para definir o target.
    """
    divisao_treino_validacao = dias_validacao + dias_futuro
    X_train = X.iloc[:-divisao_treino_validacao].copy()
    y_train = y.iloc[:-divisao_treino_validacao].copy()
    X_validacao = X.iloc[-divisao_treino_validacao:-dias_futuro].copy()
    y_validacao = y.iloc[-divisao_treino_validacao:-dias_futuro].copy()
    return X_train, y_train, X_validacao, y_validacao


def avaliar_e_logar_modelo(modelo, nome_papel, X_train, y_train, X_validacao, y_validacao, dias_janela_x, valorizacao, stop_loss, dias_futuro_valorizacao, dias_validacao, recall_minimo=None, precisao_minima=None, threshold_fixo=None,dias_janela_deslizante=30, nome_modelo=None):   
    print("Avaliar e logar modelo")
    
    utils.preparar_ambiente_mlflow()

    if nome_modelo is None:
        nome_modelo = f"{nome_papel}_{dias_janela_x}dias_x_{valorizacao}val_{dias_futuro_valorizacao}dias_alvo"

    with mlflow.start_run(run_name=nome_modelo) as run:  
        mlflow.log_param("papel", nome_papel) 
        mlflow.log_param("dias_janela_x", dias_janela_x)
        mlflow.log_param("dias_validacao", dias_validacao)
        mlflow.log_param("recall_minimo", recall_minimo)                
        mlflow.log_param("valorizacao", valorizacao)
        mlflow.log_param("stop_loss", stop_loss)
        mlflow.log_param("dias_futuro_valorizacao", dias_futuro_valorizacao)

        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id} para {nome_modelo}")                
                
        features_list = list(X_train.columns)
        with open("features.txt", "w") as f:
            for feature in features_list:
                f.write(f"{feature}\n")
        mlflow.log_artifact("features.txt", artifact_path="model_config")        

        # Avaliação no conjunto de treino                
        from sklearn.metrics import roc_auc_score
        roc = roc_auc_score(y_train, modelo.predict_proba(X_train)[:,1])
        print(f"AUC no treino: {roc:.3f}")
        roc_validacao = roc_auc_score(y_validacao, modelo.predict_proba(X_validacao)[:,1])
        print(f"AUC na validação: {roc_validacao:.3f}")
        

        print("\n--- Conjunto de Treino --- Janelas")    
        #precision, recall, num_positive_predictions = avaliar_modelo_janelas_deslizantes(modelo, X_train.values, y_train.values, threshold=0.5, window_size=dias_janela)

        avaliacao_treino_janelas = avaliar_modelo_janelas_deslizantes_thresholds(modelo, X_train, y_train, recall_minimo=recall_minimo, window_size=dias_janela_deslizante, precisao_minima=precisao_minima, threshold_fixo=threshold_fixo) 

        # Calibrar o threshold ideal
        threshold_ideal = avaliacao_treino_janelas['threshold_ideal']
        print(f"\nThreshold ideal baseado em mínimo recall ou precisão: {threshold_ideal:.2f}")

        print("\n--- Conjunto de Validação --- Janelas")
        probas_validacao = modelo.predict_proba(X_validacao)[:, 1]
        precisao_media, recall_media, janelas_com_previsao, auc_val, pr_auc_val, f1_score = avaliar_modelo_janelas_deslizantes(modelo, X_validacao, y_validacao, threshold=threshold_ideal, window_size=dias_janela_deslizante, probas=probas_validacao)        

        print(f"Precisão média (em {janelas_com_previsao} janelas com previsão positiva): {precisao_media:.3f}")
        print(f"Recall médio: {recall_media:.3f}")
        print(f"Sinais de compra presentes na vallidação: {y_validacao.sum()}")
        print(f"Sinais de compra previstos na validação: {np.sum(probas_validacao >= threshold_ideal)}")

        mlflow.log_metric("AUC", roc)
        mlflow.log_metric("AUC_Val", roc_auc_score(y_validacao, probas_validacao))

        mlflow.log_metric("Precision_Treino_Janelas", avaliacao_treino_janelas['precisao_media'])        
        mlflow.log_metric("Recall_Treino_Janelas", avaliacao_treino_janelas['recall_media'])
       
        mlflow.log_metric("Precision_Val_Janelas", precisao_media)
        mlflow.log_metric("Recall_Val_Janelas", recall_media)        

        # precisão por faixa de probabilidade                
        df_probas = pd.DataFrame({
        'proba': probas_validacao,
        'real': y_validacao
        })
        bins=np.arange(0, 1.01, 0.05)
        df_probas['faixa'] = pd.cut(df_probas['proba'], bins=bins, right=False)
        resumo = df_probas.groupby('faixa', observed=False).agg(
            total=('real', 'count'),
            positivos=('real', 'sum')
        )
        resumo['precisao_empirica'] = resumo['positivos'] / resumo['total']
        resumo.index.name = 'faixa_probabilidade'
        resumo = resumo.reset_index()

        resumo = resumo.dropna()           
        
        resumo[['faixa_probabilidade','precisao_empirica']].to_csv('precisoes.csv')
        mlflow.log_artifact("precisoes.csv", artifact_path="model_config")      
        
        mlflow.log_param("threshold", threshold_ideal)

        mlflow.sklearn.log_model(
                sk_model=modelo,
                artifact_path="model", # Nome da pasta dentro dos artefatos da run
                input_example=X_train.head(5), # Opcional, mas bom para schema
                signature=mlflow.models.infer_signature(X_train, y_train) # Opcional, mas bom para schema
            )
        print(f"Modelo {nome_modelo} salvo na run {run_id}")

    #plotar_curva_precision_recall(modelo, X_train, y_train, titulo='Curva de Precisão x Recall - Treino')
    #plotar_curva_precision_recall(modelo, X_validacao, y_validacao, titulo='Curva de Precisão x Recall - Validação')

    plotar_curvas_precision_recall(modelo, X_train, y_train, X_validacao, y_validacao)


    sns.kdeplot(modelo.predict_proba(X_train)[:, 1], label='Treino')
    sns.kdeplot(modelo.predict_proba(X_validacao)[:, 1], label='Validação')
    plt.legend()
    plt.suptitle(f"Distribuição das Probabilidades - Treino x Validação")
    plt.show()
    
    y_pred_validacao = modelo.predict(X_validacao)
    probas_validacao = modelo.predict_proba(X_validacao)[:, 1]

    sns.kdeplot(probas_validacao[y_validacao == 1], label='Classe 1')
    sns.kdeplot(probas_validacao[y_validacao == 0], label='Classe 0')
    plt.suptitle(f"Distribuição das Probabilidades - Validação (Classe 1 vs Classe 0)")
    plt.legend()
    plt.show()

    return {
        'modelo': modelo,
        'threshold_calibrado': threshold_ideal,        
        'X_validacao': X_validacao,
        'y_validacao': y_validacao,
        'probas_validacao':probas_validacao,
        'y_pred_validacao': y_pred_validacao
    }





def plotar_sinal_compras(modelo, df, X_val, dias_janela, threshold=0.5):
    """
    Plota o gráfico de fechamento com os pontos de compra sugeridos pelo modelo durante a janela de validação.

    Parâmetros:
    - modelo: modelo treinado
    - df: DataFrame original contendo colunas como 'close' e 'Target'       
    - threshold: limiar de probabilidade para sinal de compra
    """
    
    # Definir índices da validação (excluindo os últimos 5 dias)
    idx_inicio_val = X_val.index[0]
    idx_fim_val = X_val.index[-1] 

    # Preparar X_val e datas correspondentes
    
    datas_val = df.iloc[X_val.index].index
    close_val = df.iloc[X_val.index]['close'].values

    # Fazer previsões de probabilidade
    probas = modelo.predict_proba(X_val.values)[:, 1]
    sinais_compra = probas >= threshold
    

    # Plotar gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(datas_val, close_val, label='Preço de Fechamento', color='gray', marker='.')
    plt.scatter(np.array(datas_val)[sinais_compra], close_val[sinais_compra],
                label='Sinais de Compra', color='green', marker='^', s=100)

    plt.title('Sinais de Compra do Modelo')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def calcular_recall_em_precisao_minima(y_true, y_probas, precisao_minima=0.50):
    """
    (Função Auxiliar)
    Calcula o maior recall possível para uma precisão mínima desejada.

    Args:
        y_true: Rótulos verdadeiros.
        y_probas: Probabilidades previstas para a classe positiva.
        precisao_minima (float): O nível de precisão mínimo desejado.

    Returns:
        float: O valor do recall no ponto ótimo, ou 0 se a precisão mínima não for atingida.
    """
    # Se não houver nenhum evento positivo real, o recall é indefinido. Retornamos 0.
    if np.sum(y_true) == 0:
        return 0.0

    # precision_recall_curve retorna precisões, recalls e thresholds correspondentes
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probas)

    # A curva retorna um valor a mais para precisão e recall do que para thresholds.
    # Excluímos o último ponto (recall=0, precision=1) que é um artefato da função.
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Encontrar os índices onde a precisão é >= à mínima desejada
    indices_validos = np.where(precisions >= precisao_minima)[0]

    if len(indices_validos) == 0:
        # Se nenhum ponto na curva atinge a precisão mínima, o recall efetivo é 0.
        return 0.0
    else:
        # Dentre os pontos que atendem à precisão, pegamos o que tem o maior recall.
        melhor_recall = np.max(recalls[indices_validos])
        return melhor_recall
    


from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score

def avaliar_k_features_com_walk_forward(
        X_super_treino: pd.DataFrame,
        y_super_treino: pd.Series,
        k_values: list,
        modelo_template, # Modelo NÃO treinado, ex: LogisticRegression(...)
        tamanho_janela_treino_wf: int = 180,
        tamanho_janela_validacao_wf: int = 30,
        passo_wf: int = 30,
        precisao_minima_desejada: float = 0.50
    ):
    """
    Avalia o desempenho do modelo para diferentes números de features (k),
    usando um processo de walk-forward robusto DENTRO do conjunto de treino.

    Retorna um DataFrame com as métricas médias para cada valor de k.
    """
    print(f"\n--- Iniciando Avaliação Robusta de k=[{', '.join(map(str, k_values))}] com Walk-Forward ---")
    
    resultados_por_k = []

    for k in k_values:
        #print(f"\n--- Avaliando para k = {k} features ---")
        
        # Métricas para este valor de k, a serem agregadas de todos os folds
        f1s_para_k, precisions_para_k, recalls_para_k, recalls_na_precisao_minima_para_k = [], [], [], []
        num_janelas_executadas_para_k = 0

        # Loop Walk-Forward Interno para este k
        for inicio_treino_idx in range(0, len(X_super_treino) - tamanho_janela_treino_wf - tamanho_janela_validacao_wf + 1, passo_wf):
            fim_treino_idx = inicio_treino_idx + tamanho_janela_treino_wf
            inicio_validacao_idx = fim_treino_idx
            fim_validacao_idx = inicio_validacao_idx + tamanho_janela_validacao_wf
            
            num_janelas_executadas_para_k += 1
            #print(f"  WF Fold {num_janelas_executadas_para_k} para k={k}: Treinando com {tamanho_janela_treino_wf} amostras...")

            X_treino_fold = X_super_treino.iloc[inicio_treino_idx : fim_treino_idx]
            y_treino_fold = y_super_treino.iloc[inicio_treino_idx : fim_treino_idx]
            X_validacao_fold = X_super_treino.iloc[inicio_validacao_idx : fim_validacao_idx]
            y_validacao_fold = y_super_treino.iloc[inicio_validacao_idx : fim_validacao_idx]

            if X_treino_fold.empty or X_validacao_fold.empty: continue

            # a) Seleção de Features (baseado apenas nos dados de treino deste fold)
            try:
                # Treina um modelo temporário para obter importâncias
                modelo_para_importancias = clone(modelo_template)
                modelo_para_importancias.fit(X_treino_fold, y_treino_fold)
                
                # Extrai importâncias (coeficientes para Regressão Logística)
                # Usamos o valor absoluto para capturar importância em ambas as direções
                importancias_fold = pd.DataFrame({
                    'Feature': X_treino_fold.columns,
                    'Importance': np.abs(modelo_para_importancias.coef_[0])
                }).sort_values(by='Importance', ascending=False)
                
                # Seleciona as top k features para este fold
                colunas_selecionadas_fold = importancias_fold.head(k)['Feature'].tolist()

                # b) Retreinamento com features selecionadas
                modelo_retreinado_fold = clone(modelo_template)
                modelo_retreinado_fold.fit(X_treino_fold[colunas_selecionadas_fold], y_treino_fold)

                # c) Avaliação na janela de validação do fold
                y_pred_fold = modelo_retreinado_fold.predict(X_validacao_fold[colunas_selecionadas_fold])
                probas_teste_fold = modelo_retreinado_fold.predict_proba(X_validacao_fold[colunas_selecionadas_fold])[:, 1]

                f1s_para_k.append(f1_score(y_validacao_fold, y_pred_fold, zero_division=0))
                precisions_para_k.append(precision_score(y_validacao_fold, y_pred_fold, zero_division=0))
                recalls_para_k.append(recall_score(y_validacao_fold, y_pred_fold, zero_division=0))
                recall_neste_fold = calcular_recall_em_precisao_minima(y_validacao_fold, probas_teste_fold, precisao_minima_desejada)
                recalls_na_precisao_minima_para_k.append(recall_neste_fold)

            except Exception as e:
                print(f"  ERRO no Fold {num_janelas_executadas_para_k} para k={k}: {e}")
                f1s_para_k.append(0); precisions_para_k.append(0); recalls_para_k.append(0); recalls_na_precisao_minima_para_k.append(0)

        # Calcular métricas médias para este valor de k
        if f1s_para_k: # Se pelo menos um fold rodou
            resultados_por_k.append({
                'k_features': k,
                'f1_medio': np.mean(f1s_para_k),
                'precisao_media': np.mean(precisions_para_k),
                'recall_medio': np.mean(recalls_para_k),
                'recall_na_precisao_minima': np.mean(recalls_na_precisao_minima_para_k),
                'f1_std': np.std(f1s_para_k) # Desvio padrão para ver a estabilidade
            })
            print(f"  Resultado Médio para k={k}: F1={np.mean(f1s_para_k):.3f}, Precisão={np.mean(precisions_para_k):.3f}")

    df_resultados = pd.DataFrame(resultados_por_k)
    df_resultados.plot(x='k_features')
    plt.set_ylimit(0, 1)
    

    return df_resultados


### TESTE DO TAMANHO IDEAL DO CONJUNTO DE TREINO

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def avaliar_estrategia_com_recall_em_precisao_minima(
        modelo_template,
        X_super_treino: pd.DataFrame,
        y_super_treino: pd.Series,
        n_splits: int = 5,
        tamanho_janela_treino: int = None, # Esta será a 'janela rolante'
        tamanho_janela_validacao: int = 30, # Adicionado para clareza
        precisao_minima_desejada: float = 0.60
    ):
    """
    Avalia uma estratégia de modelagem usando TimeSeriesSplit para simular uma janela
    de treino rolante, focando na métrica "Recall na Precisão Mínima".
    """
    print(f"\n--- Avaliando Estratégia com Recall na Precisão Mínima de {precisao_minima_desejada:.0%} ---")
    
    # O tamanho do conjunto de teste em TimeSeriesSplit é len(X) // (n_splits + 1)
    # Se você quer um tamanho de janela de validação fixo, o loop manual é mais explícito.
    # Mas vamos adaptar a lógica do TimeSeriesSplit para se aproximar do que você quer.
    # A maneira mais limpa de usar TimeSeriesSplit para este propósito é ajustar n_splits
    # de forma que o tamanho do teste seja próximo do que você deseja.
    if tamanho_janela_treino is None: # Comportamento original de janela expansível
        print(f"  Modo: Janela Expansível (n_splits={n_splits})")
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=tamanho_janela_validacao) # Define o tamanho do teste
    else: # Simular janela rolante
        print(f"  Modo: Janela Rolante (treino_max={tamanho_janela_treino}, teste={tamanho_janela_validacao}, n_splits={n_splits})")
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=tamanho_janela_treino,
            test_size=tamanho_janela_validacao # <<< Adicionado para fixar o tamanho da validação
        )
    
    recalls_na_precisao_minima = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X_super_treino)):
        X_treino_fold = X_super_treino.iloc[train_index]
        y_treino_fold = y_super_treino.iloc[train_index] # <<< CORREÇÃO AQUI, antes estava test_index
        X_teste_fold = X_super_treino.iloc[test_index]
        y_teste_fold = y_super_treino.iloc[test_index]

        # Com test_size definido, a linha abaixo não é mais necessária, pois a janela já tem o tamanho certo
        # X_teste_fold = X_super_treino.iloc[test_index[:tamanho_janela_validacao]]
        # y_teste_fold = y_super_treino.iloc[test_index[:tamanho_janela_validacao]]

        print(f"Fold {fold + 1}/{n_splits}: Treinando com {len(X_treino_fold)} amostras, Validando com {len(X_teste_fold)}...")
        
        # <<< ADICIONADO: Pular o fold se não houver positivos no teste >>>
        # Isso evita que a métrica seja calculada em janelas onde não havia oportunidades
        if y_teste_fold.sum() == 0:
            print("  AVISO: Nenhum evento positivo na janela de teste. Recall será 0. Pulando para o próximo fold.")
            recalls_na_precisao_minima.append(0.0) # Adiciona 0 explicitamente
            continue

        modelo_fold = clone(modelo_template)
        try:
            modelo_fold.fit(X_treino_fold, y_treino_fold)
        except Exception as e:
            print(f"  ERRO no fit do Fold {fold + 1}: {e}. Pulando.")
            recalls_na_precisao_minima.append(0)
            continue
        
        probas_teste_fold = modelo_fold.predict_proba(X_teste_fold)[:, 1]

        # Supondo que tr.calcular_recall_em_precisao_minima é sua função auxiliar
        recall_neste_fold = calcular_recall_em_precisao_minima(y_teste_fold, probas_teste_fold, precisao_minima_desejada)
        recalls_na_precisao_minima.append(recall_neste_fold)
        print(f"  Recall na Precisão >= {precisao_minima_desejada:.0%}: {recall_neste_fold:.3f}")

    if not recalls_na_precisao_minima:
        # ... (resto do seu código) ...
        return {'recall_medio_na_prec_min': 0, 'recall_std_na_prec_min': 0, 'prob_sucesso_folds': 0}

    recall_medio = np.mean(recalls_na_precisao_minima)
    recall_std = np.std(recalls_na_precisao_minima)
    prob_sucesso_folds = np.mean([1 if r > 0 else 0 for r in recalls_na_precisao_minima])

    print(f"\nResultado Final da Avaliação Walk-Forward:")
    print(f"  Recall Médio (com Precisão >= {precisao_minima_desejada:.0%}): {recall_medio:.3f}")
    print(f"  Desvio Padrão do Recall: {recall_std:.3f}")
    print(f"  Percentual de Folds com Sinal Viável: {prob_sucesso_folds:.0%}")

    resultado = {
        'recall_medio_na_prec_min': recall_medio,
        'recall_std_na_prec_min': recall_std,
        'prob_sucesso_folds': prob_sucesso_folds
    }
    return resultado




from sklearn.base import clone
# Supondo que tr.calcular_recall_em_precisao_minima é sua função auxiliar

def avaliar_estrategia_walk_forward_rolante(
        modelo_template,
        X_super_treino: pd.DataFrame,
        y_super_treino: pd.Series,
        tamanho_janela_treino: int, # Tamanho FIXO da janela de treino rolante
        tamanho_janela_validacao: int = 30, # Tamanho FIXO da janela de validação
        passo_wf: int = 30, # Quanto a janela desliza a cada iteração
        precisao_minima_desejada: float = 0.60
    ):
    """
    Avalia uma estratégia de modelagem usando um loop manual de walk-forward
    com JANELA DE TREINO ROLANTE para garantir que não haja gaps.

    Retorna métricas médias e desvio padrão para o screening.
    """
    print(f"\n--- Avaliando Estratégia com Walk-Forward Rolante (Treino={tamanho_janela_treino}, Val={tamanho_janela_validacao}, Passo={passo_wf}) ---")
    print(f"    (Métrica: Recall na Precisão Mínima de {precisao_minima_desejada:.0%})")

    if len(X_super_treino) < tamanho_janela_treino + tamanho_janela_validacao:
        print(f"AVISO: Dados de treino insuficientes ({len(X_super_treino)}) para a primeira janela completa.")
        return {'recall_medio_na_prec_min': 0, 'recall_std_na_prec_min': 0, 'prob_sucesso_folds': 0}

    recalls_na_precisao_minima = []
    n_total_amostras = len(X_super_treino)
    num_janelas_executadas = 0

    # O loop desliza o INÍCIO da janela de treino a cada 'passo_wf'
    for inicio_treino_idx in range(0, n_total_amostras, passo_wf):
        fim_treino_idx = inicio_treino_idx + tamanho_janela_treino
        
        # O teste começa IMEDIATAMENTE após o fim do treino
        inicio_validacao_idx = fim_treino_idx
        fim_validacao_idx = inicio_validacao_idx + tamanho_janela_validacao

        # Se a janela de validação ultrapassar o final dos dados, paramos o loop
        if fim_validacao_idx > n_total_amostras:
            break

        num_janelas_executadas += 1
        
        X_treino_fold = X_super_treino.iloc[inicio_treino_idx : fim_treino_idx]
        y_treino_fold = y_super_treino.iloc[inicio_treino_idx : fim_treino_idx]
        X_teste_fold = X_super_treino.iloc[inicio_validacao_idx : fim_validacao_idx]
        y_teste_fold = y_super_treino.iloc[inicio_validacao_idx : fim_validacao_idx]

        print(f"Fold {num_janelas_executadas}: Treinando de {X_treino_fold.index.min()} a {X_treino_fold.index.max()} ({len(X_treino_fold)} amostras). "
              f"Validando de {X_teste_fold.index.min()} a {X_teste_fold.index.max()} ({len(X_teste_fold)} amostras).")

        if y_teste_fold.sum() == 0:
            print("  AVISO: Nenhum evento positivo na janela de teste. Recall será 0.")
            recalls_na_precisao_minima.append(0.0)
            continue

        modelo_fold = clone(modelo_template)
        try:
            modelo_fold.fit(X_treino_fold, y_treino_fold)
        except Exception as e:
            print(f"  ERRO no fit do Fold {num_janelas_executadas}: {e}. Pulando.")
            recalls_na_precisao_minima.append(0)
            continue
        
        probas_teste_fold = modelo_fold.predict_proba(X_teste_fold)[:, 1]

        recall_neste_fold = calcular_recall_em_precisao_minima(y_teste_fold, probas_teste_fold, precisao_minima_desejada)
        recalls_na_precisao_minima.append(recall_neste_fold)
        print(f"  Recall na Precisão >= {precisao_minima_desejada:.0%}: {recall_neste_fold:.3f}")

    if not recalls_na_precisao_minima:
        print("AVISO: Nenhuma janela de walk-forward foi processada com sucesso.")
        return {'recall_medio_na_prec_min': 0, 'recall_std_na_prec_min': 0, 'prob_sucesso_folds': 0}

    recall_medio = np.mean(recalls_na_precisao_minima)
    recall_std = np.std(recalls_na_precisao_minima)
    prob_sucesso_folds = np.mean([1 if r > 0 else 0 for r in recalls_na_precisao_minima])

    print(f"\nResultado Final da Avaliação ({num_janelas_executadas} folds):")
    print(f"  Recall Médio (com Precisão >= {precisao_minima_desejada:.0%}): {recall_medio:.3f}")
    print(f"  Desvio Padrão do Recall: {recall_std:.3f}")
    print(f"  Percentual de Folds com Sinal Viável: {prob_sucesso_folds:.0%}")

    resultado = {
        'recall_medio_na_prec_min': recall_medio,
        'recall_std_na_prec_min': recall_std,
        'prob_sucesso_folds': prob_sucesso_folds
    }
    return resultado

def avaliar_papel_em_diferentes_tamanhos_conjunto_treino(modelo_template, X_train, y_train, precisao_minima_desejada=0.50):

    # Passo 2: Teste Incrementalmente o Tamanho da Janela de Treino
    tamanhos_de_janela_para_testar = [120, 180, 250, 375, 500]
    resultados_por_janela = []
    for tamanho in tamanhos_de_janela_para_testar:
        metricas = avaliar_estrategia_walk_forward_rolante(
            modelo_template=modelo_template,
            X_super_treino=X_train,
            y_super_treino=y_train,
            tamanho_janela_treino=tamanho,
            tamanho_janela_validacao=30,
            passo_wf=30,
            precisao_minima_desejada=precisao_minima_desejada
        )
        metricas['tamanho_janela_treino'] = tamanho
        resultados_por_janela.append(metricas)
    df_resultados_janelas = pd.DataFrame(resultados_por_janela)
    return df_resultados_janelas.sort_values(by='recall_medio_na_prec_min', ascending=False)


from sklearn.metrics import make_scorer, average_precision_score, f1_score # PR AUC é bom para desbalanceados

def otimizar_hiperparametros_logreg(X_super_treino
                                    , y_super_treino
                                    , n_splits_cv=5
                                    , precisao_minima_para_otimizacao=0.50
                                    , pipeline_template=None):
    """
    Encontra os melhores hiperparâmetros para a Regressão Logística usando
    GridSearchCV com TimeSeriesSplit.

    Args:
        X_super_treino: Seu DataFrame de "super treino" (ex: os 510 dias).
        y_super_treino: Seus alvos correspondentes.
        n_splits_cv (int): Número de folds para a validação cruzada temporal.

    Returns:
        O melhor modelo treinado encontrado pelo GridSearchCV.
    """
    print(f"\n--- Iniciando Otimização de Hiperparâmetros com TimeSeriesSplit (n_splits={n_splits_cv}) ---")
    
    # 1. Definir o splitter de validação cruzada para séries temporais
    tscv = TimeSeriesSplit(n_splits=n_splits_cv)

    # 2. Definir o modelo base (não treinado)
    # Aumentar max_iter para garantir que solvers como 'saga' e 'lbfgs' convirjam.
    class_weight = 'balanced'
    

    if pipeline_template is not None:
        # Se um pipeline foi fornecido, usar ele como modelo base
        log_reg_template = pipeline_template
    else:
        log_reg_template = LogisticRegression(random_state=42, max_iter=5000, class_weight=class_weight)
        
    # 3. Definir o "grid" de parâmetros a serem testados
    param_grid = [
        {
            'modelo__penalty': ['l1'], # Penalidade Lasso (seleção de features)
            'modelo__solver': ['liblinear', 'saga'],
            'modelo__C': np.logspace(-4, 1, 6) # Isso testa: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]                                        
        },
        {
            'modelo__penalty': ['l2'], # Penalidade Ridge (encolhimento de features)
            'modelo__solver': ['liblinear', 'lbfgs', 'saga'],
            'modelo__C': np.logspace(-4, 1, 6) # Isso testa: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]                                        
        }
    ]

    # 4. Definir a métrica de scoring
    # 'roc_auc' é bom, mas 'average_precision' (PR AUC) é muitas vezes melhor para
    # dados desbalanceados e quando a precisão na classe positiva é importante.
    #scorer = make_scorer(average_precision_score)
    scorer = make_scorer(f1_score)

    #custom_scorer = make_scorer(
    #    recall_at_min_precision_scorer,
    ###    needs_proba=True,
    #    greater_is_better=True, # Queremos maximizar este score
    #    min_precision=precisao_minima_para_otimizacao # Passa o parâmetro para a nossa função
    #)

    # 5. Configurar e executar o GridSearchCV
    grid_search = GridSearchCV(
        estimator=log_reg_template,
        param_grid=param_grid,
        scoring=scorer,
        cv=tscv, # Usar o splitter temporal
        n_jobs=-1, # Usar todos os cores da CPU
        verbose=1  # Mostrar progresso
    )

    grid_search.fit(X_super_treino, y_super_treino)

    print("\n--- Resultados da Otimização ---")
    print(f"Melhor Score ({scorer}): {grid_search.best_score_:.4f}")
    print("Melhores Parâmetros Encontrados:")
    print(grid_search.best_params_)

    # O grid_search.best_estimator_ já é o modelo treinado com os melhores parâmetros
    # em todo o X_super_treino.
    return grid_search.best_estimator_