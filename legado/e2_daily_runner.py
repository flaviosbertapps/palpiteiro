import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import e2_engine_signals as es
import e2_risk_manager as rm
import preprocessamento as pp
import utils

# --- CONFIGURAÇÕES DE OPERAÇÃO ---
PORTFOLIO_PATH = 'dados/meu_portfolio.csv'
CAPITAL_TOTAL = 200000       # Seu capital atualizado
RISCO_POR_TRADE = 0.01       # 1%
ATR_MULTIPLIER_STOP = 2.0    # Mesmo padrão do backtest

# Mapeamento: Nome da String -> Função Real
# Se você criar novas estratégias, adicione aqui.
STRATEGY_MAP = {
    'Tendência': es.strategy_trend,
    'Breakout': es.strategy_breakout,
    'Retorno_Média': es.strategy_mean_reversion,
    'Momentum': es.strategy_momentum_swing,
    'IFR2': es.strategy_ifr2_quant
}

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
            return datetime.today().date()
    else:
        # Nenhum argumento de data fornecido, usa a data atual
        data_base_atual = datetime.today().date()
        print(f"INFO: Nenhum argumento de data fornecido. Executando para a data atual: {data_base_atual.strftime('%Y-%m-%d')}")
        return data_base_atual

def registrar_execucao_posicao(df_compras_confirmadas):
    """
    Salva as posições que foram realmente executadas no mercado.
    df_compras_confirmadas deve conter: Ativo, Estratégia, Preço_Ref, Stop_Loss
    """
    caminho_posicoes = 'dados/posicoes_abertas.csv'
    
    # Prepara os dados para o formato de monitoramento
    df_para_abrir = df_compras_confirmadas.copy()
    df_para_abrir['Data_Entrada'] = datetime.now().strftime('%Y-%m-%d')
    # Renomeamos para manter consistência com o monitor de saídas
    df_para_abrir = df_para_abrir.rename(columns={
        'Preço_Ref': 'Preco_Entrada',
        'Stop_Loss': 'Stop_Atual',
        'Estratégia': 'Estrategia'
    })
    
    # Selecionamos apenas o necessário para o monitoramento
    colunas_finais = ['Ativo', 'Estrategia', 'Preco_Entrada', 'Stop_Atual', 'Data_Entrada']
    df_para_abrir = df_para_abrir[colunas_finais]

    if not os.path.exists(caminho_posicoes):
        df_para_abrir.to_csv(caminho_posicoes, index=False)
    else:
        # Append sem repetir cabeçalho
        df_para_abrir.to_csv(caminho_posicoes, mode='a', header=False, index=False)
    
    print(f"✅ {len(df_para_abrir)} posições registradas em 'posicoes_abertas.csv'")


def gerar_dataframe_ordens(lista_sinais):
    rows = []
    for sinal in lista_sinais:
        # Formatação: Métrica: Valor (Condição Limite)
        racional = " | ".join([
            f"{k}: {v[0]:.2f} ({v[2]} {v[1]:.2f})" 
            for k, v in sinal['checklist'].items()
        ])
        
        rows.append({
            'Ativo': sinal['ativo'],
            'Estratégia': sinal['estrategia'],
            'Preço': f"R$ {sinal['preco_atual']:.2f}",
            'Stop': f"R$ {sinal['stop_loss']:.2f}",
            'Qtd': int(sinal['qtd']),
            'Financeiro': f"R$ {sinal['investimento']:,.2f}",
            'Risco Abs.': f"R$ {sinal['risco']:.2f}",
            'Critérios (Valor vs Limite)': racional
        })
    return pd.DataFrame(rows)


def preparar_e_enviar_relatorio(df_ordens):
    if df_ordens.empty:
        print("Nenhuma ordem gerada hoje.")
        return

    assunto = f"🚨 RELATÓRIO DE COMPRA B3 - {pd.Timestamp.now().strftime('%d/%m/%Y')}"
    
    # Estilização básica para o DataFrame ficar apresentável no Gmail
    estilo_html = """
    <style>
        table { border-collapse: collapse; width: 100%; font-family: sans-serif; }
        th { background-color: #f2f2f2; text-align: left; padding: 8px; border: 1px solid #ddd; }
        td { padding: 8px; border: 1px solid #ddd; font-size: 12px; }
        tr:nth-child(even) { background-color: #fafafa; }
        .buy-signal { color: #2ecc71; font-weight: bold; }
    </style>
    """
    
    corpo_html = f"""
    <html>
    <head>{estilo_html}</head>
    <body>
        <h2>Ordens Identificadas para Execução</h2>
        <p>O robô identificou as seguintes oportunidades baseadas nos critérios stress-testados:</p>
        {df_ordens.to_html(index=False, classes='table')}
        <br>
        <p><i>Este é um relatório automático. Verifique o book antes de executar.</i></p>
    </body>
    </html>
    """
    
    # Chamada da sua função existente
    utils.enviar_email_gmail_com_app_password(
        assunto=assunto,
        corpo_html=corpo_html
    )

def main():
    print(f"\n--- 🤖 ROBÔ DE EXECUÇÃO DIÁRIA - {datetime.now().strftime('%d/%m/%Y %H:%M')} ---")
    
    
    try:
        portfolio = pd.read_csv(PORTFOLIO_PATH)
        print(f"📂 Portfólio carregado: {len(portfolio)} ativos monitorados.")
    except FileNotFoundError:
        print("❌ Erro: 'meu_portfolio.csv' não encontrado. Rode o Screener primeiro.")
        return

    sinais_para_relatorio = []
    ordens_compra = []
    base_date = obter_data_base()


    print("\nVerificando sinais de fechamento...")
    
    for index, row in portfolio.iterrows():
        ativo = row['Ativo']
        nome_estrategia = row['Estratégia']
        
        # 1. Seleciona a função de estratégia correta
        if nome_estrategia not in STRATEGY_MAP:
            print(f"⚠️ Estratégia '{nome_estrategia}' não encontrada no mapa. Pulando {ativo}.")
            continue
            
        func_estrategia = STRATEGY_MAP[nome_estrategia]
        
        # 2. Baixa dados recentes (suficiente para calcular indicadores)
        # Precisamos de ~250 dias para garantir MM200 se a estratégia usar
        ticker_fmt = f"{ativo}.SA" if ".SA" not in ativo else ativo
        df = pp.montar_dataset(ticker_fmt, dias_total_historico=300, data_base=base_date)
        df = pp.adicionar_features_sniper(df)

        if df is None or df.empty:
            print(f"⚠️ Falha de dados para {ativo}.")
            continue            

        # 3. Aplica a Lógica da Estratégia
        # Isso cria a coluna 'entry_signal' baseada nos dados até hoje
        df_sinal, checklist = func_estrategia(df.copy())
        
        # 4. Verifica o ÚLTIMO dia (Hoje)
        # Se entry_signal do último candle for 1, temos compra para AMANHÃ na abertura
        last_row = df_sinal.iloc[-1]
        last_date = df_sinal.index[-1]
        
        # Verifica se o sinal é de hoje (compara datas para não pegar sinal velho se o download falhar)
        # Nota: yfinance as vezes retorna o dia atual com data 00:00:00
        hoje_str = datetime.now().strftime('%Y-%m-%d')
        data_sinal_str = last_date.strftime('%Y-%m-%d')
        
        # Opcional: Validar se a data do sinal é realmente hoje (ignorar se rodar fds)
        # if data_sinal_str != hoje_str: ... (cuidado com fuso horário)

        if last_row['entry_signal'] == 1:
            print(f"🔥 SINAL DETECTADO: {ativo} ({nome_estrategia})")
            
            # 5. Cálculo de Position Sizing
            preco_fechamento = last_row['adj_close']
            atr_atual = last_row['atr']
            stop_price = preco_fechamento - (atr_atual * ATR_MULTIPLIER_STOP)
            
            sizing = rm.calculate_position_size(
                CAPITAL_TOTAL, 
                RISCO_POR_TRADE, 
                preco_fechamento, 
                stop_price
            )
            
            ordem = {
                'Ativo': ativo, 'Estratégia': nome_estrategia, 'Qtd': sizing['quantity'],
                'Preço_Ref': preco_fechamento, 'Stop_Loss': round(stop_price, 2),
                'Investimento': sizing['financial_exposure'], 'Risco': sizing['risk_capital']
            }
            ordens_compra.append(ordem)

            sinais_para_relatorio.append({
                'ativo': ativo,
                'estrategia': nome_estrategia,
                'preco_atual': preco_fechamento,
                'stop_loss': round(stop_price, 2),
                'qtd': sizing['quantity'],
                'investimento': sizing['financial_exposure'],
                'risco': sizing['risk_capital'],
                'checklist': checklist
            })



        else:
            # Apenas log silencioso ou "." para mostrar progresso
            print(f". {ativo}: Sem sinal", end="\r")

    # --- RELATÓRIO DE ORDENS ---
    print("\n" + "="*50)
    if ordens_compra:
        df_ordens = pd.DataFrame(ordens_compra)
        
        # 1. Salva o histórico com data (Auditoria)
        arquivo_hist = f"dados/historico_sinais_{datetime.now().strftime('%Y%m%d')}.csv"
        df_ordens.to_csv(arquivo_hist, index=False)
        
        # 2. Salva o arquivo de trabalho (Pendente)
        # Este arquivo você vai abrir para conferir o que comprou
        df_ordens.to_csv('dados/ordens_pendentes.csv', index=False)
        
        print(f"\n📄 Sinais salvos em: {arquivo_hist}")
        print(f"📂 Arquivo de trabalho atualizado: dados/ordens_pendentes.csv")
        
        df_visual = gerar_dataframe_ordens(sinais_para_relatorio)
        preparar_e_enviar_relatorio(df_visual)

    else:
        print("😴 Nenhuma oportunidade de compra para amanhã. Durma tranquilo.")
    print("="*50)

if __name__ == "__main__":
    main()