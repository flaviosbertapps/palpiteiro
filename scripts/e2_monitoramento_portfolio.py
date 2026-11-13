import pandas as pd
import preprocessamento as pp
import utils
from datetime import datetime
import os

# --- CONFIGURAÇÕES ---
POSICOES_ATUAIS_PATH = 'dados/posicoes_abertas.csv'
ATR_MULTIPLIER_EXIT = 2.0 # Distância do Trailing Stop

def monitorar_portfolio():
    print(f"\n--- 🛡️ MONITORAMENTO DE PORTFÓLIO - {datetime.now().strftime('%d/%m/%Y')} ---")
    
    if not os.path.exists(POSICOES_ATUAIS_PATH):
        print("📭 Nenhuma posição aberta para monitorar.")
        return

    posicoes = pd.read_csv(POSICOES_ATUAIS_PATH)
    alertas_venda = []
    ajustes_stop = []
    indices_para_remover = []

    base = datetime.today().date()

    for index, row in posicoes.iterrows():
        ativo = row['Ativo']
        preco_entrada = row['Preco_Entrada']
        stop_anterior = row['Stop_Atual']
        estrategia = row['Estrategia']
        
        print(f"🔍 Analisando {ativo}...")
        
        ticker_fmt = f"{ativo}.SA" if ".SA" not in ativo else ativo
        df = pp.montar_dataset(ticker_fmt, dias_total_historico=100, data_base=base)
        df = pp.adicionar_features_sniper(df)
        
        if df is None or df.empty:
            print(f"⚠️ Erro ao baixar dados de {ativo}. Pulando...")
            continue
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        preco_agora = last['adj_close']
        atr = last['atr']        

        # --- 1. GATILHOS DE SAÍDA TÉCNICA OU STOP ---
        venda_urgente = False
        motivo = ""

        # A. Verificação de Stop (Crítico)
        if preco_agora <= stop_anterior:
            venda_urgente = True
            motivo = "Stop Atingido (Preço <= Stop)"

        # B. Saídas por Estratégia
        elif estrategia == 'Tendência' and last['mm20'] < prev['mm20']:
            venda_urgente = True
            motivo = "Fim de Tendência (MM20 virou)"
            
        elif estrategia == 'Momentum' and preco_agora < last['mm5']:
            venda_urgente = True
            motivo = "Perda de Impulso (Abaixo da MM5)"

        elif estrategia == 'IFR2' and last['ifr2'] > 75:
            venda_urgente = True
            motivo = "Alvo Alcançado (IFR2 > 75)"

        # --- 3. REGISTRO DE ALERTAS E REMOÇÃO ---
        if venda_urgente:
            print(f"🚨 ALERTA DE VENDA: {ativo} por {motivo}")
            lucro_percent = ((preco_agora / preco_entrada) - 1) * 100
            alertas_venda.append({
                'Ativo': ativo,
                'Motivo': motivo,
                'Preço Entrada': preco_entrada,
                'Preço Atual': preco_agora,
                'Stop Original': stop_anterior,
                'Lucro/Preju': f"{lucro_percent:.2f}%"
            })
            indices_para_remover.append(index)
        else:
            # --- 1. LÓGICA DE TRAILING STOP --- Só é realizada se não for recomendada venda do papel
            # O stop só sobe. Usamos a máxima do dia anterior para projetar o stop.
            novo_stop_potencial = round(last['high'] - (atr * ATR_MULTIPLIER_EXIT), 2)
            
            if novo_stop_potencial > stop_anterior:
                ajustes_stop.append({
                    'Ativo': ativo,
                    'Stop Anterior': stop_anterior,
                    'Novo Stop': novo_stop_potencial,
                    'Preço Atual': preco_agora
                })
                posicoes.at[index, 'Stop_Atual'] = novo_stop_potencial
                stop_vigente = novo_stop_potencial
            else:
                stop_vigente = stop_anterior


    # --- 4. FINALIZAÇÃO E ENVIO DE E-MAILS ---
    
    # Remove as posições vendidas do portfólio
    if indices_para_remover:
        posicoes = posicoes.drop(indices_para_remover)
    
    # Salva o arquivo atualizado (com novos stops e sem as vendas)
    posicoes.to_csv(POSICOES_ATUAIS_PATH, index=False)

    # E-mail de Vendas (Prioridade Máxima)
    if alertas_venda:
        enviar_email_saida(alertas_venda)
    
    # E-mail de Ajustes de Stop (Informativo)
    if ajustes_stop:
        enviar_email_stops(ajustes_stop)

    if not alertas_venda and not ajustes_stop:
        print("✅ Tudo sob controle. Sem vendas ou ajustes necessários hoje.")

def enviar_email_saida(alertas):
    df_venda = pd.DataFrame(alertas)
    assunto = f"⚠️ [AÇÃO NECESSÁRIA] Venda B3 - {datetime.now().strftime('%d/%m')}"
    corpo = f"<h3>O robô identificou gatilhos de saída:</h3>{df_venda.to_html(index=False)}"
    utils.enviar_email_gmail_com_app_password(assunto, corpo_html=corpo)

def enviar_email_stops(ajustes):
    df_stops = pd.DataFrame(ajustes)
    assunto = f"⬆️ [Ajuste de Stop] Portfólio Protegido - {datetime.now().strftime('%d/%m')}"
    corpo = f"<h3>O Trailing Stop de alguns ativos subiu:</h3>{df_stops.to_html(index=False)}"
    utils.enviar_email_gmail_com_app_password(assunto, corpo_html=corpo)

if __name__ == "__main__":
    monitorar_portfolio()