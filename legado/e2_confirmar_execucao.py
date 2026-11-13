import pandas as pd
import os
from datetime import datetime

def confirmar_compras():
    caminho_pendentes = 'dados/ordens_pendentes.csv'
    caminho_posicoes = 'dados/posicoes_abertas.csv'
    
    if not os.path.exists(caminho_pendentes):
        print("❌ Nenhuma ordem pendente encontrada.")
        return

    df_pendentes = pd.read_csv(caminho_pendentes)
    
    print("\n--- 🛒 CONFIRMAÇÃO DE EXECUÇÃO ---")
    print("As seguintes ordens foram sugeridas:")
    print(df_pendentes[['Ativo', 'Estratégia', 'Qtd', 'Preço_Ref']])
    
    confirmar = input("\nVocê executou TODAS as ordens acima com os valores sugeridos? (s/n): ").lower()
    
    if confirmar == 's':
        df_executado = df_pendentes.copy()
    else:
        print("💡 Dica: Abra o arquivo 'dados/ordens_pendentes.csv', apague o que não comprou, ajuste o 'Preço_Ref' se necessário e salve.")
        input("Após ajustar e salvar o CSV, aperte ENTER para continuar...")
        df_executado = pd.read_csv(caminho_pendentes)

    # Preparar para o Portfólio Ativo
    df_executado['Data_Entrada'] = datetime.now().strftime('%Y-%m-%d')
    df_executado = df_executado.rename(columns={
        'Preço_Ref': 'Preco_Entrada',
        'Stop_Loss': 'Stop_Atual',
        'Estratégia': 'Estrategia'
    })
    
    colunas = ['Ativo', 'Estrategia', 'Preco_Entrada', 'Stop_Atual', 'Data_Entrada']
    df_final = df_executado[colunas]

    # Salvar nas Posições Abertas
    if not os.path.exists(caminho_posicoes):
        df_final.to_csv(caminho_posicoes, index=False)
    else:
        df_final.to_csv(caminho_posicoes, mode='a', header=False, index=False)
    
    # Limpar pendentes
    os.remove(caminho_pendentes)
    print(f"\n✅ {len(df_final)} ativos movidos para o seu Portfólio Ativo!")

if __name__ == "__main__":
    confirmar_compras()