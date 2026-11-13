import numpy as np

def calculate_position_size(account_balance, risk_pct, entry_price, stop_loss_price, ticker_volatility=None):
    """
    Calcula o tamanho da posição baseado no risco financeiro fixo.
    
    Parâmetros:
    - account_balance: Capital total disponível (ex: 100000)
    - risk_pct: Quanto do capital você aceita perder SE DER ERRADO (ex: 0.01 para 1%)
    - entry_price: Preço de entrada
    - stop_loss_price: Preço do Stop Inicial
    
    Retorna:
    - quantity: Quantidade de ações a comprar (arredondado para lote padrão de 100 se possível)
    - financial_exposure: Quanto dinheiro será alocado na operação
    """
    
    # 1. Quanto dinheiro posso perder nesta operação?
    # Ex: 100k * 1% = R$ 1.000 de risco
    risk_amount = account_balance * risk_pct
    
    # 2. Qual o risco por ação?
    # Ex: Entrada 10.00, Stop 9.00 -> Risco de R$ 1.00 por ação
    risk_per_share = entry_price - stop_loss_price
    
    if risk_per_share <= 0:
        return 0, 0 # Erro: Stop acima da entrada (em compra)
    
    # 3. Quantas ações posso comprar para que, se bater no stop, eu perca só o risk_amount?
    raw_quantity = risk_amount / risk_per_share
    
    # 4. Ajuste para Lote Padrão (B3 usa lotes de 100, ou fracionário)
    # Vamos arredondar para baixo para ser conservador
    quantity = int(raw_quantity // 100) * 100 
    
    # Se a quantidade for 0 (capital insuficiente para lote padrão), tenta fracionário
    if quantity == 0:
        quantity = int(raw_quantity)
        
    # 5. Trava de Segurança de Exposição (Max Position Size)
    # Nunca coloque mais de 25% do patrimônio em um único papel, mesmo que o stop seja curto.
    max_capital_allowed = account_balance * 0.25
    financial_exposure = quantity * entry_price
    
    if financial_exposure > max_capital_allowed:
        quantity = int(max_capital_allowed / entry_price)
        financial_exposure = quantity * entry_price

    return {
        "quantity": quantity,
        "financial_exposure": round(financial_exposure, 2),
        "risk_capital": round(quantity * risk_per_share, 2),
        "entry": entry_price,
        "stop": round(stop_loss_price, 2)
    }