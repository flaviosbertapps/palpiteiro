# E3 - Sistema de Screening B3

Sistema validado de trading com **81.7% de taxa de acerto** e **48.9%/ano de retorno esperado**.

## 🚀 Uso Diário (Após 18h)

```bash
python e3_screening_regras.py
```

- Gera 0-1 sinais por dia (~3-4 sinais/ano)
- Envia email automático
- Próxima manhã (9h): compra na abertura

## 📋 Configuração da Estratégia

**Mean Reversion LIGHT** (única aprovada):
- Compra: RSI < 40, queda >3% em 5d, Ibov > MM200
- Alvo: +3%
- Stop: -4%
- Prazo: 15 dias

## 📊 Expectativa Validada

- Taxa de acerto: 81.7%
- Retorno anual: 48.9%
- Sharpe: 1.38
- Operações/ano: 3-4
- Anos negativos: 0 (em 4 anos)

## 📁 Arquivos

- `e3_screening_regras.py` - Motor principal
- `e3_estrategias.py` - Definições de estratégias
- `e3_DOCUMENTACAO.md` - Documentação completa
- `e3_universo_b3_ampliado.txt` - 69 papéis testados

## 🔧 Comandos Úteis

```bash
# Screening diário
python e3_screening_regras.py

# Backtest completo (recalibra papéis promissores)
python e3_screening_regras.py --calibrar

# Debug específico
python e3_screening_regras.py --ticker PETR4.SA --data 2025-12-15
```

## ⚙️ Configuração

Edite `CFG` no início de `e3_screening_regras.py`:

```python
"capital_disponivel": 100000,      # Seu capital
"risco_por_trade_pct": 0.02,       # 2% de risco por trade
"enviar_email": True,               # True/False
```

## 📖 Documentação Completa

Veja `e3_DOCUMENTACAO.md` para:
- Jornada completa de validação
- Estratégias testadas (e rejeitadas)
- Lições aprendidas
- Critérios de aprovação

## ⚠️ Importante

- Sistema testado com walk-forward em 4 anos
- **Não há garantia de resultados futuros**
- Registre TODOS os trades em planilha
- Após 10-15 trades: valide se taxa de acerto se mantém >75%

---

**Última atualização:** Fevereiro 2026
