# 🔍 ANÁLISE DE CORRELAÇÕES - Guia Completo

Sistema completo para análise de correlações entre:
- Papéis brasileiros
- Câmbio (USD/BRL, EUR/USD)
- Índices globais (S&P 500, Ibovespa)
- Commodities (WTI - petróleo)
- Indicadores de risco (VIX, DXY)

---

## 📦 INSTALAÇÃO

```bash
pip install yfinance pandas numpy matplotlib seaborn --break-system-packages
```

---

## 🚀 COMO USAR

### 1. Análise Completa (Matriz de Correlações)

Gera matriz de correlação entre TODOS os 27 papéis líquidos:

```bash
python analise_correlacoes_cli.py --analise-completa
```

**Output:**
- `resultados_correlacoes/correlacao_papeis.csv` - Matriz completa
- `resultados_correlacoes/heatmap_papeis.png` - Visualização
- `resultados_correlacoes/top_pares_correlacionados.csv` - Top 10 pares

**Tempo:** ~2-3 minutos

---

### 2. Papel vs Variáveis Macro

Analisa correlação de UM papel específico vs TODAS as variáveis macro:

```bash
# Exemplos:
python analise_correlacoes_cli.py --papel-vs-macro PETR4
python analise_correlacoes_cli.py --papel-vs-macro VALE3
python analise_correlacoes_cli.py --papel-vs-macro ITUB4
```

**Testa:**
- USD/BRL (dólar)
- EUR/USD
- DXY (índice do dólar)
- S&P 500
- VIX (medo)
- WTI (petróleo)
- Ibovespa

**Com e SEM lag (0 a 5 dias)**

**Output:**
```
CORRELAÇÕES (sem lag)
=====================
usd_brl        : +0.682
sp500          : +0.445
wti            : +0.523
vix            : -0.312
...

CORRELAÇÕES COM LAG (macro antecipa papel)
==========================================
sp500:
  Lag 1d: +0.512
  Lag 2d: +0.378
  Lag 3d: +0.201
...

TOP 5 CORRELAÇÕES (com lag)
============================
   variavel  correlacao  lag
    wti          0.682    0
    usd_brl      0.645    1
    sp500        0.512    1
```

**Tempo:** ~30 segundos

---

### 3. Liderança S&P 500 → Ibovespa

Testa se S&P 500 ANTECIPA movimentos do Ibovespa:

```bash
python analise_correlacoes_cli.py --lideranca-sp500
```

**Hipótese:**
- S&P 500 fecha ANTES do Brasil (horário)
- S&P cai forte hoje → Ibov cai amanhã?

**Output:**
```
Correlação S&P 500 hoje vs Ibov em t+N:

  Lag 0d (Ibov hoje): +0.612
  Lag 1d (Ibov daqui a 1d): +0.687
  Lag 2d (Ibov daqui a 2d): +0.423
  ...

🏆 Melhor correlação:
   Lag 1 dias: +0.687

💡 HIPÓTESE VALIDADA!
   S&P 500 ANTECIPA Ibov em 1 dia(s)
   Correlação: 68.7%

   Estratégia possível:
   → S&P cai forte hoje → Short Ibov amanhã
```

**Tempo:** ~20 segundos

---

### 4. Detectar Oportunidades de Pairs Trading

Encontra pares altamente correlacionados que estão DESCORRELACIONADOS no momento:

```bash
python analise_correlacoes_cli.py --detectar-pairs
```

**Lógica:**
1. Busca pares com correlação histórica > 0.70
2. Verifica se nos últimos 5 dias houve descorrelação anormal
3. Calcula Z-score do spread
4. Se |Z| > 1.5 → Oportunidade!

**Output:**
```
DESCORRELAÇÕES RECENTES (últimos 5 dias)
=========================================

ticker1  ticker2  ret1_5d  ret2_5d  z_score              acao
  ITUB4    BBDC4   -0.030   -0.005    -2.34  Long ITUB4 / Short BBDC4
  PETR4    VALE3   +0.025   +0.008    +1.87  Short PETR4 / Long VALE3
  WEGE3    SUZB3   -0.015   +0.002    -1.62  Long WEGE3 / Short SUZB3

✓ 3 oportunidades detectadas!
```

**Interpretação:**
- ITUB4 caiu 3% mas BBDC4 caiu só 0.5%
- Correlação histórica = 0.88 (andam juntos)
- Z = -2.34 (muito anormal)
- **Trade:** Long ITUB4 / Short BBDC4 (convergência esperada)

**Tempo:** ~1-2 minutos

---

## 📊 VARIÁVEIS MACRO DISPONÍVEIS

| Variável | Ticker (Yahoo) | Descrição |
|----------|----------------|-----------|
| **usd_brl** | BRL=X | Dólar vs Real |
| **eur_usd** | EURUSD=X | Euro vs Dólar |
| **dxy** | DX-Y.NYB | Dollar Index (força do USD) |
| **sp500** | ^GSPC | S&P 500 |
| **vix** | ^VIX | Índice de Volatilidade (medo) |
| **wti** | CL=F | Petróleo WTI |
| **ibov** | ^BVSP | Ibovespa |

---

## 💡 HIPÓTESES TESTÁVEIS

### 1. Exportadores ganham com Dólar Alto

**Hipótese:** PETR4, VALE3 têm correlação POSITIVA com USD/BRL

**Teste:**
```bash
python analise_correlacoes_cli.py --papel-vs-macro PETR4
python analise_correlacoes_cli.py --papel-vs-macro VALE3
```

**Esperado:** Correlação > +0.60

**Trade:** Se USD/BRL sobe 2% mas PETR4 não subiu → Long PETR4 (convergência)

---

### 2. S&P Antecipa Brasil com 1 dia

**Hipótese:** S&P fecha antes → Antecipa Ibov

**Teste:**
```bash
python analise_correlacoes_cli.py --lideranca-sp500
```

**Esperado:** Correlação lag 1 dia > 0.60

**Trade:** S&P cai 2%+ hoje → Short Ibov amanhã

---

### 3. VIX Alto = Ibov Cai

**Hipótese:** VIX (medo) antecipa quedas globais

**Teste:**
```bash
python analise_correlacoes_cli.py --papel-vs-macro IBOV  # (se implementarmos)
```

**Esperado:** Correlação VIX vs Ibov NEGATIVA (-0.50)

**Trade:** VIX sobe 15%+ hoje → Short Ibov amanhã

---

### 4. Pairs Trading - ITUB4 vs BBDC4

**Hipótese:** Bancos andam juntos (correlação ~0.85)

**Teste:**
```bash
python analise_correlacoes_cli.py --detectar-pairs
```

**Trade:** Quando spread > 2 std devs → Convergência

---

## 📁 ESTRUTURA DE SAÍDA

```
resultados_correlacoes/
├── correlacao_papeis.csv              # Matriz completa
├── heatmap_papeis.png                 # Visualização
├── top_pares_correlacionados.csv     # Top 10 pares
│
├── PETR4/
│   └── correlacoes_macro.csv          # PETR4 vs todas variáveis
│
├── VALE3/
│   └── correlacoes_macro.csv
│
├── lideranca/
│   └── sp500_ibov_lag.csv             # S&P → Ibov lag analysis
│
└── oportunidades/
    └── pairs_descorrelacionados.csv   # Opportunities agora
```

---

## 🎯 ESTRATÉGIAS DERIVADAS

### A) **Mean Reversion via Correlação**

Se PETR4 tem correlação 0.75 com USD/BRL:
- USD sobe 2%, PETR4 cai 1%
- Descorrelação = Oportunidade
- **Trade:** Long PETR4 (vai convergir)

---

### B) **Pairs Trading**

Se ITUB4 e BBDC4 têm correlação 0.88:
- ITUB4 cai 3%, BBDC4 cai 0.5%
- Spread anormal
- **Trade:** Long ITUB4 / Short BBDC4

**Tamanho das pernas:**
```python
# Capital: R$ 100k
# Beta-neutral:
# BBDC4 tem beta 1.2 vs ITUB4

Comprar ITUB4: R$ 60k
Vender BBDC4: R$ 50k  # Ajustado pelo beta
```

---

### C) **Antecipação via S&P**

Se correlação S&P → Ibov (lag 1d) = 0.68:
- S&P cai 2.5% hoje
- **Trade:** Short Ibov no fechamento (ou EWZ)
- **Stop:** +1% no dia seguinte
- **Alvo:** -1.5% no dia seguinte

---

### D) **Risk-Off via VIX**

Se VIX sobe 15%+ em um dia:
- **Trade:** Short emergentes (EWZ, Ibov)
- **Holding:** 2-3 dias
- **Stop:** +2%

---

## ⚙️ OPÇÕES AVANÇADAS

### Mudar período de análise:

```bash
# Últimos 5 anos
python analise_correlacoes_cli.py --analise-completa --dias 1825

# Último 1 ano (mais sensível)
python analise_correlacoes_cli.py --analise-completa --dias 365
```

---

## 🔧 TROUBLESHOOTING

### Erro: "No module named 'yfinance'"
```bash
pip install yfinance --break-system-packages
```

### Erro: "Empty DataFrame"
- Yahoo Finance pode estar fora do ar
- Ticker pode estar errado (use `.SA` apenas internamente)
- Período pode não ter dados

### Dados parciais
- Alguns ativos podem não ter dados completos
- Sistema continua com dados disponíveis

---

## 📈 PRÓXIMOS PASSOS

### Após identificar correlações fortes:

1. **Validação histórica**
   - Testar estratégia em backtest
   - Walk-forward (evitar overfitting)

2. **Paper trading**
   - 1 mês seguindo sinais sem dinheiro real
   - Validar na prática

3. **Capital pequeno**
   - Começar com R$ 20-30k
   - 2-3 meses de validação

---

## 🎬 EXEMPLO COMPLETO

```bash
# 1. Análise geral
python analise_correlacoes_cli.py --analise-completa

# 2. Focar em PETR4
python analise_correlacoes_cli.py --papel-vs-macro PETR4

# 3. Testar liderança S&P
python analise_correlacoes_cli.py --lideranca-sp500

# 4. Buscar oportunidades HOJE
python analise_correlacoes_cli.py --detectar-pairs
```

**Tempo total:** ~5 minutos

**Resultado:** Você terá:
- Matriz de correlações de todos os papéis
- Correlações de PETR4 vs macro (com lag)
- Validação se S&P antecipa Ibov
- Oportunidades de pairs trading HOJE

---

## 📝 NOTAS IMPORTANTES

✅ **Sem lookahead bias** - Usa apenas dados até data_fim
✅ **Yahoo Finance gratuito** - Não precisa API paga
✅ **Dados globais 24/7** - Moedas/índices sempre disponíveis
⚠️ **Correlação ≠ Causalidade** - Sempre validar hipótese
⚠️ **Correlações mudam** - Recalcular mensalmente

---

**Pronto para descobrir relações ocultas no mercado!** 🚀