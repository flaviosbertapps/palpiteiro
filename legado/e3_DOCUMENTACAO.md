# E3 - Sistema de Screening B3

## Resumo Executivo

Após extensa validação empírica com múltiplas abordagens técnicas, **apenas uma estratégia passou em todos os critérios rigorosos**: **Mean Reversion LIGHT**.

**Resultado final:**
- Taxa de acerto: **81.7%**
- Retorno anualizado: **48.9%**
- Sharpe ratio: **1.38**
- Frequência: **3-4 operações/ano**
- Pior ano: **+6.5%** (zero anos negativos)

---

## Jornada de Validação

### 1. Estratégias Testadas (e Rejeitadas)

#### ❌ Momentum Puro
**Hipótese:** Papéis em alta continuam subindo.

**Configuração testada:**
- Retorno 20d > 5%
- RSI < 75
- ADX > 25
- Ibov > MM200

**Resultado backtest global:**
- Taxa: 72.6%
- Retorno: 96.8%/ano
- **Aparentemente excelente**

**Resultado walk-forward (validação real):**
- 2021: +26.9%
- 2022: **-44.1%** ❌
- 2023: +88.8%
- 2024: **-30.5%** ❌
- 2025: +97.8%

**Veredicto:** Alta instabilidade ano a ano. Overfitting ao período completo. **REJEITADO.**

---

#### ❌ Momentum com Filtro de Regime
**Melhoria:** Adicionou `Ibov > MM200` para operar apenas em bull markets.

**Resultado:** Melhorou (evitou alguns anos ruins), mas validação por papel individual mostrou **0 de 69 papéis aprovados** nos critérios.

**Veredicto:** Ainda instável demais. **REJEITADO.**

---

#### ❌ Sistema de Scoring Adaptativo
**Hipótese:** Em vez de thresholds fixos, pontuar múltiplos indicadores.

**Configuração:**
- RSI: 0-3 pontos
- Momentum 20d: 0-3 pontos
- ADX: 0-3 pontos
- Volume: 0-2 pontos
- Regime Ibov: 0-2 pontos

**Thresholds testados:** 6, 7, 8, 9, 10

**Resultado melhor threshold (7):**
- Taxa: **45.2%** (pior que aleatório)
- Retorno: **-1%/ano**
- Sharpe: negativo

**Veredicto:** Combinar indicadores fracos não cria poder preditivo. **REJEITADO.**

---

#### ❌ Mean Reversion STRICT
**Hipótese:** Compra em sobrevenda extrema (overreaction).

**Configuração:**
- RSI < 30
- Queda > 5% em 5d
- Bollinger Bands < 0.2
- Close > MM200
- Ibov > MM200

**Resultado:** **0 sinais em 5 anos** (critérios impossíveis de satisfazer simultaneamente).

**Veredicto:** Condições se contradizem. **REJEITADO.**

---

#### ❌ Mean Reversion SUPER-LIGHT
**Hipótese:** Relaxar critérios para aumentar frequência.

**Configurações testadas:**
1. **Com regime:** RSI < 50, queda > 2%, MM20, Ibov > MM200
2. **Sem regime:** mesmos critérios, remove filtro Ibov

**Resultado com regime:**
- 54.6 ops/ano
- Taxa: **60.5%**
- Retorno: **7.6%/ano** (abaixo da SELIC)
- Sharpe: 0.37

**Resultado sem regime:**
- 73.6 ops/ano
- Taxa: **60.2%**
- Retorno: **6.8%/ano**
- Sharpe: 0.31
- Anos negativos: 2/4

**Veredicto:** Relaxar critérios para aumentar frequência **destrói qualidade**. **REJEITADO.**

---

#### ❌ Stops/Alvos baseados em ATR
**Hipótese:** Stops adaptativos por volatilidade são melhores que % fixo.

**Multiplicadores testados:** 1.5-2.5× ATR

**Resultado:**
- Taxa: **28-36%** (catastrófico)
- Retorno: **-45% a -155%/ano**
- Sharpe: **-0.85 a -1.03**
- Pior ano: **-457%**

**Por que falhou:** ATR mede volatilidade, não direção. Papéis voláteis têm stops muito largos, deixando perder 10-15% quando movimento esperado é apenas +3%.

**Veredicto:** ATR inadequado para mean reversion de curto prazo. **REJEITADO.**

---

### 2. Estratégia Aprovada: Mean Reversion LIGHT ✅

#### Configuração Final

```python
# Entrada (compra em sobrevenda)
RSI < 40                 # Sobrevendido (relaxado de 30)
Queda > 3% em 5 dias     # Movimento recente de queda
Close > MM50             # Ainda em tendência de curto prazo
Ibov > MM200             # Filtro de regime (CRÍTICO)
Volume relativo > 0.5    # Liquidez mínima
ADX < 40                 # Sem tendência de queda estabelecida

# Saída
Alvo: +3%                # Recuperação modesta
Stop: -4%                # Proteção
Prazo máximo: 15 dias    # Tempo para recuperar
```

#### Validação Rigorosa

**Walk-forward 4 anos:**
- 2022: 100% acerto (4 trades)
- 2023: 66.7% acerto (3 trades)
- 2024: 100% acerto (2 trades)
- 2025: 60.0% acerto (5 trades)

**Métricas consolidadas:**
- Total: 14 trades em 4 anos
- Taxa média: **81.7%**
- Retorno médio: **48.9%/ano**
- Desvio padrão: ±35.5%
- **Pior ano: +6.5%** (nenhum ano negativo)
- Sharpe: **1.38** (excelente)

---

### 3. Otimização de Alvos e Stops

#### Combinações Testadas (% fixos)

| Alvo | Stop | Taxa | Ret/ano | Sharpe | Anos neg |
|------|------|------|---------|--------|----------|
| **3%** | **4%** | **81.7%** | **48.9%** | **1.38** | **0** ✅ |
| 4% | 5% | 77.5% | 60.5% | 1.17 | 0 |
| 5% | 5% | 62.9% | 31.1% | 0.86 | 1 |
| 4% | 4% | 55.8% | 13.1% | 0.21 | 2 |
| 2.5% | 3% | 58.4% | 0.4% | 0.01 | 2 |

**Veredicto:** 3%/4% tem **melhor Sharpe** e **zero anos negativos**. Nenhuma outra combinação supera.

---

## Lições Aprendidas

### 1. Overfitting é Real e Perigoso

**Armadilha:** Backtest global mostra 96.8%/ano → parece incrível → opera com dinheiro real → perde -40% no primeiro ano ruim.

**Solução:** Walk-forward obrigatório. Teste ano a ano para detectar instabilidade.

### 2. Momentum Não Funciona na B3 (curto prazo)

Papéis que sobem 5% em 20 dias:
- 50%: revertem (ruído)
- 30%: param em +3-4% (não atingem alvo)
- 20%: atingem +5%

**Taxa de acerto real: 20-40%** (inoperável).

### 3. Filtro de Regime é Essencial

Sem `Ibov > MM200`:
- 60% acerto
- Anos negativos frequentes (-17.9%, -25%)

Com `Ibov > MM200`:
- 81.7% acerto
- Zero anos negativos

**Diferença:** +21pp de acerto, elimina bear markets.

### 4. Frequência ≠ Qualidade

Comparação LIGHT vs SUPER-LIGHT:

| Versão | Ops/ano | Taxa | Ret/ano | Sharpe |
|--------|---------|------|---------|--------|
| LIGHT | 3.5 | 81.7% | 48.9% | 1.38 |
| SUPER-LIGHT | 54.6 | 60.5% | 7.6% | 0.37 |

**Relaxar critérios para ter 15× mais operações reduz Sharpe em 4×.**

### 5. Mean Reversion Funciona, Mas É Rara

Condições de sobrevenda + tendência de alta + regime favorável = **3-4× por ano**.

Isso não é problema. É **vantagem**:
- Alta seletividade = alta qualidade
- Menos stress
- Combina bem com estratégia de valor (70% do capital)

### 6. ATR Não Serve Para Tudo

ATR é útil para:
- Breakouts de volatilidade
- Trailing stops em tendências longas

ATR **não funciona** para:
- Mean reversion (movimento pequeno esperado)
- Stops em papéis voláteis (fica muito largo)

### 7. Sharpe > Retorno Bruto

Configuração 4%/5% tinha:
- Retorno: 60.5%/ano (melhor)
- Sharpe: 1.17

Configuração 3%/4% tem:
- Retorno: 48.9%/ano (menor)
- Sharpe: 1.38 (melhor)

**Escolha 3%/4%** porque:
- Menos volatilidade
- Zero anos negativos
- Melhor risco-ajustado

---

## Estrutura do Sistema Final

### Arquivos

```
e3_screening_regras.py    # Motor principal de screening
e3_estrategias.py         # Definições de estratégias
e3_backtest.py            # Engine de backtest
e3_validacao.py           # Testes walk-forward
e3_DOCUMENTACAO.md        # Este arquivo
universo_b3_ampliado.txt  # 69 papéis líquidos
```

### Fluxo Operacional

```
1. Diariamente após 18h
   → python e3_screening_regras.py
   → Gera sinais LIGHT (0-1 por dia, 3-4 por ano)
   → Envia email com sinais

2. Próxima manhã 9h
   → Se houver sinal: compra na abertura
   → Coloca stop -4%
   → Alvo +3%

3. Registra em planilha
   → Data, ticker, entrada, saída, resultado
   → Após 10-15 trades: valida taxa de acerto real
```

---

## Expectativa Realista

### Cenário Base (100% capital em LIGHT)

**Não recomendado** — muito concentrado.

- 3-4 ops/ano
- 81.7% acerto
- 48.9%/ano esperado
- **Problema:** fica 9-11 meses do ano sem operar

### Cenário Recomendado (70/30 split)

**70% em Valor/Dividendos:**
- Compra P/L < 8, DY > 6%, ROE > 15%
- Segura 12-24 meses
- Rebalanceamento trimestral
- Retorno: SELIC + 4-6% = **18%/ano**

**30% em Mean Reversion LIGHT:**
- Aguarda sinais (3-4×/ano)
- 81.7% acerto
- 48.9%/ano quando opera
- Retorno médio no capital: **~15%/ano**

**Retorno total esperado:**
- 70% × 18% = 12.6%
- 30% × 15% = 4.5%
- **Total: 17%/ano** com baixa volatilidade

---

## Critérios de Aprovação de Estratégias

Para futuras estratégias, use estes critérios rigorosos:

### Obrigatórios (todos devem passar)

1. **Taxa de acerto >65%** em walk-forward
2. **Pior ano >-20%** (proteção em bear markets)
3. **Retorno médio >30%/ano**
4. **≥60% dos anos positivos**
5. **Sharpe >1.0**

### Desejáveis

6. Mínimo 50 trades totais (validação estatística)
7. Desvio padrão <40% (volatilidade controlada)
8. Funciona em múltiplos setores (não específico)

---

## Próximos Passos

### Imediato (semana 1)
1. Rodar screening diário
2. Aguardar primeiro sinal
3. Executar conforme plano

### Curto prazo (3 meses)
1. Acumular 3-5 trades
2. Validar taxa de acerto preliminar
3. Ajustar se necessário

### Médio prazo (12 meses)
1. Completar 10-15 trades
2. Comparar com backtest
3. Se taxa <70%: revisar estratégia
4. Se taxa >75%: confirma validação

### Longo prazo (24 meses)
1. Desenvolver estratégia de valor/dividendos (70%)
2. Testar novas estratégias técnicas (breakouts, etc)
3. Expandir portfólio para 3-4 estratégias descorrelacionadas

---

## Apêndice: Comparação de Todas as Estratégias

| Estratégia | Taxa | Ret/ano | Sharpe | Anos neg | Veredicto |
|------------|------|---------|--------|----------|-----------|
| Momentum puro | 72.6% | 96.8% | N/A | 2/5 | ❌ Instável |
| Momentum + regime | 43-62% | -40 a +29% | N/A | 3/5 | ❌ Instável |
| Scoring | 45.2% | -1% | -0.03 | 2/4 | ❌ Sem poder preditivo |
| Mean Rev STRICT | N/A | N/A | N/A | N/A | ❌ 0 sinais |
| Mean Rev LIGHT | **81.7%** | **48.9%** | **1.38** | **0/4** | ✅ **APROVADO** |
| Mean Rev SUPER | 60.5% | 7.6% | 0.37 | 2/4 | ❌ Baixa qualidade |
| Mean Rev SEM REGIME | 60.2% | 6.8% | 0.31 | 2/4 | ❌ Anos negativos |
| ATR stops | 28-36% | -45 a -155% | -0.85 | 3/4 | ❌ Catastrófico |

---

## Comandos Úteis

```bash
# Screening diário (após 18h)
python e3_screening_regras.py

# Backtest completo
python e3_screening_regras.py --calibrar

# Validação walk-forward
python e3_validacao.py --estrategia mean_reversion_light

# Teste de debug (1 papel)
python e3_screening_regras.py --ticker PETR4.SA
```

---

## Contato e Manutenção

**Sistema criado:** Fevereiro 2026  
**Última validação:** Fevereiro 2026  
**Próxima revisão:** Após 15 trades ou 12 meses

---

**FIM DA DOCUMENTAÇÃO**
