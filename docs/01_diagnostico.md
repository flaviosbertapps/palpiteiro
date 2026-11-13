# Diagnóstico — o que foi feito, o que falhou e por quê

Revisão de novembro/2026 de todo o conteúdo de `legado/` (código, 35 notebooks, CSVs de resultado e o backup `e4_valor_holding_backup_20260414.zip`). Os números marcados com **[controle]** foram recalculados nesta revisão, com o código em `analises/2026-11_revisao/`; os demais vêm dos arquivos do próprio repositório.

## 1. Veredito

1. **Nenhuma das quatro frentes produziu evidência de vantagem estatística.** Isso inclui a única estratégia marcada como "✅ APROVADA" (Mean Reversion LIGHT).
2. **O problema principal não foi falta de ideias nem de calibração; foi o método de validação.** Com as métricas usadas, uma estratégia sem valor nenhum consegue parecer boa, e uma estratégia boa não conseguiria ser distinguida de sorte. Calibrar mais dentro desse método só produz mais falsos positivos.
3. **A meta "lucro modesto, porém provável" foi traduzida em "alta taxa de acerto", e essa tradução está errada.** Taxa de acerto é fabricada pela relação alvo/stop. Entradas aleatórias com alvo de 3% e stop de 4% acertam 58% das vezes e não ganham nada **[controle]**.
4. **A régua nunca entrou na conta.** No período dos dados (jan/2020 a fev/2026) o CDI rendeu cerca de 9,6% a.a. sem risco, mais que o Ibovespa (8,0% a.a., com queda máxima de 47%) **[controle]**. A Selic estava em 15% a.a. no último dado do repositório (mar/2026). Qualquer rotina de trading precisa bater isso depois de custos e imposto para valer o trabalho.

## 2. Linha do tempo

| Fase | Período | Abordagem | Onde está | Desfecho |
|---|---|---|---|---|
| **E1** | mai/2025 – jan/2026 | Classificador por papel (RandomForest, Regressão Logística, LightGBM, XGBoost) prevendo "sobe X% em N dias" | `_palpiteiro.py`, `preprocessamento.py`, `treinamento*.py`, `janelas_rolantes.py`, notebooks por papel, `experimentos*.csv` | AUC de validação ~0,54; precisão no hold-out igual ao ponto de equilíbrio |
| **E2** | fev/2026 | 5 regras técnicas (tendência, breakout, retorno à média, momentum MM5, IFR2), escolhendo a melhor regra por papel | `e2_*.py`, `e2_screening.ipynb` | A última execução do screener terminou com **zero** combinações aprovadas |
| **E3** | fev – mar/2026 | Screening por regras no universo líquido; grade de alvos/stops; momentum, scoring, swing, breakout, reversão à média | `e3_*.py` (7,7 mil e 8,5 mil linhas), `e3_refatorado/`, `resultados_screening/` | "LIGHT" aprovada com 81,7% de acerto; ao corrigir a execução, o resultado some |
| **E4a** | mar/2026 | Correlações defasadas entre papéis e variáveis macro (dólar, S&P, VIX, petróleo) | `e4_analise_correlacoes*.py`, `resultados_correlacoes/` | Maior correlação defasada: 0,13, dentro do ruído |
| **E4b** | mar – abr/2026 | Valor/dividendos com holding de meses | zip `e4_valor_holding` | Fundamentalista testado com dado do futuro; versão técnica é grade em 5 a 166 trades |

Há uma evolução real de rigor ao longo do caminho (entrada na abertura seguinte e custos no `e3_refatorado`; CAGR com caixa rendendo Selic no E4). O diagnóstico abaixo é duro com os resultados, não com essa trajetória.

## 3. Fase a fase

### E1 — Machine learning por papel

**O que foi feito.** Para cada papel, cerca de 80 features agregadas em janelas de 6 a 60 dias (indicadores técnicos, Ibovespa, Selic, IPCA, dólar, papéis do mesmo setor), alvo binário "fecha acima de +X% em algum dos próximos N dias" (depois com stop, depois em múltiplos de ATR), threshold de probabilidade para emitir COMPRAR, modelos registrados no MLflow e e-mail diário. Mais de 960 combinações papel × parâmetro registradas nos CSVs.

**O que os números dizem.**

| Arquivo | Resultado |
|---|---|
| `experimentos.csv`, `_2`, `_3` | AUC treino 0,81–0,90 contra AUC validação 0,57–0,61: sobreajuste de 0,21 a 0,32 |
| `experimentos_v2.csv` (modelo regularizado) | AUC treino 0,59, validação 0,54; 36% das linhas abaixo de 0,50 |
| `experimentos_wf.csv` | Precisão média em walk-forward 20,6% (máxima 38%) |
| `experimentos_wf_recall_prec_60.csv` | Recall médio com precisão ≥ 60%: **0,5%**. O modelo quase nunca consegue 60% |
| `z_papeis_ITUB4_janelas.ipynb` (versão mais madura) | Hold-out: taxa-base 20%; melhor precisão 39,5%. Com alvo de 2 ATR e stop de 1,3 ATR o ponto de equilíbrio é 39,4%. **Zero antes de custos** |
| `z_papeis_ITUB4_simplificado.ipynb` | Hold-out: precisão máxima 30%, com 10 sinais |

Com 90 dias de validação, o erro-padrão de um AUC é da ordem de 0,06. Um AUC de 0,54 é indistinguível de 0,50.

**Por que falhou.**

- **Amostra minúscula para o problema.** 500 a 1.500 dias por papel, 80 features, alvo de 5 a 10 dias que se sobrepõe: o número de observações independentes é da ordem de 100 a 200. Não há como aprender um sinal fraco com isso.
- **Modelo por papel.** Joga fora a única fonte abundante de dados (a seção transversal: dezenas de papéis por dia).
- **Vazamento na fronteira treino/teste.** O rótulo dos últimos dias de treino olha para preços que já estão dentro do período de teste; não há *purge* nem *embargo* em `dividir_treino_validacao` nem nas janelas rolantes.
- **Threshold escolhido olhando o gabarito.** `calcular_recall_em_precisao_minima` escolhe, dentro de cada fold de teste, o corte que maximiza o resultado naquele fold. Os 26,7% de recall médio com precisão ≥ 50% são um teto otimista, não uma estimativa.
- **Precisão sem taxa-base.** Em `_modelo_inicio_analise.ipynb` o alvo "+2% em 5 dias" para CSNA3 ocorre em 48% dos dias. Uma precisão de 50% ali é não fazer nada.
- **O alvo não é operável.** É medido a partir do fechamento do dia do sinal, mas a compra só é possível na abertura seguinte. O gap fica de fora.
- **Nunca houve simulação de resultado financeiro.** Todo o E1 foi julgado por AUC, precisão e recall. Nenhum notebook calcula quanto o sinal teria rendido.
- **Seleção de features e de `k` usando a própria validação**, repetida papel a papel.

### E2 — Regras técnicas com a melhor estratégia por papel

**O que foi feito.** Cinco regras clássicas, backtest com stop em ATR, trailing stop ou alvo fixo, divisão em in-sample e últimos 300 dias, dimensionamento por risco de 1% e fluxo operacional completo (ordens pendentes, confirmação, monitoramento).

**Desfecho.** No notebook salvo, `ranking_geral` está vazio: nenhuma das 140 combinações (28 papéis × 5 regras) passou no filtro de robustez do próprio screener.

**Problemas.**

- **Mistura de preço ajustado com não ajustado.** `e2_backtest.py` entra em `open` (sem ajuste), sai em `adj_close` (ajustado) e calcula stop com `low` e `atr` (sem ajuste). Em papel que paga provento, o retorno de cada trade sai errado.
- **Saída sempre no preço do stop**, mesmo quando o papel abre em gap abaixo dele.
- **Escolher a melhor regra por papel é otimização in-sample.** O filtro "expectativa fora da amostra ≥ 40% da de dentro" com cerca de 5 trades fora da amostra não tem poder para separar nada.
- `run_backtest` (a primeira versão) entra no fechamento do dia do sinal.

### E3 — Screening por regras e a estratégia "aprovada"

**O que foi feito.** A frente com mais trabalho: momentum, momentum com regime, scoring adaptativo, reversão à média em quatro níveis, stops por ATR, swing em cinco variantes, breakout, grade de alvos, stops e prazos, calibração de "papéis promissores". A documentação conclui que só a Mean Reversion LIGHT passa, com 81,7% de acerto, 48,9% ao ano e Sharpe 1,38.

**Os três números não se sustentam.**

| Número divulgado | De onde vem | Problema |
|---|---|---|
| 81,7% de acerto | Média das taxas anuais de **14 trades** em 4 anos | Com 14 trades, o intervalo de 95% para a taxa real vai de cerca de 50% a 95% |
| 48,9% ao ano | `retorno médio × 252 / dias em posição` | Supõe capital 100% alocado o ano inteiro, emendando um trade no outro. Com 3,5 trades por ano, o ganho real sobre o capital de uma posição seria da ordem de 5% a.a. |
| Sharpe 1,38 | Mesma anualização, com 4 pontos anuais | Sem significado |

Além disso:

- **A entrada era no fechamento do dia do sinal** (`backtest_estrategia` em `e3_screening_regras.py`: `preco_entrada = _val(row, "close")`). O sinal só existe depois do fechamento. Não é executável.
- **Quando a execução foi corrigida, o resultado mudou.** O `e3_refatorado` entra na abertura seguinte e desconta custos. Última execução (`trades_detalhados_2026-03-13.csv`): LIGHT com 16 trades, retorno médio líquido de **+0,85% por trade** (t ≈ 1,1), negativo em 2025 (−0,2%) e em 2026 (−0,9%). SUPER-LIGHT: 58 trades, **−0,34% por trade**. Nada disso foi refletido na documentação, que continua dizendo "VALIDADA".
- **Fragilidade a parâmetros.** Em `teste_alvos_stops_fixos_2026-02-19.csv`, a mesma estratégia rende 48,9% com alvo 3%/stop 4% e **−2,6%** com alvo 3%/stop 3%. Quando meio ponto de stop inverte o sinal do resultado, o que se está medindo é ruído de 14 trades.
- **O "walk-forward" não é walk-forward.** Nenhum parâmetro é estimado no passado e aplicado no futuro. É o detalhamento ano a ano de uma regra cujos limiares, alvo e stop foram escolhidos olhando esses mesmos anos.
- **"Papéis promissores" é seleção in-sample.** A lista filtra por taxa de acerto no próprio período de backtest, com 5 a 12 trades por papel. Os retornos de `papeis_promissores.json` são sempre combinações exatas de +5% e −4%: preenchimento perfeito no alvo e no stop, sem gap e sem custo.
- **Inconsistência não investigada.** `validacao_swing_liquido` mostra variantes de swing com 91% de acerto e Sharpe 6,6 em 26/02 e 38% de acerto e Sharpe negativo em 27/02. Um Sharpe de 6 deveria ter sido tratado como sintoma de erro no momento em que apareceu.
- **Preços sem ajuste de proventos** em todo o E3. Uma data ex-provento vira "queda de 5 dias" e RSI baixo. Em `precos.db`, PETR4 fecha a 36,25 em 11/08/2022 e abre a 29,10 no dia seguinte; é provento, não queda. No recálculo com os 20 papéis disponíveis, um terço dos dias de sinal da LIGHT desaparece quando o preço é ajustado **[controle]**.

**Replicação independente [controle].** LIGHT com preço ajustado e entrada na abertura seguinte, nos 20 papéis de `legado/temp/`: 7 trades em 6 anos, 43% de acerto, −1,0% por trade antes de custos. Entradas aleatórias no mesmo regime têm média de −0,01%; o p-valor do teste de permutação é 0,80. O universo é menor que o original, então a comparação não é exata, mas a direção coincide com a última execução do próprio `e3_refatorado`.

**As rejeições estavam certas.** Momentum, scoring, ATR e swing foram descartados corretamente, embora parte das explicações dadas ("ATR mede volatilidade, não direção") seja racionalização posterior: retornos de −150% ao ano são artefato da fórmula de anualização, não uma propriedade do ATR.

### E4a — Correlações defasadas

Cerca de 950 testes (27 papéis × 7 variáveis × 5 defasagens) em ~500 dias. O erro-padrão de uma correlação nessa amostra é ~0,045, e o máximo esperado entre 950 correlações nulas é ~0,17. O máximo encontrado foi **0,13** (RADL3 × VIX, 3 dias). O mesmo vale para a liderança entre papéis (máximo de 0,15). Resultado correto: não há nada ali. O `e4_exemplo_output_correlacoes.md` mostra uma "hipótese validada" de S&P antecipando o Ibovespa com correlação 0,69; esse texto é um exemplo ilustrativo, não um resultado obtido.

### E4b — Valor e dividendos com holding longo

- **Estratégias fundamentalistas testadas com dado do futuro.** `fundamentalista.db` tem 63 linhas: é uma fotografia atual do Fundamentus. `backtest_papel` replica essa fotografia para todos os trimestres desde 2015. O P/L de 2026 decide a compra de 2016.
- **Estratégias técnicas (queda da máxima de 52 semanas com RSI baixo e dividend yield).** A execução é a mais correta do repositório (entrada e saída na abertura seguinte, slippage, um trade por vez). Mas a avaliação é uma grade de 27 combinações, e os dois arquivos de *stress test* gerados com 9 minutos de diferença têm 5–14 trades num e 22–166 no outro, com Sharpe de −0,2 a 0,7 num e de 0,6 a 2,1 no outro. Algo mudou entre as execuções e não está registrado.
- Preços sem ajuste de proventos (o retorno do investidor de dividendos fica subestimado e as quedas ex-provento geram sinal falso), universo de 24 sobreviventes.
- É a frente menos explorada e a que mais combina com o objetivo declarado. Ver `02_caminhos.md`.

## 4. Os erros que atravessam todas as fases

Em ordem de impacto.

1. **Métrica errada.** Taxa de acerto, precisão e AUC no lugar de retorno esperado líquido contra uma linha de base. Entradas sem sinal nenhum **[controle]**:

   | Alvo | Stop | Acerto | Retorno médio bruto |
   |---|---|---|---|
   | 2% | 6% | 74,2% | +0,09% |
   | 3% | 4% | 58,0% | +0,09% |
   | 3% | 3% | 50,9% | +0,05% |
   | 5% | 4% | 47,0% | +0,13% |
   | 5,5% | 2,8% | 36,9% | +0,10% |

   A taxa de acerto vai de 37% a 74% e o retorno não sai do lugar. Para a configuração 3%/4%, o ponto de equilíbrio é 57% de acerto sem custos e 66% com os custos assumidos no `e3_refatorado`.

2. **Anualização por `252 / dias em posição`.** Infla qualquer estratégia que opera pouco e produz os números absurdos nos dois sentidos (+97% e −457% ao ano). O E4 corrigiu isso com `calcular_cagr_com_selic`; E2 e E3 não.

3. **Amostra pequena e nenhum teste de significância.** Com desvio de ~3,5% por trade, detectar uma vantagem de 0,3% por trade com t = 2 exige cerca de **540 trades**. Uma estratégia com 3,5 trades por ano não pode ser validada nem refutada em tempo útil. Decisões foram tomadas com 5, 14 e 16 trades.

4. **Busca sem contabilidade.** Centenas de variantes foram testadas (o `e3_screening_regras.py` tem mais de 20 funções de teste e validação) e a melhor foi relatada como se fosse a única. Sem registrar quantas tentativas houve, não dá para descontar a sorte da vencedora.

5. **Otimização e avaliação nos mesmos dados.** Limiares, alvo, stop, prazo, lista de papéis e regra por papel foram escolhidos no período em que depois se mede o desempenho. Não existe em nenhuma fase um período reservado que tenha sido tocado uma única vez.

6. **Execução irreal.** Entrada no fechamento do sinal (E1, E2 inicial, E3 original). **[controle]** Na LIGHT, trocar o fechamento pela abertura seguinte tira 1,0 ponto percentual por trade (em apenas 7 trades).

7. **Dados.** Preço sem ajuste de proventos ou mistura dos dois; histórico curto (o cache começa em 2020, então metade da amostra é pandemia e recuperação); universo formado só por papéis líquidos hoje (viés de sobrevivência); checagem de dado que descarta o papel quando há variação diária acima de 30% (remove justamente os eventos extremos).

8. **Documentação à frente da evidência.** `e3_README.md` e `e3_DOCUMENTACAO.md` afirmam resultados que a execução seguinte do próprio código desmentiu.

## 5. O que os testes de controle mostram

Saída completa em `analises/2026-11_revisao/saida_controles.txt`. Dados: 20 papéis, jan/2020 a fev/2026, preços ajustados, entrada na abertura do dia seguinte ao sinal, retorno em excesso à média do universo.

| Família de sinal | 1 dia | 5 dias | 10 dias | Leitura |
|---|---|---|---|---|
| RSI14 < 30 | −0,03% | +0,01% | +0,31% | nada (t < 0,6) |
| IFR2 < 10 acima da MM200 | −0,02% | **−0,35%** | −0,39% | negativo (t = −2,1 em 5 dias), nas duas metades |
| Queda de 5 dias > 5% | −0,09% | −0,10% | +0,03% | nada |
| Abaixo da Bollinger inferior | −0,12% | −0,02% | +0,03% | nada |
| 3 piores do universo em 5 dias | **−0,10%** | **−0,32%** | −0,32% | negativo (t ≈ −3,1), nas duas metades |
| Acima da Bollinger superior | +0,06% | +0,10% | +0,02% | nada |
| Momentum 20 dias + ADX | +0,04% | +0,14% | +0,08% | nada (t ≈ 1) |
| Tendência (MM200, MM20, ADX) | +0,04% | +0,10% | +0,10% | nada; sinal troca entre as metades |
| 3 melhores em ~6 meses | −0,05% | −0,20% | −0,35% | nada (t ≈ −1,5) |

Conclusões:

- **Nenhuma das famílias de sinal de compra usadas no E2 e no E3 mostra efeito positivo.** Não era questão de calibrar.
- **Comprar a queda de curto prazo teve sinal negativo nesta amostra.** As três piores de 5 dias continuaram perdendo para o universo nos dias seguintes. É o oposto da premissa da reversão à média, em que foi gasto mais tempo. São 36 testes; um t de −3,1 sobrevive mal à correção por múltiplas comparações e a amostra é pequena. Fica como hipótese a testar em dados maiores, não como achado.
- **O filtro "Ibov acima da MM200" não melhorou a linha de base.** Entradas aleatórias renderam +0,09% em todos os dias e −0,03% só nos dias de regime favorável. A lição "filtro de regime é essencial (+21 pontos de acerto)" da documentação do E3 não aparece nos dados.

Limitações destes controles: 20 papéis escolhidos em 2026, 6 anos, sem imposto nem aluguel. Servem para descartar, não para aprovar.

## 6. O que se aproveita

- **O padrão de estratégia como arquivo** do `e3_refatorado` e do E4 (`PARAMETROS`, `fn_sinal`, `fn_saida`): é a interface certa para testar ideias rápido.
- **O modelo de execução do `e3_refatorado` e do E4**: sinal no fechamento, entrada na abertura seguinte, stop com gap, custos.
- **`calcular_cagr_com_selic`** do E4: mede o resultado no nível do capital, com o caixa rendendo.
- **A rotina operacional do E2** (ordens pendentes, confirmação, monitoramento, e-mail): utilizável no dia em que houver um sinal que mereça.
- **O cache de download** de `preprocessamento.py`, depois de trocar para preço ajustado.
- **As rejeições**: momentum de curto prazo, scoring de indicadores, stops por ATR, correlações defasadas com macro e ML por papel estão descartados com razão e não precisam ser revisitados nessa forma.

## 7. Higiene do repositório

- **Senha de aplicativo do Gmail em texto puro** em `legado/utils.py`, `legado/e3_refatorado/utils.py`, `legado/_palpiteiro.py` e dentro do zip. **Revogue essa senha na conta Google.** Apagar o arquivo não resolve: ela continua no histórico do Git.
- Um virtualenv inteiro versionado que estava em `scripts/agentes/` (removido do versionamento) (1.487 dos 1.684 arquivos do repositório), além de `__pycache__`.
- `legado/temp/*.csv` salvos sem a coluna de data (o calendário foi reconstruído nesta revisão em `analises/2026-11_revisao/calendario.csv`).
- Arquivos duplicados (`PETR4.ipynb` e `PETR4 copy.ipynb`; `preprocessamento.py` em dois lugares), arquivos vazios (`execucao_diaria.py`, `treino_modelo.py`), caminhos absolutos do Windows no código, bancos SQLite dentro de um zip de 5 MB.
- Scripts de 7 a 8 mil linhas acumulando todas as variantes já testadas. Não é possível saber qual trecho gerou qual resultado.
