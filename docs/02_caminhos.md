# Caminhos a seguir

Continuação de `01_diagnostico.md`. Este documento propõe (1) uma correção no objetivo, (2) o protocolo que toda ideia nova precisa seguir, (3) a estrutura no GitHub para validar ideias conversando por aqui, (4) a fila de hipóteses em ordem de prioridade e (5) as decisões que dependem de você.

## 1. Antes de qualquer ideia nova: três correções de rumo

### 1.1 O cenário mais provável é não encontrar nada

Vantagem em sinais técnicos diários sobre as ações mais líquidas da B3, com dados gratuitos e execução na abertura, é o que todo mundo procura. A expectativa razoável é que a maioria das hipóteses abaixo falhe. O projeto precisa de um **critério de parada** definido agora, e não depois do décimo resultado ruim. Sugestão: 10 hipóteses pré-registradas; se nenhuma passar no período reservado, a conclusão é que a rotina diária não compensa, e o dinheiro vai para CDI e índice.

Isso também é um resultado útil. O que não é útil é o que aconteceu até aqui: um ano de trabalho sem conseguir dizer com segurança nem que funciona nem que não funciona.

### 1.2 "Um palpite por dia" é uma restrição que atrapalha

Exigir uma ação todo pregão empurra para sinais fracos e frequentes. As duas formas que têm alguma base são outras:

- **Carteira com rebalanceamento periódico** (mensal ou semanal): a rotina diária pós-fechamento existe, mas na maior parte dos dias a resposta é "manter".
- **Sinal raro com amostra larga**: o sinal é raro por papel, mas é testado em 100+ papéis e 15+ anos, o que dá centenas de eventos.

A LIGHT era um sinal raro em amostra pequena, a pior combinação: 3,5 trades por ano não se validam em tempo útil.

### 1.3 "Modesto, porém provável" precisa de definição numérica

Com a Selic em dois dígitos, "modesto e provável" já existe e se chama CDI. A meta precisa ser escrita contra ele. Proposta de definição operacional:

- **Retorno**: CDI + X% ao ano, líquido de custos e imposto (X a definir; 3 a 5 pontos já é ambicioso).
- **"Provável"**: probabilidade de perder para o CDI em janelas de 12 meses abaixo de um limite (por exemplo, 30%), estimada por bootstrap, e queda máxima tolerada definida de antemão.
- **Taxa de acerto por trade sai do vocabulário.** Ela não mede probabilidade de lucro da estratégia.

## 2. Protocolo de validação

Regras fixas, aplicadas pelo código e não pela disciplina de quem roda.

| # | Regra | Erro passado que evita |
|---|---|---|
| 1 | **Hipótese escrita antes de rodar**: sinal, universo, execução, métrica e critério de aprovação num arquivo versionado | Escolher a métrica depois de ver o resultado |
| 2 | **Uma métrica principal**: retorno em excesso à linha de base, líquido, com estatística t por data | Taxa de acerto, AUC, precisão |
| 3 | **Linha de base obrigatória** no mesmo relatório: CDI, Ibovespa, universo em pesos iguais e entradas aleatórias com a mesma saída | Comparar com zero |
| 4 | **Preço ajustado por proventos**; sinal em t, execução na abertura de t+1; gap respeitado no stop | Entrada no fechamento, quedas ex-provento |
| 5 | **Custos e imposto no resultado**: emolumentos e spread, IR de 15% (swing) ou 20% (day trade) sobre o ganho líquido, aluguel em posições vendidas. Valores a confirmar com a sua corretora | Lucro bruto |
| 6 | **Resultado no nível do capital**: curva de patrimônio com caixa rendendo CDI, limite de posições simultâneas | Anualização por `252 / dias` |
| 7 | **Três períodos**: desenvolvimento, validação e **reserva**. A reserva (os últimos 2 a 3 anos) é lida uma única vez por hipótese, e o código registra essa leitura | Otimizar e avaliar nos mesmos dados |
| 8 | **Amostra mínima**: 300 eventos ou 60 rebalanceamentos; abaixo disso o relatório diz "inconclusivo", não "aprovado" | Decidir com 14 trades |
| 9 | **Registro de tentativas**: toda execução entra em `registro.csv`, inclusive as que falharam. O limiar de aprovação sobe com o número de tentativas | Relatar só a vencedora |
| 10 | **Sensibilidade**: o resultado precisa sobreviver a ±20% em cada parâmetro e manter o sinal nas duas metades do período | Alvo 3%/stop 4% funciona, 3%/3% não |
| 11 | **Juiz final é o papel**: toda hipótese aprovada passa a gerar sinais diários registrados no repositório antes de receber dinheiro | Confiar no backtest |

**Critério de aprovação proposto** (ajustável antes de começar, nunca depois): t ≥ 3 no desenvolvimento e validação, mesmo sinal e pelo menos metade da magnitude na reserva, excesso líquido sobre a linha de base maior que o custo assumido, e 6 meses de papel coerentes com o backtest.

## 3. Estrutura no GitHub

### 3.1 Fluxo

```
conversa aqui ──► eu escrevo hipoteses/NNN_nome.yaml + estrategias/nome.py ──► push
                                                                               │
          você lê o relatório ◄── relatorios/NNN_nome.md + linha em registro.csv ◄── GitHub Actions roda a validação
```

A validação roda no GitHub Actions, não nesta sessão. Motivo prático: o ambiente em que eu rodo não alcança Yahoo, B3 nem Banco Central (as conexões são recusadas pela política de rede), então eu não consigo baixar cotações daqui. O Actions consegue. A alternativa é liberar esses domínios na configuração de rede do ambiente; as duas coisas podem coexistir.

### 3.2 Árvore proposta

```
palpiteiro/
├── docs/                      # diagnóstico, caminhos, protocolo
├── pesquisa/
│   ├── dados.py               # download, ajuste por proventos, cache, checagens
│   ├── universo.py            # composição do universo por data (sem viés de sobrevivência)
│   ├── motor.py               # execução, custos, imposto, curva de patrimônio
│   ├── avaliacao.py           # linhas de base, estatística t por data, bootstrap, sensibilidade
│   ├── validar.py             # CLI: python -m pesquisa.validar hipoteses/003_xxx.yaml
│   └── testes/                # testes do próprio motor (anti-lookahead, custos, gap)
├── estrategias/               # um arquivo por ideia (padrão PARAMETROS / fn_sinal / fn_saida)
├── hipoteses/                 # um YAML por hipótese pré-registrada
├── relatorios/                # um MD por execução, gerado pelo Actions
├── registro.csv               # todas as tentativas, com hash do commit
├── papel/                     # sinais diários e carteira simulada das hipóteses aprovadas
├── legado/                    # o conteúdo atual de scripts/, congelado
└── .github/workflows/
    ├── dados.yml              # diário, após o fechamento: atualiza o cache de preços
    ├── validar.yml            # dispara quando muda algo em hipoteses/
    └── diario.yml             # diário: gera sinais das aprovadas e atualiza papel/
```

### 3.3 Decisões de projeto

- **Fonte de dados.** A fonte oficial é o arquivo de séries históricas da B3 (COTAHIST), que traz todos os papéis de cada ano, inclusive os que deixaram de existir, e resolve o viés de sobrevivência. Não vem ajustado: é preciso aplicar proventos e desdobramentos. O yfinance vem ajustado e é simples, mas só cobre quem está listado hoje e tem falhas. Recomendação: começar com yfinance para andar rápido, com o universo definido por liquidez em cada data, e migrar para COTAHIST se alguma hipótese chegar perto de passar. Declarar a limitação em cada relatório enquanto isso.
- **Histórico.** Desde 2010 no mínimo. O cache atual começa em 2020, e metade dele é pandemia.
- **Universo.** Os 100 papéis mais líquidos em cada data, não uma lista fixa de 27.
- **Motor testado.** O motor ganha testes automáticos antes de qualquer estratégia: uma estratégia que usa o preço de amanhã precisa ser barrada, e uma estratégia aleatória precisa dar zero menos custos. Os erros do passado foram quase todos no motor, não nas ideias.
- **Reserva protegida por código.** `validar.py` só lê o período de reserva com uma opção explícita e grava o uso em `registro.csv`.
- **Segredos.** E-mail e senha passam a ser *secrets* do GitHub. A senha atual precisa ser revogada (ver diagnóstico, seção 7).
- **Legado.** `legado/` vai para `legado/` sem o virtualenv, os `__pycache__` e os CSVs de `temp/`. Nada é apagado do histórico.

### 3.4 Formato de uma hipótese

```yaml
id: 003
nome: perdedoras_5d_continuam
ideia: "As 10% piores do universo em 5 dias seguem abaixo do universo nos 5 dias seguintes"
origem: "controle E do diagnóstico (t = -3,1 em 20 papéis, 2020-2026)"
universo: top100_liquidez
sinal: estrategias/perdedoras_5d.py
execucao: {entrada: abertura_t1, saida: fechamento_t5}
metrica: excesso_vs_universo_liquido
criterio: {t_min: 3.0, eventos_min: 300, mesma_direcao_na_reserva: true}
parametros_fixos: {janela: 5, fracao: 0.10}
sensibilidade: {janela: [3, 5, 10], fracao: [0.05, 0.10, 0.20]}
```

## 4. Fila de hipóteses

A probabilidade indicada é a minha estimativa de que a hipótese sobreviva ao protocolo, não uma medição. Nenhuma passa de "talvez".

### Prioridade alta: baratas de testar e alinhadas com "modesto e provável"

**H1. Alocação por tendência no índice.** Ficar em BOVA11 quando o Ibovespa está acima de uma média longa (ou com retorno de 10 a 12 meses positivo) e em CDI caso contrário. Uma decisão por mês. A evidência internacional é de redução de quedas mais do que de aumento de retorno; com CDI alto, a perna de caixa pesa a favor. É a hipótese mais próxima do objetivo declarado e a mais simples. O filtro "Ibov acima da MM200" não ajudou no controle com trades de 15 dias, então isso precisa ser medido, não presumido. Probabilidade: média para reduzir risco, baixa para superar o CDI.

**H2. Carteira de fatores, só comprada, rebalanceada mensalmente.** Ordenar o universo por uma característica e manter os 10 a 20 melhores. Candidatas, uma hipótese cada: baixa volatilidade; dividend yield com filtro de qualidade; momentum de 6 a 12 meses. A evidência para o Brasil é mista e depende do período; no controle com 20 papéis o momentum de 6 meses teve sinal negativo, sem significância. Usa só preço e proventos, então não tem o problema de dado fundamentalista histórico do E4. Probabilidade: média-baixa.

**H3. Perdedoras de curto prazo continuam fracas.** O único resultado do controle com t perto de 3 e sinal igual nas duas metades. Dois usos possíveis: como filtro de exclusão em H2, ou como venda (exige aluguel e margem). O efeito medido é pequeno (−0,3% em 5 dias) e pode não pagar os custos. Vale testar primeiro porque, se for confirmado em amostra grande, encerra de vez a linha de "comprar a queda". Probabilidade: média de confirmar o sinal, baixa de ser explorável.

### Prioridade média: exigem dado ou execução a mais

**H4. Deriva pós-resultado.** Papéis que reagem forte no dia da divulgação do resultado tendem a continuar na mesma direção por semanas. É uma das anomalias mais documentadas fora do Brasil. Exige o calendário de divulgações (CVM, gratuito). O sinal aparece poucas vezes por papel, mas em 100 papéis e 15 anos são milhares de eventos. Encaixa bem em "decidir após o fechamento, agir na abertura". Probabilidade: média-baixa.

**H5. Pares estruturais.** ON contra PN da mesma empresa (PETR3/PETR4, BBDC3/BBDC4, ITUB3/ITUB4) e holding contra controlada (ITSA4/ITUB4, BRAP4/VALE3). A relação tem fundamento econômico, diferente das correlações do E4a. A versão completa exige venda; a versão comprada troca para a classe mais barata dentro de uma posição que já se quer ter. Muito conhecido e arbitrado. Probabilidade: baixa.

**H6. Efeito intradiário e noturno.** Decompor o retorno em fechamento→abertura e abertura→fechamento e ver se há assimetria persistente por papel ou por dia. Com o motor pronto, custa uma tarde. Operar isso seria day trade (IR de 20%) ou compra no leilão de fechamento. Probabilidade: baixa.

### Prioridade baixa: só depois que algo simples mostrar sinal

**H7. Machine learning, refeito do zero.** Um único modelo para todos os papéis, prevendo a posição relativa do retorno futuro entre os papéis do dia, com validação cruzada com *purge* e *embargo*, alimentado apenas por características que tenham mostrado sinal isoladamente em H2 a H4. O E1 falhou em parte por tentar extrair de um modelo o que as features não tinham. Se as características simples não mostrarem nada, o modelo também não vai. Probabilidade: baixa.

### Não recomendo retomar

- Combinações de RSI, MACD, Bollinger e ADX com alvo e stop em papéis individuais: o controle E cobre as famílias usadas e não há nada.
- Otimização de alvo, stop e prazo: não cria retorno, só muda a taxa de acerto.
- Modelo por papel e listas de "papéis promissores".
- Correlações defasadas com variáveis macro.
- Estratégias fundamentalistas sem base histórica *point-in-time*. Dá para construir essa base a partir dos demonstrativos da CVM, mas é um projeto à parte.

## 5. Sequência sugerida

| Etapa | Conteúdo | Critério de saída |
|---|---|---|
| **0. Higiene** | Revogar a senha do Gmail; mover `legado/` para `legado/`; remover virtualenv e caches do versionamento | Repositório limpo, sem segredo ativo |
| **1. Motor** | `dados`, `universo`, `motor`, `avaliacao`, testes do motor, workflow de validação | Estratégia aleatória dá zero menos custos; estratégia com lookahead é barrada; os controles A–E são reproduzidos em 100 papéis desde 2010 |
| **2. Linhas de base** | Relatório único: CDI, Ibovespa, universo em pesos iguais, por ano | Régua publicada em `relatorios/000_base.md` |
| **3. Hipóteses** | H1, H3, H2 (três variantes), nessa ordem; depois H4 | Cada uma com relatório e linha no registro |
| **4. Papel** | Workflow diário para o que passar | 6 meses de registro |
| **5. Decisão** | Dinheiro real em tamanho pequeno, ou encerramento pelo critério de parada | — |

As etapas 0 a 2 não dependem de nenhuma ideia dar certo e são onde está o maior ganho: hoje o gargalo é não conseguir confiar em nenhum resultado, bom ou ruim.

## 6. O que preciso que você decida

1. **Meta numérica**: quanto acima do CDI, e qual queda máxima você tolera.
2. **Critério de parada**: aceita 10 hipóteses pré-registradas como limite?
3. **Restrições operacionais**: pode operar vendido (aluguel)? Day trade está fora? Quantas posições simultâneas você administra? Qual ordem de grandeza de capital (muda imposto, lote e custo relativo)?
4. **Frequência**: aceita uma rotina em que na maioria dos dias a instrução é "manter"?
5. **Fonte de dados**: começar com yfinance e migrar depois, ou já investir no COTAHIST?
6. **Rede do ambiente**: quer liberar Yahoo, B3 e Banco Central para esta sessão, ou fica tudo no Actions?
7. **Legado**: posso reorganizar `legado/` em `legado/` e tirar o virtualenv do versionamento?
