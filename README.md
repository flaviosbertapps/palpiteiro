# palpiteiro

Tentativas de construir uma rotina diária, pós-fechamento, de sugestão de operações na B3.

## Por onde começar

- [`docs/01_diagnostico.md`](docs/01_diagnostico.md): o que foi feito, o que falhou e por quê (revisão de nov/2026).
- [`docs/02_caminhos.md`](docs/02_caminhos.md): protocolo de validação, estrutura proposta e fila de hipóteses.
- [`analises/2026-11_revisao/`](analises/2026-11_revisao/): testes de controle que sustentam o diagnóstico. Para reproduzir: `python controles.py` (precisa só de pandas e numpy; usa os CSVs de `legado/temp/`).

## Conteúdo legado

`legado/` guarda o material das fases E1 a E4, como foi produzido. Nenhuma estratégia ali está validada, incluindo a marcada como aprovada em `legado/e3_README.md`.
