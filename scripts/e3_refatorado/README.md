# e3_screening — Estrutura Refatorada

## Arquivos

```
e3_refatorado/
│
├── e3_screening.py          # Motor principal (backtest + relatório)
├── e3_estrategias.json      # Lista de estratégias a testar
├── e3_alvos_stops.json      # Configuração de alvos, stops e modo
│
└── estrategias/             # Um arquivo por estratégia
    ├── mean_reversion_light.py          ✅ VALIDADA
    ├── mean_reversion_super_light.py    ⚠️ em avaliação
    └── swing_balanceado_d.py            ⚠️ em avaliação
```

---

## Uso

### Backtest com estratégias ativas (listadas em e3_estrategias.json)
```bash
python e3_screening.py
```

### Backtest de estratégias específicas
```bash
python e3_screening.py --estrategias mean_reversion_light swing_balanceado_d
```

### Modo grade (testa combinações de alvo/stop definidas em e3_alvos_stops.json)
```bash
python e3_screening.py --modo grade
```

### Alterar número de anos
```bash
python e3_screening.py --anos 3
```

### Debug em papel único
```bash
python e3_screening.py --ticker PETR4.SA
```

---

## Como adicionar uma nova estratégia

1. Crie `estrategias/minha_estrategia.py` seguindo o padrão:

```python
import numpy as np
import pandas as pd

PARAMETROS = {
    "rsi_max":   40,
    "alvo_pct":  0.04,
    "stop_pct":  0.03,
    "max_dias":  15,
    # ...outros parâmetros
}

def _val(row, col, default=np.nan):
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)

def fn_sinal(row: pd.Series, p: dict) -> bool:
    return (
        _val(row, "rsi") <= p["rsi_max"]
        # ...outras condições
    )

ESTRATEGIA = {
    "nome":      "minha_estrategia",
    "descricao": "Descrição curta",
    "parametros": PARAMETROS,
    "fn_sinal":   fn_sinal,
}
```

2. Adicione o nome em `e3_estrategias.json`:
```json
{
  "estrategias_ativas": ["mean_reversion_light", "minha_estrategia"],
  "estrategias_disponiveis": ["...", "minha_estrategia"]
}
```

3. Rode:
```bash
python e3_screening.py
```

---

## Configuração de alvos/stops (e3_alvos_stops.json)

| Modo   | Comportamento |
|--------|---------------|
| `fixo` | Usa alvo_pct e stop_pct definidos em cada estratégia |
| `grade`| Testa todas as combinações de alvos × stops × max_dias |

---

## Saídas geradas (pasta resultados_screening/)

| Arquivo | Conteúdo |
|---------|----------|
| `trades_detalhados_YYYY-MM-DD.csv` | Todos os trades com ticker, data, retorno, resultado |
| `resumo_anual_YYYY-MM-DD.csv`      | Métricas por estratégia × ano (trades, taxa, ret/trade, ret anual, sharpe, drawdown) |
