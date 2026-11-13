"""
Script de diagnóstico: Verificar se correlações papel×papel com lag existem
Ou se realmente são todas < 0.40
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Ler arquivo completo
arquivo = "resultados_correlacoes/lideranca_papeis/correlacoes_com_lag_completo.csv"

if not Path(arquivo).exists():
    print(f"❌ Arquivo não encontrado: {arquivo}")
    print("Execute primeiro: python e4_analise_correlacoes_cli.py --lideranca-papeis")
    exit()

df = pd.read_csv(arquivo)

print(f"\n{'='*70}")
print("🔍 DIAGNÓSTICO: Correlações Papel × Papel com LAG")
print(f"{'='*70}\n")

print(f"📊 Dataset:")
print(f"   Total de correlações: {len(df)}")
print(f"   Papéis únicos (líderes): {df['lider'].nunique()}")
print(f"   Papéis únicos (seguidores): {df['seguidor'].nunique()}")

print(f"\n{'='*70}")
print("📈 DISTRIBUIÇÃO DAS CORRELAÇÕES")
print(f"{'='*70}\n")

# Estatísticas
print(f"Correlação absoluta:")
print(f"   Média:   {df['abs_corr'].mean():.3f}")
print(f"   Mediana: {df['abs_corr'].median():.3f}")
print(f"   Máxima:  {df['abs_corr'].max():.3f}")
print(f"   Mínima:  {df['abs_corr'].min():.3f}")

print(f"\nDistribuição por faixa:")
ranges = [
    (0.70, 1.00, "Muito forte"),
    (0.60, 0.70, "Forte"),
    (0.50, 0.60, "Moderado-alto"),
    (0.40, 0.50, "Moderado"),
    (0.30, 0.40, "Fraco"),
    (0.20, 0.30, "Muito fraco"),
    (0.00, 0.20, "Irrelevante"),
]

for min_val, max_val, label in ranges:
    count = len(df[(df['abs_corr'] >= min_val) & (df['abs_corr'] < max_val)])
    pct = 100 * count / len(df)
    print(f"   {label:20s} ({min_val:.2f}-{max_val:.2f}): {count:4d} ({pct:5.1f}%)")

print(f"\n{'='*70}")
print("🏆 TOP 50 CORRELAÇÕES (mesmo que < 0.40)")
print(f"{'='*70}\n")

top50 = df.nlargest(50, 'abs_corr')
print(top50[['lider', 'seguidor', 'correlacao', 'lag', 'abs_corr']].to_string(index=False))

# Salvar top 50
top50.to_csv("resultados_correlacoes/lideranca_papeis/top50_todas_correlacoes.csv", index=False)

print(f"\n{'='*70}")
print("💡 ANÁLISE")
print(f"{'='*70}\n")

max_corr = df['abs_corr'].max()

if max_corr < 0.20:
    print("❌ PROBLEMA GRAVE: Correlações máximas < 0.20")
    print("   Isso NÃO é normal. Possíveis causas:")
    print("   1. Bug no cálculo de retornos")
    print("   2. Dados com problemas (datas desalinhadas)")
    print("   3. Período muito volátil/atípico")
    
elif max_corr < 0.30:
    print("⚠️  CORRELAÇÕES MUITO FRACAS: Máxima < 0.30")
    print("   Papéis BR podem realmente ter pouca liderança entre si")
    print("   Período testado pode ser atípico")
    
elif max_corr < 0.40:
    print("⚠️  CORRELAÇÕES FRACAS: Máxima < 0.40")
    print("   Liderança papel×papel no Brasil é fraca")
    print("   Melhor focar em correlações macro→papel")
    
elif max_corr < 0.50:
    print("✅ CORRELAÇÕES MODERADAS: Máxima < 0.50")
    print("   Existem sinais fracos, mas usáveis com cautela")
    print(f"   Top correlação: {max_corr:.3f}")
    
else:
    print("✅ CORRELAÇÕES BOAS: Máxima >= 0.50")
    print(f"   Top correlação: {max_corr:.3f}")
    print("   Existem sinais usáveis!")

print(f"\n📁 Arquivo salvo: top50_todas_correlacoes.csv")
print(f"{'='*70}\n")