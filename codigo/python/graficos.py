"""Graficos adicionales para la memoria: espacio vocalico, ejemplo visual de
comparacion mismo/distinto locutor, y resumen de metricas por condicion.
Salida: resultados/espacio_vocalico.png, resultados/ejemplo_comparacion.png,
        resultados/resumen_metricas.png
Uso:  python codigo/python/graficos.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("resultados")
df = pd.read_csv("rasgos/formantes.csv")

COLORES = {"a": "#B96C2A", "e": "#3A6FB0", "i": "#2E7D5B", "o": "#8A3B8A", "u": "#C0392B"}

# 1. Espacio vocalico (F1-F2), condicion limpia, muestra de puntos
muestra = df.sample(n=min(3000, len(df)), random_state=42)
plt.figure(figsize=(6, 6))
for v, sub in muestra.groupby("vocal"):
    plt.scatter(sub["F2_limpio"], sub["F1_limpio"], s=8, alpha=0.35,
                color=COLORES.get(v, "gray"), label=f"/{v}/")
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel("F2 (Hz)")
plt.ylabel("F1 (Hz)")
plt.title("Espacio vocalico (condicion limpia)")
plt.legend(title="Vocal")
plt.tight_layout()
plt.savefig(OUT / "espacio_vocalico.png", dpi=150)
plt.close()

# 2. Ejemplo visual: mismo locutor vs distinto locutor
agg = (df.groupby(["hablante", "sesion", "vocal"])
         .agg(F1=("F1_limpio", "mean"), F2=("F2_limpio", "mean"))
         .reset_index())

conteo = agg.groupby("hablante")["vocal"].nunique()
candidatos = sorted(conteo[conteo == 5].index)  # hablantes con las 5 vocales
h1, h2 = candidatos[0], candidatos[1]

def puntos(hablante, sesion):
    sub = agg[(agg.hablante == hablante) & (agg.sesion == sesion)]
    return sub.set_index("vocal")[["F1", "F2"]]

def dibuja(ax, pa, pb, titulo):
    for v in pa.index:
        if v in pb.index:
            f2a, f1a = pa.loc[v, "F2"], pa.loc[v, "F1"]
            f2b, f1b = pb.loc[v, "F2"], pb.loc[v, "F1"]
            c = COLORES.get(v, "gray")
            ax.plot([f2a, f2b], [f1a, f1b], "-", color=c, alpha=0.6)
            ax.scatter(f2a, f1a, color=c, marker="o", s=80, edgecolor="k", linewidth=0.4)
            ax.scatter(f2b, f1b, color=c, marker="^", s=80, edgecolor="k", linewidth=0.4)
            ax.annotate(f"/{v}/", (f2a, f1a), fontsize=9, xytext=(4, 4),
                       textcoords="offset points")
    ax.set_title(titulo)
    ax.set_xlabel("F2 (Hz)")
    ax.invert_xaxis()
    ax.invert_yaxis()

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
dibuja(axes[0], puntos(h1, "A"), puntos(h1, "B"),
       f"Mismo locutor\n({h1}: sesion A vs sesion B)")
axes[0].set_ylabel("F1 (Hz)")
dibuja(axes[1], puntos(h1, "A"), puntos(h2, "B"),
       f"Distinto locutor\n({h1} ses. A  vs  {h2} ses. B)")
fig.suptitle("Circulo = sesion A · Triangulo = sesion B · lineas cortas = locuciones parecidas")
plt.tight_layout()
plt.savefig(OUT / "ejemplo_comparacion.png", dpi=150)
plt.close()

# 3. Resumen de metricas por condicion
met = pd.read_csv(OUT / "metricas.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(met))
w = 0.35
ax.bar(x - w / 2, met["Cllr"], w, label="Cllr (obtenido)", color="#B96C2A")
ax.bar(x + w / 2, met["Cllr_min"], w, label="Cllr_min (potencial)", color="#3A6FB0")
ax.set_xticks(x)
ax.set_xticklabels(met["condicion"], rotation=20, ha="right")
ax.set_ylabel("Cllr (bits, menor es mejor)")
ax.set_title("Coste del log-LR por condicion")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "resumen_metricas.png", dpi=150)
plt.close()

print("Guardado: resultados/espacio_vocalico.png, ejemplo_comparacion.png, resumen_metricas.png")
print(f"Ejemplo usado: mismo={h1} · distinto={h1} vs {h2}")
