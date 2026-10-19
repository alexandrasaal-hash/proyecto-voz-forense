"""Figura del apartado 3.4: distribucion de F1-F4 en la condicion limpia frente al
limite superior de la banda telefonica (3400 Hz), y comparacion del F4 devuelto
por el algoritmo en condicion limpia y telefonica.
Entrada: rasgos/formantes.csv. Salida: resultados/figura_formantes_banda.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIMITE = 3400
COLORES = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 4: "#1f77b4"}


def pct(x):
    return f"{100 * x:.1f}".replace(".", ",") + " %"


d = pd.read_csv("rasgos/formantes.csv")
bins = np.arange(0, 6001, 50)

plt.rcParams.update({"font.size": 10.5})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.4), gridspec_kw={"height_ratios": [1.15, 1]})

# Panel A: F1-F4 en limpio
for n in (1, 2, 3, 4):
    x = d[f"F{n}_limpio"].dropna()
    fuera = (x > LIMITE).mean()
    ax1.hist(x, bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=COLORES[n],
             label=f"F{n}:  {pct(fuera)}")
ax1.axvspan(LIMITE, 6000, color="#9999aa", alpha=0.25, lw=0)
ax1.axvline(LIMITE, color="k", ls="--", lw=1)
ax1.text(LIMITE + 120, ax1.get_ylim()[1] * 0.40, "fuera de la banda\ntelefónica", fontsize=9.5, va="top", color="#333355", style="italic")
ax1.set_xlim(0, 6000)
ax1.set_xlabel("Frecuencia (Hz)")
ax1.set_ylabel("Densidad")
ax1.set_yticks([])
ax1.set_title("A. Formantes en condición limpia", loc="left", fontsize=11)
ax1.legend(loc="upper right", fontsize=10, framealpha=0.95, title=f"% por encima de {LIMITE} Hz", title_fontsize=10)

# Panel B: F4 limpio frente a F4 devuelto en telefonico
x4l = d["F4_limpio"].dropna()
x4t = d["F4_tel"].dropna()
ax2.hist(x4l, bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=COLORES[4],
         label=f"F4 limpio (mediana {x4l.median():.0f} Hz)")
ax2.hist(x4t, bins=bins, density=True, histtype="stepfilled", alpha=0.55, color="#7f7f7f",
         label=f"F4 devuelto en telefónico\n(mediana {x4t.median():.0f} Hz)")
ax2.axvspan(LIMITE, 6000, color="#9999aa", alpha=0.25, lw=0)
ax2.axvline(LIMITE, color="k", ls="--", lw=1)
ax2.set_xlim(1000, 6000)
ax2.set_xlabel("Frecuencia (Hz)")
ax2.set_yticks([])
ax2.set_title("B. F4: limpio frente a telefónico", loc="left", fontsize=11)
ax2.legend(loc="upper right", fontsize=10, framealpha=0.95)

fig.tight_layout()
fig.savefig("resultados/figura_formantes_banda.png", dpi=200)
print("Guardado: resultados/figura_formantes_banda.png")
