"""Figuras del apartado 4.4: Cllr y Cllr_min por condicion, y curvas de Tippett de las cuatro condiciones.
Entradas: resultados/metricas.csv y resultados/llr_<condicion>.npz (generados por comparar_locutores.py).
Salidas: resultados/figura_cllr.png y resultados/figura_tippett.png
Uso:  python codigo/python/figura_resultados.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, CLARO, VERDE, ROJO, GRIS = "#2f5d8a", "#9db7d1", "#2e8b57", "#c0392b", "#555555"
COND = [("limpio_F1234", "1. Limpio\nF1–F4"), ("limpio_F123", "2. Limpio\nF1–F3"),
        ("telefono_F123", "3. Telefónico\nF1–F3"), ("cruzado_F123", "4. Cruzada\nF1–F3")]
met = pd.read_csv("resultados/metricas.csv").set_index("condicion")


def coma(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


# ---- Figura 1: Cllr y Cllr_min por condicion
plt.rcParams.update({"font.size": 9.5})
fig, ax = plt.subplots(figsize=(5.4, 3.5))
x = np.arange(4); w = 0.36
c1 = [met.loc[k, "Cllr"] for k, _ in COND]; c2 = [met.loc[k, "Cllr_min"] for k, _ in COND]
b1 = ax.bar(x - w / 2, c1, w, color=AZUL, label="Cllr (el sistema tal cual)")
b2 = ax.bar(x + w / 2, c2, w, color=CLARO, label="Cllr_min (con calibración ideal)")
for b, v in list(zip(b1, c1)) + list(zip(b2, c2)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, coma(v), ha="center", va="bottom", fontsize=8.5)
ax.axhline(1, color=GRIS, ls="--", lw=1)
ax.text(0.5, 0.97, "1 = no aporta información", ha="center", va="top", fontsize=8.5, color=GRIS, style="italic")
ax.set_xticks(x); ax.set_xticklabels([n for _, n in COND])
ax.set_ylim(0, 1.18); ax.set_ylabel("Cllr (menor es mejor)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
fig.tight_layout(pad=0.6)
fig.savefig("resultados/figura_cllr.png", dpi=220)
plt.close(fig)

# ---- Figura 2: Tippett (proporcion de pares con LR >= valor), 4 condiciones
fig, axs = plt.subplots(2, 2, figsize=(5.4, 4.8), sharex=True, sharey=True)
xs = np.linspace(-4, 2, 400)
for ax, (k, nombre) in zip(axs.ravel(), COND):
    d = np.load(f"resultados/llr_{k}.npz"); l10 = d["llr"] / np.log(10); y = d["y"]
    for datos, col, lab in ((l10[y == 1], VERDE, "mismo locutor"), (l10[y == 0], ROJO, "distinto locutor")):
        ax.plot(xs, [(datos >= v).mean() for v in xs], color=col, lw=1.8, label=lab)
    ax.axvline(0, color=GRIS, lw=0.8, ls="--")
    ax.set_title(nombre.replace("\n", " ") + f"  (Cllr_min {coma(met.loc[k, 'Cllr_min'])})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axs[1]:
    ax.set_xlabel("log10 de la LR")
for ax in axs[:, 0]:
    ax.set_ylabel("Proporción de pares\ncon LR ≥ valor")
axs[1, 1].legend(loc="lower left", fontsize=8, frameon=False)
fig.tight_layout(pad=0.6)
fig.savefig("resultados/figura_tippett.png", dpi=220)
print("Guardado: resultados/figura_cllr.png y resultados/figura_tippett.png")
