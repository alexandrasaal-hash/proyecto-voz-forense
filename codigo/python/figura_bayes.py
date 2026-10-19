"""Figura del apartado 3.2: esquema de la relacion entre las probabilidades a priori, la razon de
verosimilitud (aportada por el perito) y las probabilidades a posteriori (tribunal).
Salida: resultados/figura_bayes.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COL_TRIB = ("#e8eef5", "#5a7da8")   # tribunal: relleno, borde
COL_PERI = ("#fdeee0", "#d9822b")   # perito

fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 41.5); ax.axis("off")


def caja(x, w, titulo, sub, dueno, col):
    ax.add_patch(FancyBboxPatch((x, 14), w, 24, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=col[0], ec=col[1], lw=1.8))
    ax.text(x + w / 2, 32.5, titulo, ha="center", va="center", fontsize=9.6, fontweight="bold", color="#222")
    ax.text(x + w / 2, 22.5, sub, ha="center", va="center", fontsize=8.2, color="#333")
    ax.text(x + w / 2, 9.2, dueno, ha="center", va="center", fontsize=10.5, fontweight="bold", color=col[1])


caja(1, 27, "Razón de\nprobabilidades\na priori", "según el resto de\nlas pruebas del caso", "Tribunal", COL_TRIB)
caja(37, 27, "Razón de\nverosimilitud\n(LR)", "p(evidencia de voz | H1)\n———————————\np(evidencia de voz | H2)", "Perito", COL_PERI)
caja(72, 27, "Razón de\nprobabilidades\na posteriori", "valoración final\nde H1 frente a H2", "Tribunal", COL_TRIB)

ax.text(32.5, 26, "×", ha="center", va="center", fontsize=26, color="#444")
ax.text(68, 26, "=", ha="center", va="center", fontsize=26, color="#444")

ax.text(50, 3.0, "H1: las dos grabaciones proceden del mismo locutor      H2: proceden de locutores distintos",
        ha="center", va="center", fontsize=8.2, color="#444", style="italic")
fig.tight_layout()
fig.savefig("resultados/figura_bayes.png", dpi=200)
print("Guardado: resultados/figura_bayes.png")
