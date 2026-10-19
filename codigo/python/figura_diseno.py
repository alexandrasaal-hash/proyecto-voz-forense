"""Figura del apartado 4.2: esquema ilustrado del diseno experimental, de los datos a las metricas.
Disenada a su tamano final (5,2 in = ~375 pt de ancho): se inserta sin reducirla, letra de 8-10 pt.
Salida: resultados/figura_diseno.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

AZUL, NARANJA, VERDE, GRIS, ROJO = "#2f5d8a", "#d9822b", "#2e8b57", "#8a8a8a", "#c0392b"
MUJER, HOMBRE, LIMPIO, TEL = "#7b4fa0", "#1b8a8f", "#5b6770", "#8c564b"   # sexo (morado/turquesa); canal (gris/marron)
FONDO = "#f3f6fa"

fig, ax = plt.subplots(figsize=(5.2, 4.95))
ax.set_xlim(0, 100); ax.set_ylim(0, 80.5); ax.axis("off")

W, H = 30, 36
COLS = [0.5, 35.0, 69.5]
FILA_Y = [43.8, 1.0]
T_TIT, T_CUE, T_PEQ = 9.4, 9, 8.5


def caja(x, y, num, titulo, tam=None):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.3,rounding_size=1.6", fc=FONDO, ec=AZUL, lw=1.4))
    ax.add_patch(Circle((x + 3.6, y + H - 3.8), 2.5, fc=AZUL, ec="none"))
    ax.text(x + 3.6, y + H - 3.8, str(num), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(x + 7.2, y + H - 3.8, titulo, ha="left", va="center", fontsize=tam or T_TIT, fontweight="bold", color="#222")


def persona(cx, cy, col, s=1.0):
    ax.add_patch(Circle((cx, cy + 1.9 * s), 1.1 * s, fc=col, ec="none"))
    ax.add_patch(FancyBboxPatch((cx - 1.5 * s, cy - 1.6 * s), 3.0 * s, 3.0 * s, boxstyle="round,pad=0.05,rounding_size=1.2", fc=col, ec="none"))


def flecha(p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.6, color=AZUL))


# ---------- 1. Datos ----------
x, y = COLS[0], FILA_Y[0]
caja(x, y, 1, "Datos")
for (corpus, yy) in [("SLR72 · Colombia", y + 27.6), ("SLR73 · Perú", y + 15.0)]:
    ax.text(x + 2.4, yy, corpus, ha="left", va="center", fontsize=T_CUE - 0.3, color="#222", fontweight="bold")
    persona(x + 5.0, yy - 4.4, MUJER, s=0.68)
    ax.text(x + 8.6, yy - 4.0, "6 mujeres", ha="left", va="center", fontsize=T_PEQ + 0.2, color=MUJER, fontweight="bold")
    persona(x + 5.0, yy - 8.4, HOMBRE, s=0.68)
    ax.text(x + 8.6, yy - 8.0, "6 hombres", ha="left", va="center", fontsize=T_PEQ + 0.2, color=HOMBRE, fontweight="bold")
ax.text(x + W / 2, y + 1.9, "Total: 24 hablantes", ha="center", va="center", fontsize=T_CUE - 0.3, color="#222", fontweight="bold")
# ---------- 2. Pseudosesiones ----------
x, y = COLS[1], FILA_Y[0]
caja(x, y, 2, "Pseudosesiones", tam=8.2)
ax.add_patch(Rectangle((x + 3, y + 22.0), 24, 7.6, fc="white", ec=GRIS, lw=1.2))
ax.text(x + 15, y + 25.8, "80 grabaciones\nde cada hablante", ha="center", va="center", fontsize=T_CUE, linespacing=1.1)
for (bx, lab, col) in [(x + 1.0, "A", AZUL), (x + 16.0, "B", NARANJA)]:
    ax.add_patch(Rectangle((bx, y + 10.4), 13.0, 8.4, fc="white", ec=col, lw=1.8))
    ax.text(bx + 6.5, y + 15.6, f"Sesión {lab}", ha="center", va="center", fontsize=T_PEQ - 0.4, color=col, fontweight="bold")
    ax.text(bx + 6.5, y + 12.2, "40 grab.", ha="center", va="center", fontsize=T_PEQ, color="#222")
ax.add_patch(FancyArrowPatch((x + 10.5, y + 21.8), (x + 8, y + 19.2), arrowstyle="-|>", mutation_scale=9, lw=1.3, color=GRIS))
ax.add_patch(FancyArrowPatch((x + 19.5, y + 21.8), (x + 22, y + 19.2), arrowstyle="-|>", mutation_scale=9, lw=1.3, color=GRIS))
ax.text(x + W / 2, y + 5.0, "dos conjuntos\ndisjuntos (simulan\ndos sesiones)", ha="center", va="center", fontsize=T_PEQ, color="#444", style="italic", linespacing=1.1)

# ---------- 3. Canal ----------
x, y = COLS[2], FILA_Y[0]
caja(x, y, 3, "Canal")
X0, ANCHO = x + 2.2, 25.6          # eje comun: 0 - 8 kHz
kx = lambda khz: X0 + ANCHO * khz / 8.0
ax.text(x + 15, y + 27.2, "Cada grabación (A y B)\nen dos versiones:", ha="center", va="center", fontsize=T_PEQ, color="#222", linespacing=1.1)
# limpio: toda la banda
ax.text(X0, y + 23.4, "Limpio", ha="left", va="center", fontsize=T_CUE, color=LIMPIO, fontweight="bold")
ax.add_patch(Rectangle((X0, y + 19.2), ANCHO, 3.0, fc=LIMPIO, ec="none"))
ax.text(kx(4), y + 20.7, "0 – 8 kHz", ha="center", va="center", fontsize=T_PEQ, color="white", fontweight="bold")
# telefonico: solo 0,3 - 3,4 kHz; el resto, sin senal
ax.text(X0, y + 16.3, "Telefónico", ha="left", va="center", fontsize=T_CUE, color=TEL, fontweight="bold")
ax.add_patch(Rectangle((kx(3.4), y + 12.0), kx(8) - kx(3.4), 3.0, fc="#e3e3e3", ec=GRIS, lw=0.6, hatch="////"))
ax.add_patch(Rectangle((kx(0.3), y + 12.0), kx(3.4) - kx(0.3), 3.0, fc=TEL, ec="none"))
ax.text(kx(5.7), y + 13.5, "sin señal", ha="center", va="center", fontsize=T_PEQ - 0.8, color="#333", fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.8))
ax.text(kx(1.85), y + 13.5, "0,3–3,4", ha="center", va="center", fontsize=T_PEQ - 1.0, color="white", fontweight="bold")
# eje de frecuencia
ax.plot([X0, X0 + ANCHO], [y + 10.0, y + 10.0], color="#555", lw=1)
for k in (0, 2, 4, 6, 8):
    ax.plot([kx(k), kx(k)], [y + 10.0, y + 9.1], color="#555", lw=1)
    ax.text(kx(k), y + 7.7, str(k), ha="center", va="center", fontsize=T_PEQ - 0.7, color="#333")
ax.text(x + 15, y + 5.0, "frecuencia (kHz)", ha="center", va="center", fontsize=T_PEQ - 0.9, color="#333")
ax.text(x + 15, y + 1.7, "F4 ≈ 3,6 kHz: fuera", ha="center", va="center", fontsize=T_PEQ - 0.5, color="#444", style="italic")
# ---------- 4. Pares ----------
x, y = COLS[0], FILA_Y[1]
caja(x, y, 4, "Comparaciones", tam=8.6)
XA, XB = x + 17.6, x + 26.0
filas = [("Hablante 1", y + 25.2), ("Hablante 2", y + 22.0), ("Hablante 3", y + 18.8), ("Hablante 4", y + 15.6), ("Hablante 24", y + 9.4)]
for (etq, yy) in filas:
    ax.text(x + 1.6, yy, etq, ha="left", va="center", fontsize=T_PEQ - 1.0, color="#222")
    ax.add_patch(Circle((XA, yy), 1.1, fc=AZUL, ec="none"))
    ax.add_patch(Circle((XB, yy), 1.1, fc=NARANJA, ec="none"))
    ax.plot([XA, XB], [yy, yy], color=VERDE, lw=1.8)
# distinto locutor: algunos ejemplos (A de un hablante con B de otro)
ax.plot([XA, XB], [filas[0][1], filas[2][1]], color=ROJO, lw=1.2, ls="--")
ax.plot([XA, XB], [filas[1][1], filas[3][1]], color=ROJO, lw=1.2, ls="--")
ax.plot([XA, XB], [filas[2][1], filas[4][1]], color=ROJO, lw=1.2, ls="--")
# puntos suspensivos (hablantes 5 a 23)
for xx in (x + 8.0, XA, XB):
    for k in range(3):
        ax.add_patch(Circle((xx, y + 13.3 - k * 1.2), 0.30, fc="#555", ec="none"))
ax.text(XA, y + 28.4, "A", ha="center", fontsize=T_CUE, color=AZUL, fontweight="bold")
ax.text(XB, y + 28.4, "B", ha="center", fontsize=T_CUE, color=NARANJA, fontweight="bold")
ax.text(x + 15, y + 5.4, "verde: mismo locutor", ha="center", va="center", fontsize=T_PEQ - 0.3, color=VERDE, fontweight="bold")
ax.text(x + 15, y + 2.2, "rojo: distinto locutor", ha="center", va="center", fontsize=T_PEQ - 0.3, color=ROJO, fontweight="bold")
# ---------- 5. Sistema ----------
x, y = COLS[1], FILA_Y[1]
caja(x, y, 5, "Sistema")
ax.add_patch(FancyBboxPatch((x + 2.0, y + 21.2), 26, 7.2, boxstyle="round,pad=0.2,rounding_size=1.2", fc="white", ec=AZUL, lw=1.4))
ax.text(x + 15, y + 24.8, "Diferencia de\nformantes entre A y B\n(F1–F3 o F1–F4)", ha="center", va="center", fontsize=T_PEQ - 0.7, linespacing=1.15)
ax.add_patch(FancyArrowPatch((x + 15, y + 20.8), (x + 15, y + 19.3), arrowstyle="-|>", mutation_scale=10, lw=1.4, color=AZUL))
ax.add_patch(FancyBboxPatch((x + 2.0, y + 13.8), 26, 5.0, boxstyle="round,pad=0.2,rounding_size=1.2", fc=AZUL, ec=AZUL))
ax.text(x + 15, y + 16.4, "Regresión logística\n→ LR", ha="center", va="center", fontsize=T_CUE - 0.4, color="white", fontweight="bold", linespacing=1.1)
ax.text(x + 15, y + 11.0, "LR > 1: misma persona", ha="center", va="center", fontsize=T_PEQ - 0.5, color=VERDE, fontweight="bold")
ax.text(x + 15, y + 8.8, "LR < 1: distinta", ha="center", va="center", fontsize=T_PEQ - 0.5, color=ROJO, fontweight="bold")
ax.text(x + 15, y + 4.0, "leave-one-speaker-out:\n23 entrenan, 1 prueba\n(uno distinto cada vez)", ha="center", va="center", fontsize=T_PEQ - 1.0, color="#444", style="italic", linespacing=1.15)

# ---------- 6. Metricas ----------
x, y = COLS[2], FILA_Y[1]
caja(x, y, 6, "Métricas")
t = np.linspace(-3.5, 3.5, 100)
base = y + 19.0
ax.plot(x + 5.0 + (t + 3.5) / 7 * 21, base + 1.0 + 8.4 / (1 + np.exp(-1.6 * (t - 0.8))), color=VERDE, lw=2)
ax.plot(x + 5.0 + (t + 3.5) / 7 * 21, base + 1.0 + 8.4 / (1 + np.exp(-1.6 * (t + 0.8))), color=ROJO, lw=2)
ax.plot([x + 5.0, x + 5.0, x + 26.5], [base + 10.4, base + 1.0, base + 1.0], color="#555", lw=1)
ax.text(x + 15, y + 14.6, "Cllr, Cllr_min\ny EER", ha="center", va="center", fontsize=T_CUE, fontweight="bold", color="#222", linespacing=1.35)
ax.text(x + 15, y + 9.0, "Cllr: 0 = perfecto,\n1 = no informa", ha="center", va="center", fontsize=T_PEQ - 0.8, color="#444", style="italic", linespacing=1.15)
ax.text(x + 15, y + 3.6, "gráficos de Tippett", ha="center", va="center", fontsize=T_PEQ - 0.8, color="#444", style="italic")

# ---------- flechas ----------
flecha((COLS[0] + W + 0.5, FILA_Y[0] + H / 2), (COLS[1] - 0.5, FILA_Y[0] + H / 2))
flecha((COLS[1] + W + 0.5, FILA_Y[0] + H / 2), (COLS[2] - 0.5, FILA_Y[0] + H / 2))
y_mid = (FILA_Y[0] + FILA_Y[1] + H) / 2
ax.plot([COLS[2] + W / 2, COLS[2] + W / 2, COLS[0] + W / 2], [FILA_Y[0] - 0.4, y_mid, y_mid], color=AZUL, lw=1.6, solid_capstyle="round")
flecha((COLS[0] + W / 2, y_mid), (COLS[0] + W / 2, FILA_Y[1] + H + 0.7))
flecha((COLS[0] + W + 0.5, FILA_Y[1] + H / 2), (COLS[1] - 0.5, FILA_Y[1] + H / 2))
flecha((COLS[1] + W + 0.5, FILA_Y[1] + H / 2), (COLS[2] - 0.5, FILA_Y[1] + H / 2))

fig.tight_layout(pad=0.3)
fig.savefig("resultados/figura_diseno.png", dpi=240)
print("Guardado: resultados/figura_diseno.png")









