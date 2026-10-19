"""Figura del apartado 4.1: cronograma del trabajo por fases (abril-noviembre de 2026) con las fechas
oficiales de entrega del Periodo 3: borrador 19/10, deposito 09/11 y defensa 16-27/11.
Fechas documentadas: solicitud de tema y tutorizacion (documentos de 19-24/04), inicio del trabajo tecnico
(29-30/08). Junio-julio: busqueda de informacion y lectura, segun la autora.
Salida: resultados/figura_cronograma.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MESES = ["Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov"]   # indices 0..7
COL = {"hecho": "#2f5d8a", "curso": "#8fb4d9", "previsto": "#ffffff", "intento": "#d5dde6"}
BORDE = "#2f5d8a"
ROJO = "#c0392b"


def x_dia(mes, dia, dias_mes):
    return mes + (dia - 1) / dias_mes


INI_AGO = x_dia(4, 29, 31)           # 29/08
BORRADOR = x_dia(6, 19, 31)          # 19/10
DEPOSITO = x_dia(7, 9, 30)           # 09/11
DEF_INI, DEF_FIN = x_dia(7, 16, 30), x_dia(7, 27, 30)   # 16-27/11

FASES = [
    ("Solicitud formal del TFM\n(Anexos I y II)", []),   # hitos: anexos fechados 24/04; solicitud entregada 05/06
    ("Recopilación y lectura de\nartículos; exploración de\ndatos; planteamiento", [(2, 4, "hecho")]),
    ("Definición del alcance y\nrevisión bibliográfica", [(x_dia(3, 15, 31), 4, "intento"), (INI_AGO, 5, "hecho")]),
    ("Entorno de trabajo\n(herramientas)", [(x_dia(3, 13, 31), 4, "intento"), (INI_AGO, x_dia(5, 2, 30), "hecho")]),   # Git instalado 13/07; venv 30/08
    ("Selección y preparación\nde los datos", [(x_dia(3, 15, 31), 4, "intento"), (5, 6, "hecho")]),
    ("Extracción de formantes y\nsistema de comparación", [(5, 6, "hecho")]),
    ("Evaluación y análisis\nde los resultados", [(5, 6, "hecho")]),
    ("Redacción del borrador\nde la memoria", [(5, BORRADOR, "curso")]),
    ("Revisión y reescritura\nde la memoria", [(6, DEPOSITO, "curso")]),
    ("Ampliaciones previstas\n(grabaciones propias, MVKD,\nECAPA-TDNN, F0 y sexo)", [(6, DEPOSITO, "previsto")]),
]

plt.rcParams.update({"font.size": 9.5})
fig, ax = plt.subplots(figsize=(7.4, 7.2))
n = len(FASES)
for i, (nombre, tramos) in enumerate(FASES):
    y = n - i
    for (a, b, est) in tramos:
        ax.barh(y, b - a, left=a, height=0.62, color=COL[est], edgecolor=BORDE, linewidth=1.3,
                hatch="//" if est == "previsto" else (".." if est == "intento" else None))

# hitos de tutorizacion y solicitud (fila superior)
for x, txt in [(x_dia(0, 24, 30), "Anexos\n24/04"), (x_dia(2, 5, 30), "Entrega\n05/06")]:
    ax.scatter([x], [n], marker="D", s=62, color=COL["hecho"], zorder=5)
    ax.text(x + 0.14, n, txt, ha="left", va="center", fontsize=8, color=BORDE)

# fila de entregas
ax.scatter([BORRADOR, DEPOSITO], [0, 0], marker="D", s=70, color=ROJO, zorder=5)
ax.barh(0, DEF_FIN - DEF_INI, left=DEF_INI, height=0.5, color=ROJO, edgecolor=ROJO)
for x, txt, ylab in [(BORRADOR, "Borrador\n19/10", -1.1), (DEPOSITO, "Depósito\n09/11", -2.1),
                     ((DEF_INI + DEF_FIN) / 2, "Defensa\n16–27/11", -1.1)]:
    ax.plot([x, x], [-0.3, ylab + 0.05], color=ROJO, lw=0.9, ls=":", zorder=1)
    ax.text(x, ylab, txt, ha="center", va="top", fontsize=8, color=ROJO, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1.5))

ax.set_yticks(range(n + 1))
ax.set_yticklabels(["Entregas y defensa\n(Periodo 3)"] + [f[0] for f in reversed(FASES)])
ax.set_xticks([k + 0.5 for k in range(len(MESES))])
ax.set_xticklabels(MESES)
ax.set_xlim(0, len(MESES))
ax.set_ylim(-3.1, n + 0.6)
ax.xaxis.tick_top()
ax.set_xlabel("2026", labelpad=4)
ax.xaxis.set_label_position("top")
for k in range(1, len(MESES)):
    ax.axvline(k, color="#cccccc", lw=0.8, zorder=0)
ax.tick_params(length=0)
for s in ("right", "bottom", "left"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(fc=COL["hecho"], ec=BORDE, label="Realizado"),
                   Patch(fc=COL["intento"], ec=BORDE, hatch="..", label="Primeros intentos"),
                   Patch(fc=COL["curso"], ec=BORDE, label="En curso"),
                   Patch(fc="white", ec=BORDE, hatch="//", label="Previsto")],
          loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False)
fig.tight_layout()
fig.savefig("resultados/figura_cronograma.png", dpi=200)
print("Guardado: resultados/figura_cronograma.png")

