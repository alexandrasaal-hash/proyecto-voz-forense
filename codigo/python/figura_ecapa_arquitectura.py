"""Figura del apartado 3.3.3: diagrama simplificado de la arquitectura ECAPA-TDNN (Desplanques et al., 2020),
con la configuracion del modelo preentrenado spkrec-ecapa-voxceleb de SpeechBrain que se usa en este trabajo
(80 filtros Mel; canales 1024/1024/1024/1024/3072; nucleos 5/3/3/3/1; dilataciones 1/2/3/4/1; 128 canales de
atencion; embedding de 192 dimensiones). Es un esquema: no ejecuta el modelo ni lee datos.
Salida: resultados/figura_ecapa_arquitectura.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "resultados" / "figura_ecapa_arquitectura.png"

C_E, C_B, C_A, C_S = "#fde9d9", "#dbe9f6", "#e3f1e0", "#eeeeee"   # entrada, bloques, agregacion/pooling, entrada/salida
EDGE = "#555555"
filas = [  # (texto principal, detalle, color)
    ("Grabación de voz", "16 kHz, un canal", C_S),
    ("Características de entrada", "80 filtros Mel", C_E),
    ("Capa TDNN inicial", "convolución 1D, núcleo 5, 1024 canales", C_B),
    ("3 bloques residuales con atención de canal", "núcleo 3, dilataciones 2, 3 y 4, 1024 canales", C_B),
    ("Agregación de capas", "se unen las salidas de los bloques; núcleo 1, 3072 canales", C_A),
    ("Agrupación estadística con atención", "128 canales de atención; media y desviación ponderadas", C_A),
    ("Embedding", "vector de 192 dimensiones", C_S),
]

fig, ax = plt.subplots(figsize=(5.0, 5.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")
h, gap = 1.07, 0.42
y = 10.4 - h
cx, w = 5, 8.6
for k, (t, d, c) in enumerate(filas):
    ax.add_patch(FancyBboxPatch((cx - w / 2, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=c, ec=EDGE, lw=0.9))
    ax.text(cx, y + h * 0.64, t, ha="center", va="center", fontsize=8.6, fontweight="bold", color="#222222")
    ax.text(cx, y + h * 0.27, d, ha="center", va="center", fontsize=7.4, color="#333333")
    if k < len(filas) - 1:
        ax.add_patch(FancyArrowPatch((cx, y), (cx, y - gap), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=EDGE))
    y -= h + gap
fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=300)
print("figura guardada:", OUT)
