"""Figura de la Introduccion: la misma frase en condicion limpia y telefonica,
con los formantes superpuestos y la banda telefonica (300-3400 Hz) marcada.
Solo se dibujan formantes en tramos sonoros con energia suficiente; en la
condicion telefonica no se dibuja F4 porque no se mide de forma fiable.
Salida: resultados/figura_intro_canal.png
"""
import numpy as np
import pandas as pd
import parselmouth
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FILEID = "cof_00610_00008989777"
TECHO_LIMPIO = 5500  # hablante femenina
TECHO_TEL = 3900
T_MAX = 2.0
UMBRAL_DB = 25  # se dibuja solo a menos de 25 dB del maximo de intensidad

d = pd.read_csv("datos/seleccion_final.csv")
fila = d[(d.fileid == FILEID) & (d.sesion == "A")].iloc[0]


def espectrograma(snd, f_max):
    sp = snd.to_spectrogram(window_length=0.005, maximum_frequency=f_max)
    return sp.x_grid(), sp.y_grid(), 10 * np.log10(sp.values + 1e-12)


def formantes(snd, techo, maximos):
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                             maximum_formant=techo, window_length=0.025,
                             pre_emphasis_from=50)
    pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=500)
    inten = snd.to_intensity(minimum_pitch=75)
    umbral = inten.values.max() - UMBRAL_DB
    ts = np.arange(0.05, min(snd.duration, T_MAX), 0.01)
    ok = np.array([
        (not np.isnan(pitch.get_value_at_time(t) or np.nan))
        and inten.get_value(t) >= umbral for t in ts
    ])
    ts = ts[ok]
    return ts, {n: np.array([fm.get_value_at_time(n, t) for t in ts]) for n in maximos}


COLORES = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 4: "#1f77b4"}

paneles = [
    ("Grabación limpia (16 kHz)", fila.archivo_procesado, 8000, TECHO_LIMPIO, (1, 2, 3, 4)),
    ("Misma grabación, canal telefónico simulado (300–3400 Hz, 8 kHz)",
     fila.archivo_telefono, 4000, TECHO_TEL, (1, 2, 3)),
]

fig, ejes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for ax, (titulo, ruta, f_max, techo, maximos) in zip(ejes, paneles):
    snd = parselmouth.Sound(ruta)
    X, Y, S = espectrograma(snd, f_max)
    vmax = S.max()
    ax.pcolormesh(X, Y, S, cmap="Greys", vmin=vmax - 70, vmax=vmax, shading="auto")
    ts, fs = formantes(snd, techo, maximos)
    for n, v in fs.items():
        ax.plot(ts, v, "o", ms=3, color=COLORES[n])
    ax.axhspan(3400, 8000, color="#9999aa", alpha=0.25, lw=0)
    ax.axhline(3400, color="k", ls="--", lw=1)
    ax.set_ylim(0, 8000)
    ax.set_xlim(0.7, T_MAX)
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_title(titulo, fontsize=11, loc="left")

ejes[1].text((0.7 + T_MAX) / 2, 5900,
             "sin información por encima de 3400 Hz\nF4 no medible de forma fiable",
             ha="center", va="center", fontsize=10.5, style="italic", color="#333355")
ejes[1].set_xlabel("Tiempo (s)")
h = [Line2D([0], [0], marker="o", ls="", color=COLORES[n], label=f"F{n}") for n in (1, 2, 3, 4)]
h.append(Line2D([0], [0], color="k", ls="--", lw=1, label="Límite del canal telefónico (3400 Hz)"))
ejes[0].legend(handles=h, loc="upper left", ncol=3, fontsize=8.5, framealpha=0.95)

fig.tight_layout()
fig.savefig("resultados/figura_intro_canal.png", dpi=200)
print("Guardado: resultados/figura_intro_canal.png")
