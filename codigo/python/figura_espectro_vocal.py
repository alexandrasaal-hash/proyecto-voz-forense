"""Figura del apartado 3.1: espectro de una vocal /a/ real del corpus, con su
envolvente (LPC) y los cuatro primeros formantes marcados.
Se elige el nucleo vocalico de /a/ (hablante masculino, condicion limpia) cuyos F1-F4
estan mas cerca de las medianas de /a/ masculina, para que sea un ejemplo representativo.
Entradas: rasgos/formantes.csv y datos/procesados/*.wav. Salida: resultados/figura_espectro_vocal.png
"""
import numpy as np
import pandas as pd
import parselmouth
from scipy.linalg import solve_toeplitz
from scipy.signal import freqz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORES = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 4: "#1f77b4"}
d = pd.read_csv("rasgos/formantes.csv")
g = d[(d.vocal == "a") & (d.sexo == "masculino")].dropna(subset=[f"F{n}_limpio" for n in (1, 2, 3, 4)])
med = {n: g[f"F{n}_limpio"].median() for n in (1, 2, 3, 4)}
dist = sum(((g[f"F{n}_limpio"] - med[n]) / med[n]) ** 2 for n in (1, 2, 3, 4))
fila = g.loc[dist.idxmin()]
print("Token elegido:", fila.fileid, "sesion", fila.sesion, "t =", fila.t, "| F1-F4:",
      [round(fila[f"F{n}_limpio"]) for n in (1, 2, 3, 4)], "| medianas:", {k: round(v) for k, v in med.items()})

snd = parselmouth.Sound(f"datos/procesados/{fila.sesion}__{fila.fileid}.wav")
snd = snd.resample(10000)  # techo de formantes de 5000 Hz (hablante masculino), como en extraer_formantes.py
fs = snd.sampling_frequency
x = snd.extract_part(from_time=fila.t - 0.0200, to_time=fila.t + 0.0200, preserve_times=False).values[0]
x = np.append(x[0], x[1:] - 0.97 * x[:-1])          # preenfasis
xw = x * np.hamming(len(x))

N = 8192
esp = 20 * np.log10(np.abs(np.fft.rfft(xw, N)) + 1e-12)
f = np.fft.rfftfreq(N, 1 / fs)

orden = 10  # 5 formantes = 10 polos, igual que en la extraccion
r = np.correlate(xw, xw, "full")[len(xw) - 1:len(xw) + orden]
a = solve_toeplitz(r[:orden], r[1:orden + 1])
w, h = freqz([1.0], np.concatenate(([1.0], -a)), worN=N // 2 + 1, fs=fs)
env = 20 * np.log10(np.abs(h) + 1e-12)
env += np.max(esp[(f > 200) & (f < 1200)]) - np.max(env[(w > 200) & (w < 1200)])

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(f, esp, color="#b0b0b0", lw=0.9, label="Espectro de la vocal")
ax.plot(w, env, color="black", lw=2, label="Envolvente espectral")
techo = max(esp.max(), env.max())
for n in (1, 2, 3, 4):
    fr = fila[f"F{n}_limpio"]
    ax.axvline(fr, color=COLORES[n], ls="--", lw=1.4)
    ax.text(fr + 40, techo + 4 - (n % 2) * 7, f"F{n}", color=COLORES[n], fontsize=11, fontweight="bold", va="bottom")
ax.set_xlim(0, 5000)
ax.set_ylim(esp[(f < 5000)].min() - 2, techo + 18)
ax.set_xlabel("Frecuencia (Hz)")
ax.set_ylabel("Nivel (dB, escala relativa)")
ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
fig.tight_layout()
fig.savefig("resultados/figura_espectro_vocal.png", dpi=200)
print("Guardado: resultados/figura_espectro_vocal.png")
