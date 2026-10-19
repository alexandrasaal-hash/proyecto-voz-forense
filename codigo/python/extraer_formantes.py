"""Extraccion de formantes por vocal, en local (parselmouth). Sustituye a WebMAUS.
Version reanudable: escribe resultados fila a fila; si se corta, al volver a
ejecutar continua donde se quedo (no repite grabaciones ya procesadas).
Salida: rasgos/formantes.csv
Uso:  python codigo/python/extraer_formantes.py
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import parselmouth
from tqdm import tqdm

MANIF = "datos/seleccion_final.csv"
OUT = Path("rasgos/formantes.csv"); OUT.parent.mkdir(exist_ok=True)
CAMPOS = ["fileid","hablante","sexo","sesion","corpus","t","vocal",
          "F1_limpio","F2_limpio","F3_limpio","F4_limpio","F0_limpio",
          "F1_tel","F2_tel","F3_tel","F4_tel","F0_tel"]

REGIONES = {
    "i": {"F1": (200, 500), "F2": (1700, 3200)},
    "e": {"F1": (300, 700), "F2": (1500, 2600)},
    "a": {"F1": (550, 1200), "F2": (900, 1900)},
    "o": {"F1": (451, 700), "F2": (500, 1250)},
    "u": {"F1": (200, 450), "F2": (500, 1100)},   # F1 450 Hz separa /u/ (cerrada) de /o/
}

def clasifica(f1, f2):
    c = [v for v, r in REGIONES.items()
         if r["F1"][0] <= f1 <= r["F1"][1] and r["F2"][0] <= f2 <= r["F2"][1]]
    return c[0] if len(c) == 1 else None

def nucleos(snd):
    it = snd.to_intensity(minimum_pitch=75)
    t, v = it.xs(), it.values[0]
    if len(v) < 9:
        return []
    thr = np.nanmedian(v) + 3.0
    w, picos = 4, []
    for i in range(w, len(v) - w):
        if v[i] >= thr and v[i] == np.nanmax(v[i - w:i + w + 1]):
            if not picos or (t[i] - picos[-1]) > 0.08:
                picos.append(t[i])
    return picos

def formantes(snd, ceiling, n_formantes=5):
    # Limpio: 5 formantes (techo 5000/5500 Hz). Telefonico: 4 formantes (techo 3900 Hz, Nyquist a 8 kHz);
    # con 5, el algoritmo reparte formantes de mas en 0-3900 Hz y desplaza F2 y F3 (~-260 y ~-830 Hz).
    return snd.to_formant_burg(time_step=0.01, max_number_of_formants=n_formantes,
                               maximum_formant=ceiling, window_length=0.025,
                               pre_emphasis_from=50)

def procesa_par(ruta_l, ruta_t, ceil_l):
    snd_l = parselmouth.Sound(str(ruta_l))
    f0_l = snd_l.to_pitch(pitch_floor=75, pitch_ceiling=500)
    fmt_l = formantes(snd_l, ceil_l)
    snd_t = parselmouth.Sound(str(ruta_t))
    f0_t = snd_t.to_pitch(pitch_floor=75, pitch_ceiling=500)
    fmt_t = formantes(snd_t, 3900, 4)
    filas = []
    for tt in nucleos(snd_l):
        v0 = f0_l.get_value_at_time(tt)
        if v0 is None or np.isnan(v0):
            continue
        f1l = fmt_l.get_value_at_time(1, tt)
        f2l = fmt_l.get_value_at_time(2, tt)
        if f1l is None or f2l is None or np.isnan(f1l) or np.isnan(f2l):
            continue
        vocal = clasifica(f1l, f2l)
        if vocal is None:
            continue
        fila = {"t": round(tt, 3), "vocal": vocal}
        for pref, fmt, f0 in [("limpio", fmt_l, f0_l), ("tel", fmt_t, f0_t)]:
            for n in (1, 2, 3, 4):
                fila[f"F{n}_{pref}"] = fmt.get_value_at_time(n, tt)
            fila[f"F0_{pref}"] = f0.get_value_at_time(tt)
        filas.append(fila)
    return filas

df = pd.read_csv(MANIF, dtype=str).dropna(subset=["archivo_procesado", "archivo_telefono"])
print(f"{len(df)} grabaciones")

ya_hechos = set()
if OUT.exists():
    ya_hechos = set(pd.read_csv(OUT, usecols=["fileid"])["fileid"].unique())
    print(f"Reanudando: {len(ya_hechos)} grabaciones ya procesadas")

nuevo = not OUT.exists()
with open(OUT, "a", newline="", encoding="utf-8") as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=CAMPOS)
    if nuevo:
        writer.writeheader()
    errores = 0
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        if row.fileid in ya_hechos:
            continue
        meta = {"fileid": row.fileid, "hablante": row.hablante, "sexo": row.sexo,
                "sesion": row.sesion, "corpus": row.corpus}
        ceil_l = 5500 if row.sexo == "femenino" else 5000
        try:
            for f in procesa_par(row.archivo_procesado, row.archivo_telefono, ceil_l):
                writer.writerow({**meta, **f})
        except Exception as e:
            errores += 1
            if errores <= 3:
                tqdm.write(f"  error en {row.fileid}: {e}")
        fcsv.flush()

print(f"\nTerminado. errores: {errores}")
res = pd.read_csv(OUT)
print(f"Total tokens: {len(res)}")
print("\nTokens por vocal:")
print(res["vocal"].value_counts().to_string())
print("\nMedia de tokens por hablante y sesion:")
print(res.groupby(["hablante", "sesion"]).size().mean().round(1))