"""Procesa los audios seleccionados a formato uniforme (mono, 16 kHz, 16 bits)
y los guarda en datos/procesados/. Actualiza el CSV con la ruta procesada.
Uso:  python codigo/python/procesar_audios.py
"""
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Decisiones (memoria, cap. 4.2):
# - Condicion "limpia": mono, 16 kHz, PCM 16 bits.
# - Sin normalizacion de sonoridad (los formantes son invariantes al nivel).
# - Nombre de salida: <sesion>__<fileid>.wav

ENTRADA = Path("datos/seleccion.csv")
DEST = Path("datos/procesados")
DEST.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ENTRADA, dtype=str)
print(f"{len(df)} audios a procesar")

rutas, errores = [], []
for row in tqdm(df.itertuples(index=False), total=len(df)):
    origen = Path(row.archivo_origen)
    salida = DEST / f"{row.sesion}__{row.fileid}.wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(origen),
           "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(salida)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and salida.exists():
        rutas.append(str(salida))
    else:
        rutas.append("")
        errores.append((row.fileid, r.stderr.strip()[:200]))

df["archivo_procesado"] = rutas
df.to_csv("datos/seleccion_procesada.csv", index=False, encoding="utf-8")

ok = sum(1 for x in rutas if x)
print(f"\nProcesados OK: {ok} / {len(df)}")
if errores:
    print(f"Errores: {len(errores)} (primeros 5)")
    for fid, err in errores[:5]:
        print(" ", fid, "->", err)
print("Guardado: datos/seleccion_procesada.csv")