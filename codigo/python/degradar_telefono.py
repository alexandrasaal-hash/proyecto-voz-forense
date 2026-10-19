"""Genera la version telefonica de cada audio procesado:
filtro paso banda 300-3400 Hz + remuestreo a 8 kHz. Guarda en datos/telefono/
y anade la columna archivo_telefono al CSV.
Uso:  python codigo/python/degradar_telefono.py
"""
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Decisiones (memoria, cap. 4.2):
# - Simulacion de canal telefonico de banda estrecha: paso banda 300-3400 Hz + 8 kHz.
# - Efecto buscado: perdida del F4 (queda sobre Nyquist, 4 kHz -> no medible) y
#   degradacion del F3. Es exactamente lo que el estudio quiere cuantificar.

ENTRADA = Path("datos/seleccion_procesada.csv")
DEST = Path("datos/telefono")
DEST.mkdir(parents=True, exist_ok=True)

AF = "highpass=f=300,lowpass=f=3400"

df = pd.read_csv(ENTRADA, dtype=str).dropna(subset=["archivo_procesado"])
print(f"{len(df)} audios a degradar")

rutas, errores = [], []
for row in tqdm(df.itertuples(index=False), total=len(df)):
    origen = Path(row.archivo_procesado)
    salida = DEST / origen.name
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(origen),
           "-ac", "1", "-ar", "8000", "-af", AF, str(salida)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and salida.exists():
        rutas.append(str(salida))
    else:
        rutas.append("")
        errores.append((origen.name, r.stderr.strip()[:200]))

df["archivo_telefono"] = rutas
df.to_csv("datos/seleccion_final.csv", index=False, encoding="utf-8")

ok = sum(1 for x in rutas if x)
print(f"\nDegradados OK: {ok} / {len(df)}")
if errores:
    print(f"Errores: {len(errores)}")
    for n, e in errores[:5]:
        print(" ", n, "->", e)
print("Guardado: datos/seleccion_final.csv")