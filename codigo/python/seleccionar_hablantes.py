"""Selecciona hablantes equilibrados por sexo y divide cada uno en dos
pseudo-sesiones (A/B). Salida: datos/seleccion.csv
Uso:  python codigo/python/seleccionar_hablantes.py
"""

# Decisiones metodologicas (para la memoria, cap. 4.2):
# - 24 hablantes (12 F + 12 M): tamano comparable al de Lopez-Escobedo et al. (2025).
# - Pseudo-sesiones A/B dividiendo las frases: OpenSLR es de sesion unica.
#   Es una limitacion (infla el rendimiento); ver Enzinger & Morrison (2012).
# - SEED = 42: la seleccion es reproducible.


from pathlib import Path
import pandas as pd

# ---- parametros (ajusta si la segmentacion va lenta) ----
MIN_GRAB   = 80
N_POR_SEXO = 12          # total = 2 * N_POR_SEXO
N_FRASES   = 80          # por hablante; mitad por sesion
SEED       = 42

MANIFIESTOS = {"SLR72": Path("datos/manifiesto_OpenSLR72.csv"),
               "SLR73": Path("datos/manifiesto_OpenSLR73.csv")}

partes = []
for corpus, ruta in MANIFIESTOS.items():
    d = pd.read_csv(ruta, dtype=str)
    d["corpus"] = corpus
    partes.append(d)
df = pd.concat(partes, ignore_index=True).drop_duplicates("fileid")
print(f"Total grabaciones: {len(df)}  |  hablantes: {df['hablante'].nunique()}")

conteo = df.groupby(["hablante", "sexo", "corpus"]).size().reset_index(name="n")
elegibles = conteo[conteo["n"] >= MIN_GRAB]
print(f"\nElegibles (>= {MIN_GRAB} grabaciones):")
print(elegibles.groupby("sexo").size().to_string())

sel = []
for sexo in ["femenino", "masculino"]:
    cand = elegibles[elegibles["sexo"] == sexo].sort_values("hablante")
    if len(cand) < N_POR_SEXO:
        print(f"AVISO: solo {len(cand)} hablantes {sexo} elegibles (pedidos {N_POR_SEXO})")
    sel.append(cand.sample(n=min(N_POR_SEXO, len(cand)), random_state=SEED))
sel_hab = pd.concat(sel)
print("\nHablantes seleccionados:")
print(sel_hab.sort_values(["sexo", "hablante"]).to_string(index=False))

filas = []
for _, fila in sel_hab.iterrows():
    g = (df[df["hablante"] == fila["hablante"]]
         .sample(n=min(N_FRASES, fila["n"]), random_state=SEED)
         .sort_values("fileid").reset_index(drop=True))
    mitad = len(g) // 2
    g["sesion"] = ["A"] * mitad + ["B"] * (len(g) - mitad)
    filas.append(g)

seleccion = (pd.concat(filas, ignore_index=True)
             [["corpus", "hablante", "sexo", "sesion", "fileid", "archivo", "texto"]]
             .rename(columns={"archivo": "archivo_origen"}))

seleccion.to_csv("datos/seleccion.csv", index=False, encoding="utf-8")
print(f"\nGuardado: datos/seleccion.csv  ({len(seleccion)} archivos)")
print("\nGrabaciones por hablante y sesion:")
print(seleccion.groupby(["hablante", "sesion"]).size().unstack(fill_value=0).to_string())