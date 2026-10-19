"""Inventario del corpus OpenSLR: hablantes, sexo y nº de grabaciones.
Uso:  python codigo/python/inventario_corpus.py datos/crudos/openslr72 OpenSLR72
"""
import sys
from pathlib import Path
import pandas as pd

carpeta = Path(sys.argv[1])
nombre = sys.argv[2] if len(sys.argv) > 2 else carpeta.name

wavs = sorted(carpeta.rglob("*.wav"))
print(f"{len(wavs)} archivos .wav en {carpeta}")

filas = []
for w in wavs:
    partes = w.stem.split("_")
    if len(partes) != 3:
        continue
    prefijo, id_hab, id_enun = partes
    sexo = {"f": "femenino", "m": "masculino"}.get(prefijo[-1], "desconocido")
    filas.append({"fileid": w.stem, "hablante": f"{prefijo}_{id_hab}",
                  "sexo": sexo, "archivo": str(w)})

df = pd.DataFrame(filas)
print("Prefijos:", sorted({f.split("_")[0] for f in df["fileid"]}))

tsvs = list(carpeta.rglob("line_index*.tsv"))
if tsvs:
    idx = pd.concat([pd.read_csv(t, sep="\t", header=None, names=["fileid", "texto"],
                                 quoting=3, dtype=str) for t in tsvs], ignore_index=True)
    idx["fileid"] = idx["fileid"].str.strip()
    idx = idx.drop_duplicates("fileid")  
    df = df.merge(idx, on="fileid", how="left")
    print(f"Transcripciones emparejadas: {df['texto'].notna().sum()} / {len(df)}")

resumen = (df.groupby(["hablante", "sexo"]).size()
             .reset_index(name="n_grabaciones")
             .sort_values("n_grabaciones", ascending=False))

print(f"\nHablantes: {resumen['hablante'].nunique()} "
      f"(fem: {(resumen.sexo=='femenino').sum()}, masc: {(resumen.sexo=='masculino').sum()})")
print("\nGrabaciones por hablante:")
print(resumen["n_grabaciones"].describe())
print("\n15 hablantes con más grabaciones:")
print(resumen.head(15).to_string(index=False))

df.to_csv(Path("datos") / f"manifiesto_{nombre}.csv", index=False, encoding="utf-8")
resumen.to_csv(Path("datos") / f"hablantes_{nombre}.csv", index=False, encoding="utf-8")
print(f"\nGuardado: datos/manifiesto_{nombre}.csv  y  datos/hablantes_{nombre}.csv")