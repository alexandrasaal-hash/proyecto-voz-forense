"""Diagnostico del efecto del canal sobre los formantes (apartado 4.4).
1) Diferencia media absoluta entre dos sesiones (A frente a B), para parejas del mismo hablante y de
   hablantes distintos, en limpio-limpio, telefonico-telefonico y limpio-telefonico.
2) Desplazamiento de F2 y F3 (telefonico menos limpio) por vocal.
Salidas: resultados/desglose_canal.csv y resultados/desglose_canal_vocal.csv
Uso:  python codigo/python/desglose_canal.py
"""
import numpy as np
import pandas as pd

d = pd.read_csv("rasgos/formantes.csv")
cols = {**{f"F{n}_limpio": "mean" for n in (1, 2, 3)}, **{f"F{n}_tel": "mean" for n in (1, 2, 3)}}
agg = d.groupby(["hablante", "sesion", "vocal"]).agg(cols).reset_index()
CONDS = {"2 limpio-limpio": ("limpio", "limpio"), "3 telefonico-telefonico": ("tel", "tel"), "4 limpio-telefonico": ("limpio", "tel")}

filas = []
for nombre, (ca, cb) in CONDS.items():
    mis, dis = [], []
    for v, sub in agg.groupby("vocal"):
        A = sub[sub.sesion == "A"].set_index("hablante"); B = sub[sub.sesion == "B"].set_index("hablante")
        ia, ib = [f"F{n}_{ca}" for n in (1, 2, 3)], [f"F{n}_{cb}" for n in (1, 2, 3)]
        for hi in A.index:
            for hj in B.index:
                dif = np.abs(A.loc[hi, ia].values.astype(float) - B.loc[hj, ib].values.astype(float))
                (mis if hi == hj else dis).append(dif)
    m, t = np.nanmean(mis, axis=0), np.nanmean(dis, axis=0)
    for k, n in enumerate((1, 2, 3)):
        filas.append({"condicion": nombre, "formante": f"F{n}", "mismo_hablante_Hz": round(m[k]), "distinto_hablante_Hz": round(t[k]), "cociente": round(t[k] / m[k], 2)})
tabla = pd.DataFrame(filas)
tabla.to_csv("resultados/desglose_canal.csv", index=False)
print(tabla.to_string(index=False))

for n in (1, 2, 3):
    d[f"d{n}"] = d[f"F{n}_tel"] - d[f"F{n}_limpio"]
g = d.groupby("vocal")
por_vocal = pd.DataFrame({"mediana_dF2_Hz": g["d2"].median().round(0), "mediana_dF3_Hz": g["d3"].median().round(0),
                          "pct_dF2_mayor_400": (g["d2"].apply(lambda x: (x.abs() > 400).mean() * 100)).round(0),
                          "pct_dF3_mayor_400": (g["d3"].apply(lambda x: (x.abs() > 400).mean() * 100)).round(0)}).reset_index()
por_vocal.to_csv("resultados/desglose_canal_vocal.csv", index=False)
print(por_vocal.to_string(index=False))
