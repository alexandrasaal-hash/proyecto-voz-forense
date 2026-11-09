"""Apartado 4.5.3: efecto de anadir F0 al sistema y clasificacion del sexo del locutor.
1) Sistema de comparacion (regresion logistica + leave-one-speaker-out) con F0 como rasgo adicional,
   en las cuatro condiciones de comparacion.
2) Clasificacion del sexo por sesion (F0, formantes, ambos), con leave-one-speaker-out, en limpio y telefonico.
Salidas: resultados/f0_sistema.csv y resultados/sexo_clasificacion.csv
Uso:  python codigo/python/f0_sexo.py
"""
import runpy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# reutiliza las funciones de comparar_locutores.py sin volver a ejecutar sus condiciones
src = open("codigo/python/comparar_locutores.py", encoding="utf-8").read()
corte = src.index("resumen = []")
ns = {}
exec(compile(src[:corte], "comparar_locutores.py", "exec"), ns)
cllr, cllr_min, eer, construir_pares, loso_llr = ns["cllr"], ns["cllr_min"], ns["eer"], ns["construir_pares"], ns["loso_llr"]
df = ns["df"]

agg = (df.groupby(["hablante", "sesion", "vocal"])
         .agg(F1_limpio=("F1_limpio", "mean"), F2_limpio=("F2_limpio", "mean"), F3_limpio=("F3_limpio", "mean"), F4_limpio=("F4_limpio", "mean"),
              F0_limpio=("F0_limpio", "mean"),
              F1_tel=("F1_tel", "mean"), F2_tel=("F2_tel", "mean"), F3_tel=("F3_tel", "mean"), F0_tel=("F0_tel", "mean"))
         .reset_index())
ns["agg"] = agg   # construir_pares usa la variable global 'agg'

L = ["F1_limpio", "F2_limpio", "F3_limpio"]; T = ["F1_tel", "F2_tel", "F3_tel"]
CONDS = {
    "limpio_F1234":  (L + ["F4_limpio"], L + ["F4_limpio"]),
    "limpio_F123":   (L, L),
    "telefono_F123": (T, T),
    "cruzado_F123":  (L, T),
}
filas = []
for nombre, (ci, cj) in CONDS.items():
    for con_f0 in (False, True):
        a = ci + (["F0_limpio"] if con_f0 else [])
        b = cj + ([("F0_tel" if "tel" in cj[0] else "F0_limpio")] if con_f0 else [])
        trials = construir_pares(a, b)
        llr, y = loso_llr(trials)
        filas.append({"condicion": nombre, "rasgos": "formantes + F0" if con_f0 else "formantes", "n_mismo": int(y.sum()), "n_distinto": int((y == 0).sum()),
                      "Cllr": round(cllr(llr, y), 3), "Cllr_min": round(cllr_min(llr, y), 3), "EER": round(eer(llr, y), 4)})
tabla = pd.DataFrame(filas)
tabla.to_csv("resultados/f0_sistema.csv", index=False)
print(tabla.to_string(index=False))

# ---- lo mismo, solo con parejas del mismo sexo (caso dificil: F0 no puede separar hombres de mujeres)
sexo_h = df.groupby("hablante")["sexo"].first().to_dict()
filas_ms = []
for nombre, (ci, cj) in CONDS.items():
    for con_f0 in (False, True):
        a_ = ci + (["F0_limpio"] if con_f0 else [])
        b_ = cj + ([("F0_tel" if "tel" in cj[0] else "F0_limpio")] if con_f0 else [])
        tr = construir_pares(a_, b_)
        tr = tr[(tr["mismo"] == 1) | (tr["i"].map(sexo_h) == tr["j"].map(sexo_h))].reset_index(drop=True)
        llr, y = loso_llr(tr)
        filas_ms.append({"condicion": nombre, "rasgos": "formantes + F0" if con_f0 else "formantes", "grupo": "solo_mismo_sexo", "n_mismo": int(y.sum()), "n_distinto": int((y == 0).sum()),
                         "Cllr": round(cllr(llr, y), 3), "Cllr_min": round(cllr_min(llr, y), 3), "EER": round(eer(llr, y), 4)})
tabla_ms = pd.DataFrame(filas_ms)
tabla_ms.to_csv("resultados/f0_sistema_mismo_sexo.csv", index=False)
print(); print(tabla_ms.to_string(index=False))

# ---- clasificacion del sexo por sesion (hablante-sesion), leave-one-speaker-out
fila = []
sx = df.groupby(["hablante", "sesion"]).agg(sexo=("sexo", "first"), F0_limpio=("F0_limpio", "median"), F0_tel=("F0_tel", "median"),
                                           F1_limpio=("F1_limpio", "median"), F2_limpio=("F2_limpio", "median"), F3_limpio=("F3_limpio", "median"),
                                           F1_tel=("F1_tel", "median"), F2_tel=("F2_tel", "median"), F3_tel=("F3_tel", "median")).reset_index()
sx["y"] = (sx["sexo"] == "femenino").astype(int)
CLASIF = {"limpio": {"solo F0": ["F0_limpio"], "solo F1-F3": ["F1_limpio", "F2_limpio", "F3_limpio"], "F0 + F1-F3": ["F0_limpio", "F1_limpio", "F2_limpio", "F3_limpio"]},
          "telefonico": {"solo F0": ["F0_tel"], "solo F1-F3": ["F1_tel", "F2_tel", "F3_tel"], "F0 + F1-F3": ["F0_tel", "F1_tel", "F2_tel", "F3_tel"]}}
for canal, grupos in CLASIF.items():
    for nombre, cols in grupos.items():
        X = sx[cols].values; y = sx["y"].values; acierta = []
        for h in sx["hablante"].unique():
            te = (sx["hablante"] == h).values
            clf = LogisticRegression(max_iter=5000, class_weight="balanced")
            m, s = X[~te].mean(axis=0), X[~te].std(axis=0)
            clf.fit((X[~te] - m) / s, y[~te])
            acierta.extend((clf.predict((X[te] - m) / s) == y[te]).tolist())
        fila.append({"canal": canal, "rasgos": nombre, "sesiones": len(acierta), "aciertos": int(sum(acierta)), "exactitud_pct": round(100 * np.mean(acierta), 1)})
res = pd.DataFrame(fila)
res.to_csv("resultados/sexo_clasificacion.csv", index=False)
print(); print(res.to_string(index=False))
print("\nF0 mediana (Hz): mujeres", round(sx[sx.y == 1].F0_limpio.median()), "| hombres", round(sx[sx.y == 0].F0_limpio.median()),
      "| tel: mujeres", round(sx[sx.y == 1].F0_tel.median()), "hombres", round(sx[sx.y == 0].F0_tel.median()))
