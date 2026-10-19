"""Apartado 4.5.2: evalua el MVKD (resultados de codigo/r/mvkd_formantes.R) y lo compara con la regresion logistica
sobre los mismos 22 hablantes.
- MVKD por vocal (logLR de fvclrr) -> fusion de las cinco vocales con regresion logistica, con leave-one-speaker-out.
- Regresion logistica del trabajo (diferencias de formantes), restringida a los mismos hablantes.
Salidas: resultados/mvkd_resumen.csv y resultados/mvkd_por_vocal.csv
Uso:  python codigo/python/mvkd_evaluacion.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

src = open("codigo/python/comparar_locutores.py", encoding="utf-8").read()
ns = {}
exec(compile(src[:src.index("resumen = []")], "comparar_locutores.py", "exec"), ns)
cllr, cllr_min, eer, construir_pares, loso_llr, CONDICIONES = ns["cllr"], ns["cllr_min"], ns["eer"], ns["construir_pares"], ns["loso_llr"], ns["CONDICIONES"]

hab = pd.read_csv("resultados/mvkd_hablantes.csv")["hablante"].tolist()
m = pd.read_csv("resultados/mvkd_pares.csv")
VOCALES = ["a", "e", "i", "o", "u"]
CLIP = 30.0

def metricas(llr, y):
    return {"Cllr": round(cllr(llr, y), 3), "Cllr_min": round(cllr_min(llr, y), 3), "EER": round(eer(llr, y), 4)}

filas, vocal_f = [], []
for cond in CONDICIONES:
    sub = m[m.condicion == cond]
    # --- por vocal
    for v in VOCALES:
        s = sub[sub.vocal == v]
        vocal_f.append({"condicion": cond, "vocal": v, **metricas(np.clip(s.logLR.values, -CLIP, CLIP), s.mismo.values)})
    # --- fusion de vocales: regresion logistica con leave-one-speaker-out
    ancho = sub.pivot_table(index=["sus", "off", "mismo"], columns="vocal", values="logLR").reset_index()
    X = np.clip(ancho[VOCALES].values, -CLIP, CLIP); y = ancho["mismo"].values
    acum = {i: [] for i in range(len(ancho))}
    for s_ in hab:
        tr = ((ancho.sus != s_) & (ancho.off != s_)).values; te = ~tr
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
        clf.fit((X[tr] - mu) / sd, y[tr])
        for i, sc in zip(np.where(te)[0], clf.decision_function((X[te] - mu) / sd)): acum[i].append(sc)
    llr_f = np.array([np.mean(v) for v in acum.values()])
    filas.append({"condicion": cond, "sistema": "MVKD (fusion de vocales)", "n_mismo": int(y.sum()), "n_distinto": int((y == 0).sum()), **metricas(llr_f, y)})
    # --- regresion logistica del trabajo, mismos hablantes
    ci, cj = CONDICIONES[cond]
    tr = construir_pares(ci, cj)
    tr = tr[tr["i"].isin(hab) & tr["j"].isin(hab)].reset_index(drop=True)
    llr_l, y_l = loso_llr(tr)
    filas.append({"condicion": cond, "sistema": "Regresion logistica (diferencias)", "n_mismo": int(y_l.sum()), "n_distinto": int((y_l == 0).sum()), **metricas(llr_l, y_l)})
res = pd.DataFrame(filas); res.to_csv("resultados/mvkd_resumen.csv", index=False)
pv = pd.DataFrame(vocal_f); pv.to_csv("resultados/mvkd_por_vocal.csv", index=False)
print(res.to_string(index=False)); print(); print(pv.to_string(index=False))
