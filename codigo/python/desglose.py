"""Desglose de resultados por vocal y por sexo, complementando comparar_locutores.py.
Por sexo: compara el rendimiento usando SOLO pares de distinto locutor del MISMO sexo
(el caso dificil y mas realista) frente a usar todos los pares (que incluyen
comparaciones hombre-mujer, faciles de distinguir por F0 y que pueden inflar el
rendimiento aparente del sistema).
Salida: resultados/desglose_vocal.csv, resultados/desglose_sexo.csv
Uso:  python codigo/python/desglose.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.isotonic import IsotonicRegression

OUT = Path("resultados")
df = pd.read_csv("rasgos/formantes.csv")
agg = (df.groupby(["hablante", "sexo", "sesion", "vocal"])
         .agg(F1_limpio=("F1_limpio", "mean"), F2_limpio=("F2_limpio", "mean"),
              F3_limpio=("F3_limpio", "mean"), F4_limpio=("F4_limpio", "mean"),
              F1_tel=("F1_tel", "mean"), F2_tel=("F2_tel", "mean"),
              F3_tel=("F3_tel", "mean"))
         .reset_index())
sexo_hablante = agg.drop_duplicates("hablante").set_index("hablante")["sexo"]

CONDICIONES = {
    "limpio_F1234": (["F1_limpio", "F2_limpio", "F3_limpio", "F4_limpio"],
                      ["F1_limpio", "F2_limpio", "F3_limpio", "F4_limpio"]),
    "telefono_F123": (["F1_tel", "F2_tel", "F3_tel"], ["F1_tel", "F2_tel", "F3_tel"]),
    "cruzado_F123": (["F1_limpio", "F2_limpio", "F3_limpio"], ["F1_tel", "F2_tel", "F3_tel"]),
}


def cllr(llr, y):
    llr, y = np.asarray(llr, dtype=float), np.asarray(y)
    same, diff = llr[y == 1], llr[y == 0]
    return 0.5 * ((np.logaddexp(0, -same) / np.log(2)).mean() +
                  (np.logaddexp(0, diff) / np.log(2)).mean())


def cllr_min(llr, y):
    y = np.asarray(y)
    n_pos, n_neg = y.sum(), (y == 0).sum()
    peso = np.where(y == 1, 1.0 / n_pos, 1.0 / n_neg)
    iso = IsotonicRegression(out_of_bounds="clip")
    p = np.clip(iso.fit_transform(llr, y, sample_weight=peso), 1e-6, 1 - 1e-6)
    return cllr(np.log(p / (1 - p)), y)


def eer(llr, y):
    fpr, tpr, _ = roc_curve(y, llr)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[i] + fnr[i]) / 2


def construir_pares(cols_i, cols_j, vocales=None):
    filas = []
    sub_agg = agg if vocales is None else agg[agg.vocal.isin(vocales)]
    for vocal, sub in sub_agg.groupby("vocal"):
        A = sub[sub.sesion == "A"].set_index("hablante")
        B = sub[sub.sesion == "B"].set_index("hablante")
        for hi in A.index:
            vi = A.loc[hi, cols_i].values.astype(float)
            if np.isnan(vi).any():
                continue
            for hj in B.index:
                vj = B.loc[hj, cols_j].values.astype(float)
                if np.isnan(vj).any():
                    continue
                filas.append({"i": hi, "j": hj, "vocal": vocal,
                              "mismo": int(hi == hj), "dif": np.abs(vi - vj)})
    return pd.DataFrame(filas)


def loso_llr(trials):
    hablantes = sorted(set(trials["i"]) | set(trials["j"]))
    X = np.vstack(trials["dif"].values)
    y = trials["mismo"].values
    acumulado = {k: [] for k in trials.index}
    for s in hablantes:
        train_mask = (trials["i"] != s) & (trials["j"] != s)
        test_mask = (trials["i"] == s) | (trials["j"] == s)
        if train_mask.sum() < 10 or y[train_mask.values].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[train_mask.values], y[train_mask.values])
        # class_weight="balanced" ya equivale a probabilidades previas iguales: no se resta nada mas.
        scores = clf.decision_function(X[test_mask.values])
        for idx, sc in zip(trials.index[test_mask.values], scores):
            acumulado[idx].append(sc)
    llr = np.array([np.mean(v) if v else np.nan for v in acumulado.values()])
    ok = ~np.isnan(llr)
    return llr[ok], y[ok]


def evalua(trials):
    if trials.empty:
        return None
    llr, y = loso_llr(trials)
    if len(y) == 0 or y.sum() == 0 or (y == 0).sum() == 0:
        return None
    return {"n_pares": len(y), "n_mismo": int(y.sum()), "n_distinto": int((y == 0).sum()),
            "Cllr": round(cllr(llr, y), 3), "Cllr_min": round(cllr_min(llr, y), 3),
            "EER": round(eer(llr, y), 4)}


# --- 1. Desglose por vocal ---
filas_vocal = []
for nombre, (ci, cj) in CONDICIONES.items():
    for v in ["a", "e", "i", "o", "u"]:
        trials = construir_pares(ci, cj, vocales=[v])
        r = evalua(trials)
        if r:
            filas_vocal.append({"condicion": nombre, "vocal": v, **r})
tabla_vocal = pd.DataFrame(filas_vocal)
tabla_vocal.to_csv(OUT / "desglose_vocal.csv", index=False)

# --- 2. Desglose por sexo ---
filas_sexo = []
for nombre, (ci, cj) in CONDICIONES.items():
    trials = construir_pares(ci, cj)
    trials["sexo_i"] = trials["i"].map(sexo_hablante)
    trials["sexo_j"] = trials["j"].map(sexo_hablante)
    solo_mismo_sexo = trials[(trials["mismo"] == 1) | (trials["sexo_i"] == trials["sexo_j"])]
    r_ms = evalua(solo_mismo_sexo)
    if r_ms:
        filas_sexo.append({"condicion": nombre, "grupo": "solo_mismo_sexo", **r_ms})
    r_todos = evalua(trials)
    if r_todos:
        filas_sexo.append({"condicion": nombre, "grupo": "todos_los_pares", **r_todos})
tabla_sexo = pd.DataFrame(filas_sexo)
tabla_sexo.to_csv(OUT / "desglose_sexo.csv", index=False)

print("Por vocal:")
print(tabla_vocal.to_string(index=False))
print("\nPor sexo (solo_mismo_sexo = caso dificil, sin comparaciones hombre-mujer faciles):")
print(tabla_sexo.to_string(index=False))
