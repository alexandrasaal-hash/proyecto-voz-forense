"""Sistema de razon de verosimilitud por regresion logistica sobre diferencias
de formantes, con validacion leave-one-speaker-out. Se eligio inicialmente como sistema principal por restricciones de tiempo.
Posteriormente se implemento MVKD/fvclrr como metodo complementario de
comparacion (ver codigo/r/mvkd_formantes.R y codigo/python/mvkd_evaluacion.py).
Condiciones: limpio F1-F4, limpio F1-F3, telefono F1-F3, cruzado F1-F3
(sesion A limpia vs sesion B telefonica: escenario forense realista).
Salida: resultados/metricas.csv, resultados/tippett_<condicion>.png
Uso:  python codigo/python/comparar_locutores.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Decisiones (memoria, cap. 4.3-4.4):
# - MVKD (Aitken & Lucy 2004) sustituido por un clasificador de regresion logistica
#   sobre |diferencias| de formantes entre pares de grabaciones, siguiendo el
#   principio de calibracion por regresion logistica de Morrison (2011b).
#   Simplificacion respecto al metodo de referencia por restriccion de tiempo;
#   MVKD/fvclrr queda como linea de validacion futura.
# - Las 5 vocales (a, e, i, o, u) se combinan como observaciones repetidas del mismo par de
#   grabaciones (en vez de fusion posterior de sistemas por vocal, como en el
#   paper de referencia). Diferencia metodologica a declarar.
# - Validacion leave-one-speaker-out: para cada hablante excluido, se entrena
#   con el resto y se puntuan los pares que lo involucran (cada par se puntua
#   dos veces, una por cada hablante implicado, y se promedia).

OUT = Path("resultados")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("rasgos/formantes.csv")
agg = (df.groupby(["hablante", "sesion", "vocal"])
         .agg(F1_limpio=("F1_limpio", "mean"), F2_limpio=("F2_limpio", "mean"),
              F3_limpio=("F3_limpio", "mean"), F4_limpio=("F4_limpio", "mean"),
              F1_tel=("F1_tel", "mean"), F2_tel=("F2_tel", "mean"),
              F3_tel=("F3_tel", "mean"), n=("t", "size"))
         .reset_index())

CONDICIONES = {
    "limpio_F1234": (["F1_limpio", "F2_limpio", "F3_limpio", "F4_limpio"],
                      ["F1_limpio", "F2_limpio", "F3_limpio", "F4_limpio"]),
    "limpio_F123":  (["F1_limpio", "F2_limpio", "F3_limpio"],
                      ["F1_limpio", "F2_limpio", "F3_limpio"]),
    "telefono_F123": (["F1_tel", "F2_tel", "F3_tel"],
                       ["F1_tel", "F2_tel", "F3_tel"]),
    "cruzado_F123": (["F1_limpio", "F2_limpio", "F3_limpio"],
                      ["F1_tel", "F2_tel", "F3_tel"]),
}


def cllr(llr, y):
    llr, y = np.asarray(llr, dtype=float), np.asarray(y)
    same = llr[y == 1]
    diff = llr[y == 0]
    c_same = np.logaddexp(0, -same) / np.log(2)
    c_diff = np.logaddexp(0, diff) / np.log(2)
    return 0.5 * (c_same.mean() + c_diff.mean())


def cllr_min(llr, y):
    # Cllr pesa igual la clase "mismo" y "distinto" aunque haya muchas mas
    # comparaciones de distinto locutor: la recalibracion isotonica tiene que
    # optimizar con esa misma ponderacion balanceada, si no, no es un minimo
    # valido (y puede salir peor que el Cllr original).
    y = np.asarray(y)
    n_pos, n_neg = y.sum(), (y == 0).sum()
    peso = np.where(y == 1, 1.0 / n_pos, 1.0 / n_neg)
    iso = IsotonicRegression(out_of_bounds="clip")
    p = iso.fit_transform(llr, y, sample_weight=peso)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    llr_cal = np.log(p / (1 - p))
    return cllr(llr_cal, y)


def eer(llr, y):
    fpr, tpr, _ = roc_curve(y, llr)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[i] + fnr[i]) / 2


def construir_pares(cols_i, cols_j):
    filas = []
    for vocal, sub in agg.groupby("vocal"):
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
        # class_weight="balanced" ya equivale a probabilidades previas iguales, asi que
        # la salida (log-odds) ya es el log LR: restar ademas log(n_pos/n_neg) corregiria dos veces.
        scores = clf.decision_function(X[test_mask.values])
        for idx, sc in zip(trials.index[test_mask.values], scores):
            acumulado[idx].append(sc)

    llr = np.array([np.mean(v) if v else np.nan for v in acumulado.values()])
    ok = ~np.isnan(llr)
    return llr[ok], y[ok]


resumen = []
for nombre, (cols_i, cols_j) in CONDICIONES.items():
    trials = construir_pares(cols_i, cols_j)
    if trials.empty:
        print(f"{nombre}: sin pares validos, se omite")
        continue
    llr, y = loso_llr(trials)
    if len(y) == 0 or y.sum() == 0 or (y == 0).sum() == 0:
        print(f"{nombre}: datos insuficientes tras LOSO, se omite")
        continue
    np.savez(OUT / f"llr_{nombre}.npz", llr=llr, y=y)   # para figura_resultados.py
    llr10 = llr / np.log(10)
    c = cllr(llr, y)
    cmin = cllr_min(llr, y)
    e = eer(llr, y)
    resumen.append({"condicion": nombre, "n_pares": len(y),
                    "n_mismo": int(y.sum()), "n_distinto": int((y == 0).sum()),
                    "Cllr": round(c, 3), "Cllr_min": round(cmin, 3),
                    "EER": round(e, 4)})
    print(f"{nombre}: n={len(y)} (mismo={int(y.sum())}, distinto={int((y==0).sum())}) "
          f"Cllr={c:.3f} Cllr_min={cmin:.3f} EER={e:.3f}")

    ss = np.sort(llr10[y == 1])
    ds = np.sort(llr10[y == 0])
    plt.figure(figsize=(6, 4.5))
    plt.plot(ds, np.arange(1, len(ds) + 1) / len(ds), ":", label="Distinto locutor")
    plt.plot(ss, np.arange(1, len(ss) + 1) / len(ss), "-", label="Mismo locutor")
    plt.axvline(0, color="grey", lw=0.7)
    plt.xlabel("log10 LR")
    plt.ylabel("Proporcion acumulada")
    plt.title(f"Tippett - {nombre}  (Cllr={c:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"tippett_{nombre}.png", dpi=150)
    plt.close()

pd.DataFrame(resumen).to_csv(OUT / "metricas.csv", index=False)
print(f"\nGuardado: resultados/metricas.csv y resultados/tippett_*.png")
