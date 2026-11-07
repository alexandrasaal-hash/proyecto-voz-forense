"""Apartado 4.5.2: comprobaciones adicionales de ECAPA-TDNN (no modifica ningun resultado original).
Entradas: resultados/ecapa_embeddings.npz, resultados/mvkd_hablantes.csv, resultados/comparacion_ecapa/ecapa_puntuaciones.csv,
          resultados/comparacion_B_corregida/comparacion_B_puntuaciones.csv
  1) efecto del numero de grabaciones promediadas por sesion (1, 5, 10, 20, 40)             -> 1_numero_grabaciones.csv
  2) intervalos de confianza 95 % remuestreando HABLANTES (ECAPA, regresion logistica, MVKD) -> 2_intervalos_confianza_hablantes.csv
     (si un hablante sale repetido, se excluyen las parejas entre sus copias: cada posicion aporta una sola pareja positiva)
  3) control con etiquetas barajadas por hablante (procedimiento completo, 500 permutaciones) -> 3_etiquetas_barajadas.csv
  4) intervalos 95 % de la diferencia pareada entre sistemas                                -> 4_diferencias_pareadas.csv
  5) permutacion de identidades sobre las puntuaciones guardadas de cada sistema (2000)     -> 5_permutaciones_puntuaciones_guardadas.csv
  6) variante de 5 sin las 22 parejas reales del mismo hablante en la distribucion nula     -> 6_permutaciones_variante_sin_parejas_reales.csv
Ejecutar desde la raiz del proyecto con un entorno que tenga numpy, pandas y scikit-learn:
    python codigo/python/comprobaciones_ecapa.py
Salida: resultados/comparacion_ecapa/comprobaciones/  (no requiere los audios: parte de los embeddings y las puntuaciones ya calculadas)

Semillas aleatorias (fijas): sorteo de grabaciones por sesion en 1) default_rng(1000*n + repeticion); remuestreo de hablantes en 2) default_rng(7);
permutaciones del control completo en 3) default_rng(2026); remuestreo de las diferencias pareadas en 4) default_rng(11);
permutaciones sobre puntuaciones guardadas en 5) default_rng(99) y en 6) default_rng(5).
Permutaciones de identidades: se generan sin puntos fijos (permutaciones sin ningun elemento en su sitio, por rechazo), de modo que ningun hablante
conserva su pareja real en la distribucion nula. p empirico = (1 + n.o de permutaciones tan buenas o mejores que la real) / (n.o de permutaciones + 1).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

BASE = Path.cwd()
OUT = BASE / "resultados" / "comparacion_ecapa" / "comprobaciones"; OUT.mkdir(exist_ok=True, parents=True)
COMB = {"limpio": ("limpio", "limpio"), "telefono": ("telefono", "telefono"), "cruzado": ("limpio", "telefono")}


def cllr(s, y):
    return .5 * (np.logaddexp(0, -s[y == 1]).mean() + np.logaddexp(0, s[y == 0]).mean()) / np.log(2)


def cmin(s, y):
    w = np.where(y == 1, 1 / sum(y == 1), 1 / sum(y == 0))
    p = IsotonicRegression(out_of_bounds="clip").fit_transform(s, y, sample_weight=w)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return cllr(np.log(p / (1 - p)), y)


def eer(s, y):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.argmin(abs(fpr - fnr))
    return (fpr[i] + fnr[i]) / 2


def loso_score(X, y, sus, off, hab):
    acc = [[] for _ in y]
    for sp in hab:
        train = (sus != sp) & (off != sp)
        test = ~train
        z = StandardScaler().fit(X[train])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42).fit(z.transform(X[train]), y[train])
        for k, sc in zip(np.where(test)[0], clf.decision_function(z.transform(X[test]))):
            acc[k].append(float(sc))
    return np.array([np.mean(a) for a in acc])


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


z = np.load(BASE / "resultados" / "ecapa_embeddings.npz", allow_pickle=True)
hab = pd.read_csv(BASE / "resultados" / "mvkd_hablantes.csv")["hablante"].astype(str).tolist()
hablante, sesion = z["hablante"].astype(str), z["sesion"].astype(str)
emb = {"limpio": unit(z["limpio"]), "telefono": unit(z["telefono"])}
idx = {(h, s): np.where((hablante == h) & (sesion == s))[0] for h in hab for s in ("A", "B")}
assert all(len(v) == 40 for v in idx.values())
ids = [(i, j) for i in hab for j in hab]
sus = np.array([i for i, j in ids]); off = np.array([j for i, j in ids])
y = (sus == off).astype(int)
pos = {h: k for k, h in enumerate(hab)}


def sesion_emb(canal, n, rng):
    """embedding de sesion con n grabaciones elegidas al azar (n=40: todas)"""
    d = {}
    for h in hab:
        for s in ("A", "B"):
            ii = idx[(h, s)]
            if n < 40:
                ii = rng.choice(ii, n, replace=False)
            d[(h, s)] = unit(emb[canal][ii].mean(axis=0))
    return d


def cosenos(ca, cb, rng_a, rng_b, n):
    da = sesion_emb(ca, n, rng_a); db = sesion_emb(cb, n, rng_b)
    return np.array([float(da[(i, "A")] @ db[(j, "B")]) for i, j in ids])


# ------------------------------------------------------------------ 0) reproduce el resultado principal
print("== 0) control: reproduce los valores originales (n = 40)")
base = {}
for cond, (ca, cb) in COMB.items():
    cos = cosenos(ca, cb, None, None, 40)
    sc = loso_score(cos.reshape(-1, 1), y, sus, off, hab)
    base[cond] = (cos, sc)
    print(f"   {cond:9s} Cllr {cllr(sc, y):.3f}  Cllr_min {cmin(sc, y):.3f}  EER {eer(sc, y):.4f}")

# ------------------------------------------------------------------ 1) numero de grabaciones
print("\n== 1) numero de grabaciones por sesion (30 repeticiones al azar; media [min-max])")
R = 30
filas = []
for n in (1, 5, 10, 20, 40):
    for cond, (ca, cb) in COMB.items():
        rep = 1 if n == 40 else R
        vals = []
        for r in range(rep):
            rng = np.random.default_rng(1000 * n + r)
            cos = cosenos(ca, cb, rng, rng, n)
            sc = loso_score(cos.reshape(-1, 1), y, sus, off, hab)
            vals.append((cllr(sc, y), cmin(sc, y), eer(sc, y)))
        v = np.array(vals)
        filas.append({"n_grabaciones": n, "condicion": cond, "repeticiones": rep,
                      "Cllr_media": v[:, 0].mean(), "Cllr_min_media": v[:, 1].mean(), "Cllr_min_min": v[:, 1].min(),
                      "Cllr_min_max": v[:, 1].max(), "EER_media": v[:, 2].mean(), "EER_min": v[:, 2].min(), "EER_max": v[:, 2].max()})
        f = filas[-1]
        print(f"   n={n:2d} {cond:9s} Cllr_min {f['Cllr_min_media']:.3f} [{f['Cllr_min_min']:.3f}-{f['Cllr_min_max']:.3f}]   "
              f"EER {100*f['EER_media']:.1f}% [{100*f['EER_min']:.1f}-{100*f['EER_max']:.1f}]", flush=True)
pd.DataFrame(filas).round(4).to_csv(OUT / "1_numero_grabaciones.csv", index=False)

# ------------------------------------------------------------------ 2) intervalos de confianza por hablante
print("\n== 2) IC 95 % remuestreando hablantes (2000 remuestreos; puntuaciones LOSO fijas)")
B = 2000
M = lambda v: np.asarray(v).reshape(22, 22)


def remuestra(M, b):
    """Remuestreo por hablante (posiciones b, con repeticion). Cada posicion aporta UNA pareja del mismo hablante (la diagonal);
    las parejas entre dos copias del mismo hablante se excluyen (serian la misma pareja positiva repetida) y el resto son
    parejas de hablantes distintos. Devuelve puntuaciones y etiquetas (1 = mismo hablante).
    Cada remuestreo tiene siempre 22 parejas positivas; el numero de parejas negativas varia (484 - suma de k^2, con k = veces que sale cada
    hablante), y una pareja de hablantes distintos (i, j) aparece k_i * k_j veces. Los sistemas se comparan siempre sobre las mismas parejas."""
    m = M[np.ix_(b, b)]
    diag = np.eye(len(b), dtype=bool)
    keep = diag | ~(b[:, None] == b[None, :])
    return m[keep], diag[keep].astype(int)


def ic(score_mat, rng):
    out = []
    for _ in range(B):
        b = rng.integers(0, 22, 22)
        s, lab = remuestra(score_mat, b)
        out.append((cllr(s, lab), cmin(s, lab), eer(s, lab)))
    o = np.array(out)
    return np.percentile(o, [2.5, 97.5], axis=0), o.mean(axis=0)


otros = pd.read_csv(BASE / "resultados" / "comparacion_B_corregida" / "comparacion_B_puntuaciones.csv")
filas = []
rng = np.random.default_rng(7)
for cond, cf in (("limpio", "limpio_F123"), ("telefono", "telefono_F123"), ("cruzado", "cruzado_F123")):
    sistemas = {"ECAPA-TDNN": base[cond][1]}
    for nombre in otros.sistema.unique():
        d = otros[(otros.condicion == cf) & (otros.sistema == nombre)]
        d = d.set_index(["sus", "off"]).loc[ids]       # mismo orden de parejas
        assert (d.mismo.to_numpy() == y).all()
        sistemas[nombre] = d.score_logLR.to_numpy()
    for nom, s in sistemas.items():
        (lo, hi), _ = ic(M(s), rng)
        pt = (cllr(s, y), cmin(s, y), eer(s, y))
        filas.append({"condicion": cond, "sistema": nom, "Cllr": pt[0], "Cllr_IC_inf": lo[0], "Cllr_IC_sup": hi[0],
                      "Cllr_min": pt[1], "Cllr_min_IC_inf": lo[1], "Cllr_min_IC_sup": hi[1],
                      "EER": pt[2], "EER_IC_inf": lo[2], "EER_IC_sup": hi[2]})
        f = filas[-1]
        print(f"   {cond:9s} {nom[:28]:28s} Cllr_min {pt[1]:.3f} [{lo[1]:.3f}-{hi[1]:.3f}]  EER {100*pt[2]:.1f}% [{100*lo[2]:.1f}-{100*hi[2]:.1f}]", flush=True)
pd.DataFrame(filas).round(4).to_csv(OUT / "2_intervalos_confianza_hablantes.csv", index=False)

# ------------------------------------------------------------------ 3) etiquetas barajadas
print("\n== 3) etiquetas barajadas por hablante (500 permutaciones de identidades de la sesion B, sin puntos fijos)")
P = 500
rng = np.random.default_rng(2026)
perms = []
while len(perms) < P:
    p = rng.permutation(22)
    if not (p == np.arange(22)).any():
        perms.append(p)
filas = []
for cond in COMB:
    cos = base[cond][0].reshape(22, 22)             # filas = sesion A de i, columnas = sesion B de j
    obs = (cllr(base[cond][1], y), cmin(base[cond][1], y), eer(base[cond][1], y))
    res = []
    for p in perms:
        # la sesion B del hablante j pasa a llamarse p[j]: el "mismo hablante" es ahora un par de personas distintas
        yy = np.array([[1 if p[j] == i else 0 for j in range(22)] for i in range(22)]).ravel()
        # entrenamiento LOSO sobre las etiquetas barajadas, excluyendo por hablante real de cada lado (A: i, B: j)
        sc = loso_score(cos.reshape(-1, 1), yy, sus, off, hab)
        res.append((cllr(sc, yy), cmin(sc, yy), eer(sc, yy)))
    r = np.array(res)
    filas.append({"condicion": cond, "permutaciones": P, "EER_observada": obs[2], "EER_barajada_media": r[:, 2].mean(),
                  "EER_barajada_min": r[:, 2].min(), "EER_barajada_p2_5": np.percentile(r[:, 2], 2.5),
                  "Cllr_min_observada": obs[1], "Cllr_min_barajada_media": r[:, 1].mean(), "Cllr_min_barajada_min": r[:, 1].min(),
                  "Cllr_barajada_media": r[:, 0].mean(),
                  "permutaciones_con_EER_menor_o_igual_que_la_observada": int((r[:, 2] <= obs[2]).sum())})
    f = filas[-1]
    print(f"   {cond:9s} EER barajada media {100*f['EER_barajada_media']:.1f}% (min {100*f['EER_barajada_min']:.1f}%) | "
          f"Cllr_min barajada media {f['Cllr_min_barajada_media']:.3f} (min {f['Cllr_min_barajada_min']:.3f}) | "
          f"perm. tan buenas como la real: {f['permutaciones_con_EER_menor_o_igual_que_la_observada']}/{P}", flush=True)
pd.DataFrame(filas).round(4).to_csv(OUT / "3_etiquetas_barajadas.csv", index=False)
print("\nGuardado en", OUT)


# ================= 4) y 5) con las puntuaciones LOSO guardadas =================
SIS = {"ECAPA": None, "LR": "Regresion logistica, cinco vocales por pareja", "MVKD": "MVKD, fusion LOSO (fondo recalculado)"}
ec = pd.read_csv(BASE / "resultados" / "comparacion_ecapa" / "ecapa_puntuaciones.csv")
ot = pd.read_csv(BASE / "resultados" / "comparacion_B_corregida" / "comparacion_B_puntuaciones.csv")
CF = {"limpio": "limpio_F123", "telefono": "telefono_F123", "cruzado": "cruzado_F123"}


def mat(cond, sis):
    if sis == "ECAPA":
        d = ec[ec.condicion == cond]
    else:
        d = ot[(ot.condicion == CF[cond]) & (ot.sistema == SIS[sis])]
    d = d.set_index(["sus", "off"]).loc[ids]
    assert (d.mismo.to_numpy() == y).all()
    return d.score_logLR.to_numpy().reshape(22, 22)


S = {(c, s): mat(c, s) for c in CF for s in SIS}

# ---------------- A) diferencias pareadas
B = 2000
rng = np.random.default_rng(11)
boots = [rng.integers(0, 22, 22) for _ in range(B)]
filas = []
print("== A) diferencia pareada (sistema 1 - sistema 2); negativo = el sistema 1 tiene MENOS error")
for cond in CF:
    for a, b in (("ECAPA", "LR"), ("ECAPA", "MVKD"), ("LR", "MVKD")):
        dc, de = [], []
        for bb in boots:
            sa, lab = remuestra(S[(cond, a)], bb); sb, _ = remuestra(S[(cond, b)], bb)
            dc.append(cmin(sa, lab) - cmin(sb, lab)); de.append(eer(sa, lab) - eer(sb, lab))
        pc = cmin(S[(cond, a)].ravel(), y) - cmin(S[(cond, b)].ravel(), y)
        pe = eer(S[(cond, a)].ravel(), y) - eer(S[(cond, b)].ravel(), y)
        lc = np.percentile(dc, [2.5, 97.5]); le = np.percentile(de, [2.5, 97.5])
        filas.append({"condicion": cond, "sistema_1": a, "sistema_2": b, "dif_Cllr_min": pc, "IC_inf": lc[0], "IC_sup": lc[1],
                      "dif_EER": pe, "EER_IC_inf": le[0], "EER_IC_sup": le[1]})
        print(f"   {cond:9s} {a:5s}-{b:5s}  dCllr_min {pc:+.3f} [{lc[0]:+.3f},{lc[1]:+.3f}]   dEER {100*pe:+.1f} pts [{100*le[0]:+.1f},{100*le[1]:+.1f}]", flush=True)
pd.DataFrame(filas).round(4).to_csv(OUT / "4_diferencias_pareadas.csv", index=False)

# ---------------- B) permutaciones sobre las puntuaciones guardadas
P = 2000
rng = np.random.default_rng(99)
perms = []
while len(perms) < P:
    p = rng.permutation(22)
    if not (p == np.arange(22)).any():
        perms.append(p)
print("\n== B) permutacion de identidades sobre las puntuaciones guardadas (2000 permutaciones)")
filas = []
for cond in CF:
    for s in SIS:
        M = S[(cond, s)]
        oc, oe = cmin(M.ravel(), y), eer(M.ravel(), y)
        nc, ne = [], []
        for p in perms:
            yy = (np.arange(22)[None, :] == p[:, None]).ravel().astype(int)   # fila i: su "pareja" es la columna p[i]
            nc.append(cmin(M.ravel(), yy)); ne.append(eer(M.ravel(), yy))
        nc, ne = np.array(nc), np.array(ne)
        pc = (1 + (nc <= oc).sum()) / (P + 1); pe = (1 + (ne <= oe).sum()) / (P + 1)
        filas.append({"condicion": cond, "sistema": s, "Cllr_min": oc, "EER": oe, "Cllr_min_azar_media": nc.mean(), "Cllr_min_azar_min": nc.min(),
                      "EER_azar_media": ne.mean(), "EER_azar_min": ne.min(), "p_Cllr_min": pc, "p_EER": pe})
        print(f"   {cond:9s} {s:5s} Cllr_min {oc:.3f} (azar: media {nc.mean():.3f}, min {nc.min():.3f}) p={pc:.4f} | "
              f"EER {100*oe:.1f}% (azar: media {100*ne.mean():.1f}%, min {100*ne.min():.1f}%) p={pe:.4f}", flush=True)
pd.DataFrame(filas).round(4).to_csv(OUT / "5_permutaciones_puntuaciones_guardadas.csv", index=False)
print("\nGuardado en", OUT)


# ================= 6) variante sin las parejas reales en la distribucion nula =================
P=2000; rng=np.random.default_rng(5); perms=[]
while len(perms)<P:
    p=rng.permutation(22)
    if not (p==np.arange(22)).any(): perms.append(p)
off_diag=~np.eye(22,dtype=bool).ravel()
print("Variante: se EXCLUYEN las 22 parejas reales del mismo hablante en la distribucion nula (22 falsas 'mismo' vs 440 distintas)")
filas6=[]
for cond in CF:
    for s in SIS:
        M=S[(cond,s)]; oc,oe=cmin(M.ravel(),y),eer(M.ravel(),y); nc=[];ne=[]
        for p in perms:
            yy=(np.arange(22)[None,:]==p[:,None]).ravel().astype(int)
            sc=M.ravel()[off_diag]; ll=yy[off_diag]
            nc.append(cmin(sc,ll)); ne.append(eer(sc,ll))
        nc,ne=np.array(nc),np.array(ne)
        print(f"  {cond:9s} {s:5s} Cllr_min {oc:.3f} azar media {nc.mean():.3f} min {nc.min():.3f} p={(1+(nc<=oc).sum())/(P+1):.4f} | EER {100*oe:.1f}% azar media {100*ne.mean():.1f}% min {100*ne.min():.1f}% p={(1+(ne<=oe).sum())/(P+1):.4f}")
        filas6.append({"condicion": cond, "sistema": s, "Cllr_min": oc, "EER": oe, "Cllr_min_azar_media": nc.mean(), "Cllr_min_azar_min": nc.min(),
                       "EER_azar_media": ne.mean(), "EER_azar_min": ne.min(),
                       "p_Cllr_min": (1+(nc<=oc).sum())/(P+1), "p_EER": (1+(ne<=oe).sum())/(P+1)})
pd.DataFrame(filas6).round(4).to_csv(OUT / "6_permutaciones_variante_sin_parejas_reales.csv", index=False)
print("Guardado 6_permutaciones_variante_sin_parejas_reales.csv")
