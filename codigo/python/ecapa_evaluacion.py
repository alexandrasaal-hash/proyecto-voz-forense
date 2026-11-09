"""Apartado 4.5.2: evaluacion del sistema ECAPA-TDNN y comparacion con los otros dos sistemas.

Entrada (no se modifica nada):
  resultados/ecapa_embeddings.npz            embeddings ECAPA-TDNN de las 1920 grabaciones (limpio y telefonico),
                                             calculados con codigo/python/ecapa_embeddings.py
  resultados/mvkd_hablantes.csv              los 22 hablantes de la comparacion por pareja (apartado 4.5.1)
  resultados/comparacion_B_corregida/comparacion_B_metricas.csv    MVKD (fondo corregido) y regresion logistica
  resultados/comparacion_B_corregida/comparacion_B_puntuaciones.csv

Metodo (mismo diseno que comparacion_homogenea_B_corregida.py):
  1. Embedding de sesion = media de los embeddings normalizados de las 40 grabaciones de cada sesion (A y B).
  2. Puntuacion de cada pareja (sesion A del hablante i, sesion B del hablante j) = similitud coseno.
  3. Conversion a razon de verosimilitud (logLR) con regresion logistica sobre la similitud coseno, con
     validacion leave-one-speaker-out: se excluyen del entrenamiento todas las parejas del hablante evaluado.
  4. Metricas Cllr, Cllr_min y EER, como en el resto del trabajo.
Condiciones: limpio-limpio, telefonico-telefonico y cruzada (A limpia, B telefonica). ECAPA-TDNN no usa
formantes, por lo que no distingue F1-F4 de F1-F3.

Salida: resultados/comparacion_ecapa/ (CSV y figuras)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "resultados" / "comparacion_ecapa"
OUT.mkdir(exist_ok=True)
SIST_ECAPA = "ECAPA-TDNN (coseno + regresion logistica, LOSO)"
SIST_MVKD = "MVKD, fusion LOSO (fondo recalculado)"
SIST_LR = "Regresion logistica, cinco vocales por pareja"
CONDS = [("limpio", "Limpio", "limpio_F123"), ("telefono", "Telefónico", "telefono_F123"),
         ("cruzado", "Cruzado\n(limpio vs telefónico)", "cruzado_F123")]


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
    """un score por pareja del mismo hablante y la media de dos para las de hablantes distintos (como en 4.5.1)"""
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


# ---------- 1) embeddings de sesion
z = np.load(BASE / "resultados" / "ecapa_embeddings.npz", allow_pickle=True)
hab = pd.read_csv(BASE / "resultados" / "mvkd_hablantes.csv")["hablante"].astype(str).tolist()
assert len(hab) == 22 and len(set(hab)) == 22
hablante, sesion = z["hablante"].astype(str), z["sesion"].astype(str)
emb = {"limpio": unit(z["limpio"]), "telefono": unit(z["telefono"])}
sess = {}
for canal in ("limpio", "telefono"):
    for h in hab:
        for s in ("A", "B"):
            m = (hablante == h) & (sesion == s)
            assert m.sum() == 40, (h, s, int(m.sum()))
            sess[(canal, h, s)] = unit(emb[canal][m].mean(axis=0))

# ---------- 2) condiciones, puntuaciones y metricas
COMB = {"limpio": ("limpio", "limpio"), "telefono": ("telefono", "telefono"), "cruzado": ("limpio", "telefono")}
ids = [(i, j) for i in hab for j in hab]
sus = np.array([i for i, j in ids]); off = np.array([j for i, j in ids])
y = (sus == off).astype(int)
assert len(ids) == 484 and y.sum() == 22
rows, det = [], []
for cond, (ca, cb) in COMB.items():
    cos = np.array([float(sess[(ca, i, "A")] @ sess[(cb, j, "B")]) for i, j in ids])
    sc = loso_score(cos.reshape(-1, 1), y, sus, off, hab)
    rows.append({"condicion": cond, "sistema": SIST_ECAPA, "hablantes": 22, "pares": 484, "mismo": 22, "distinto": 462,
                 "Cllr": round(cllr(sc, y), 3), "Cllr_min": round(cmin(sc, y), 3), "EER": round(eer(sc, y), 4)})
    for (i, j), lab, c, s in zip(ids, y, cos, sc):
        det.append({"condicion": cond, "sistema": SIST_ECAPA, "sus": i, "off": j, "mismo": lab, "coseno": c, "score_logLR": s})
ecapa = pd.DataFrame(rows)
ecapa.to_csv(OUT / "ecapa_metricas.csv", index=False)
pd.DataFrame(det).to_csv(OUT / "ecapa_puntuaciones.csv", index=False)

# ---------- 3) tabla comparativa con los otros dos sistemas (mismas 484 parejas, formantes F1-F3)
otros = pd.read_csv(BASE / "resultados" / "comparacion_B_corregida" / "comparacion_B_metricas.csv")
punt = pd.read_csv(BASE / "resultados" / "comparacion_B_corregida" / "comparacion_B_puntuaciones.csv")
tabla = []
for cond, nombre, cf in CONDS:
    for sist in (SIST_LR, SIST_MVKD, SIST_ECAPA):
        f = ecapa[ecapa.condicion == cond] if sist == SIST_ECAPA else otros[(otros.condicion == cf) & (otros.sistema == sist)]
        r = f.iloc[0]
        tabla.append({"condicion": cond, "sistema": sist, "Cllr": r.Cllr, "Cllr_min": r.Cllr_min, "EER": r.EER})
tabla = pd.DataFrame(tabla)
tabla.to_csv(OUT / "comparacion_tres_sistemas.csv", index=False)
print(tabla.to_string(index=False))

# ---------- 4) figuras
NOM = {SIST_LR: "Regresión logística\n(formantes)", SIST_MVKD: "MVKD\n(formantes)", SIST_ECAPA: "ECAPA-TDNN\n(embeddings)"}
COL = {SIST_LR: "#6baed6", SIST_MVKD: "#fd8d3c", SIST_ECAPA: "#74c476"}
x = np.arange(len(CONDS)); w = .26
for metric, suf, ylab, top, fac in (("Cllr_min", "cllr_min", "Cllr_min (menor = mejor)", 1.05, 1), ("EER", "eer", "EER (%) (menor = mejor)", 50, 100)):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for k, sist in enumerate((SIST_LR, SIST_MVKD, SIST_ECAPA)):
        v = [float(tabla[(tabla.condicion == c) & (tabla.sistema == sist)][metric].iloc[0]) * fac for c, _, _ in CONDS]
        b = ax.bar(x + (k - 1) * w, v, w, label=NOM[sist].replace("\n", " "), color=COL[sist])
        ax.bar_label(b, fmt="%.1f" if metric == "EER" else "%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, [n for _, n, _ in CONDS]); ax.set_ylim(0, top); ax.set_ylabel(ylab)
    ax.set_title("Comparación de los tres sistemas (22 hablantes, 484 parejas)")
    ax.legend(loc="upper left", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / f"figura_tres_sistemas_{suf}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

def scores_de(cond, cf, sist):
    if sist == SIST_ECAPA:
        d = pd.DataFrame(det); d = d[d.condicion == cond]
    else:
        d = punt[(punt.condicion == cf) & (punt.sistema == sist)]
    return d.score_logLR.to_numpy(), d.mismo.to_numpy().astype(int)

# Tippett: una condicion por panel (cuadricula 2 x 2; el cuarto hueco es la leyenda). Se dibuja al tamaño al que se ve en la memoria
# (16 cm de ancho) para que la letra sea legible.
from matplotlib.lines import Line2D
with plt.rc_context({"font.size": 9}):
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 5.6), sharex=True, sharey=True)
    af = axs.ravel()
    for ax, (cond, nombre, cf) in zip(af[:3], CONDS):
        for sist in (SIST_LR, SIST_MVKD, SIST_ECAPA):
            s, lab = scores_de(cond, cf, sist)
            l10 = s / np.log(10)
            g = np.linspace(-6, 6, 400)
            ax.plot(g, [(l10[lab == 1] >= t).mean() for t in g], color=COL[sist], lw=1.8)
            ax.plot(g, [(l10[lab == 0] >= t).mean() for t in g], color=COL[sist], lw=1.8, ls="--")
        ax.axvline(0, color="grey", lw=.8); ax.set_xlim(-6, 6); ax.set_ylim(0, 1)
        ax.set_title(nombre.replace("\n", " "), fontsize=10, loc="left", fontweight="bold"); ax.spines[["top", "right"]].set_visible(False)
    for ax in (axs[1, 0], axs[1, 1]):
        ax.set_xlabel("log10(LR)")
    axs[0, 1].tick_params(labelbottom=True); axs[0, 1].set_xlabel("log10(LR)")
    for ax in (axs[0, 0], axs[1, 0]):
        ax.set_ylabel("Proporción de parejas con\nlog10(LR) ≥ valor")
    af[3].axis("off")
    leg = [Line2D([0], [0], color=COL[s], lw=2, label=NOM[s].replace("\n", " ")) for s in (SIST_LR, SIST_MVKD, SIST_ECAPA)]
    leg += [Line2D([0], [0], color="k", lw=2, label="mismo hablante"), Line2D([0], [0], color="k", lw=2, ls="--", label="hablantes distintos")]
    af[3].legend(handles=leg, loc="center", frameon=False, fontsize=9, title="Curvas de Tippett", title_fontsize=10)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(OUT / "figura_tres_sistemas_tippett.png", dpi=300); plt.close(fig)

# DET en la condicion cruzada (dibujada a ~11 cm de ancho, tamaño al que se ve en la memoria)
with plt.rc_context({"font.size": 9}):
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    for sist in (SIST_LR, SIST_MVKD, SIST_ECAPA):
        s, lab = scores_de("cruzado", "cruzado_F123", sist)
        fpr, tpr, _ = roc_curve(lab, s); fnr = 1 - tpr
        m = (fpr > 0) & (fpr < 1) & (fnr > 0) & (fnr < 1)
        ax.plot(norm.ppf(fpr[m]), norm.ppf(fnr[m]), color=COL[sist], lw=1.8, label=NOM[sist].replace("\n", " "))
    t = np.array([.01, .02, .05, .1, .2, .4, .6, .8, .95])
    ax.set_xticks(norm.ppf(t), [f"{100*v:g}" for v in t]); ax.set_yticks(norm.ppf(t), [f"{100*v:g}" for v in t])
    ax.set_xlim(norm.ppf(.01), norm.ppf(.95)); ax.set_ylim(norm.ppf(.01), norm.ppf(.95))
    ax.plot([-3, 3], [-3, 3], color="grey", lw=.8, ls=":")
    ax.set_xlabel("Falsas aceptaciones (%)"); ax.set_ylabel("Falsos rechazos (%)"); ax.set_title("Curvas DET, condición cruzada", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "figura_tres_sistemas_det_cruzado.png", dpi=300); plt.close(fig)

# similitud coseno de cada pareja de sesiones (un punto por pareja), una condicion debajo de otra
d = pd.DataFrame(det)
TITULOS = {"limpio": "Limpio", "telefono": "Telefónico", "cruzado": "Cruzado (limpio frente a telefónico)"}
C_DIST, C_MISMO = "#6a51a3", "#238b45"
rng_puntos = np.random.default_rng(1)
with plt.rc_context({"font.size": 10}):
    fig, axs = plt.subplots(3, 1, figsize=(6.3, 5.1), sharex=True)
    for ax, (cond, _, _) in zip(axs, CONDS):
        f = d[d.condicion == cond]
        dist = f[f.mismo == 0].coseno.to_numpy(); mis = f[f.mismo == 1].coseno.to_numpy()
        ax.scatter(dist, 0 + rng_puntos.uniform(-.3, .3, len(dist)), s=11, color=C_DIST, alpha=.45, linewidths=0, zorder=2)
        ax.scatter(mis, 1 + rng_puntos.uniform(-.3, .3, len(mis)), s=30, color=C_MISMO, alpha=.9, edgecolors="white", linewidths=.6, zorder=3)
        ax.vlines(np.median(dist), -.4, .4, color=C_DIST, lw=2.6, zorder=4)   # mediana de cada grupo
        ax.vlines(np.median(mis), .6, 1.4, color=C_MISMO, lw=2.6, zorder=4)
        mn, mx = mis.min(), dist.max()   # menor similitud del mismo hablante y mayor de hablantes distintos
        ax.axvline(mn, color=C_MISMO, lw=.9, ls=":", zorder=1); ax.axvline(mx, color=C_DIST, lw=.9, ls=":", zorder=1)
        ax.set_title(TITULOS[cond], fontsize=11, loc="left", fontweight="bold")
        ax.set_yticks([0, 1], [f"hablantes\ndistintos (n = {len(dist)})", f"mismo hablante\n(n = {len(mis)})"], fontsize=9)
        ax.set_ylim(-.6, 1.6); ax.set_xlim(-.1, 1.05)
        ax.text(.02, .96, f"mínimo del mismo hablante = {mn:.2f}\nmáximo de hablantes distintos = {mx:.2f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, color="#333333", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd", lw=.6), zorder=5)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6e6e6", lw=.6, zorder=0); ax.tick_params(axis="x", labelsize=10)
    axs[-1].set_xlabel("Similitud coseno entre sesiones", fontsize=11)
    fig.tight_layout(h_pad=1.4)
    fig.savefig(OUT / "figura_ecapa_distribuciones.png", dpi=300); plt.close(fig)
print("Tablas y figuras guardadas en", OUT)

# ---------- 5) comprobacion adicional: nivel de grabacion individual (sin promediar las 40 grabaciones de cada sesion)
# Cada grabacion de la sesion A se compara con cada grabacion de la sesion B (35.200 parejas del mismo hablante y
# 739.200 de hablantes distintos). Solo se usan metricas que dependen del orden de las puntuaciones (EER y Cllr_min).
idx = {(a, b): np.where((hablante == a) & (sesion == b))[0] for a in hab for b in ("A", "B")}
filas = []
for cond, (ca, cb) in COMB.items():
    sc_all, y_all = [], []
    for i in hab:
        for j in hab:
            m = emb[ca][idx[(i, "A")]] @ emb[cb][idx[(j, "B")]].T
            sc_all.append(m.ravel()); y_all.append(np.full(m.size, int(i == j)))
    sc_all = np.concatenate(sc_all); y_all = np.concatenate(y_all)
    filas.append({"condicion": cond, "nivel": "grabacion individual", "pares": len(y_all), "mismo": int(y_all.sum()),
                  "EER": round(eer(sc_all, y_all), 4), "Cllr_min": round(cmin(sc_all, y_all), 3)})
niv = pd.DataFrame(filas)
niv.to_csv(OUT / "ecapa_nivel_grabacion.csv", index=False)
print("\nNivel de grabacion individual (ECAPA-TDNN):\n" + niv.to_string(index=False))
