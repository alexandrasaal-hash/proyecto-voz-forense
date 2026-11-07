"""Figura del apartado 4.4: distribución de F1-F4 de una hablante y de un hablante,
por vocal, en audio limpio y en audio con canal telefónico simulado (violines).

Criterio de elección de la persona de cada sexo:
  - Mujer: cof_00610 (corpus SLR72), la misma que se muestra en la Ilustración 1
    (espectrograma de la Introducción), para seguir a la misma persona desde la
    Introducción hasta los Resultados.
  - Hombre: el más representativo entre los hombres con al menos MIN_NUCLEOS núcleos
    de cada una de las cinco vocales. El hombre de la Ilustración 2 (com_09334) no se
    usa porque tiene muy pocos núcleos en alguna vocal.
  "Más representativo" = menor distancia relativa entre sus medianas de F1-F3 (audio
  limpio, por vocal) y las medianas de F1-F3 de su sexo; es el mismo criterio de
  "valores próximos a las medianas" que se usó para la Ilustración 2.

Entrada:  rasgos/formantes.csv
Salidas (<sexo> = femenino | masculino):
          resultados/figura_violin_<sexo>.png
          resultados/violin_medianas_<sexo>.csv        (medianas y cuartiles por vocal, formante y condición)
          resultados/violin_comprobacion_<sexo>.csv    (representatividad de todos los hablantes del sexo)
Ejecutar desde la raíz del proyecto.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOCALES = ["a", "e", "i", "o", "u"]
MIN_NUCLEOS = 30
COLORES = {"limpio": "#4c78a8", "tel": "#e6862a"}
FIJA = {"femenino": "cof_00610", "masculino": None}   # None = el más representativo

d = pd.read_csv("rasgos/formantes.csv")
cols = ["F1_limpio", "F2_limpio", "F3_limpio"]

for SEXO, fija in FIJA.items():
    m = d[d.sexo == SEXO]
    # --- representatividad de todos los hablantes del sexo ---------------------
    med_grupo = m.groupby("vocal")[cols].median().loc[VOCALES]
    filas = []
    for h, g in m.groupby("hablante"):
        n = g.groupby("vocal").size().reindex(VOCALES).fillna(0)
        med = g.groupby("vocal")[cols].median().reindex(VOCALES)
        dist = (((med - med_grupo) / med_grupo) ** 2).sum().sum()
        filas.append({"hablante": h, "corpus": g.corpus.iloc[0], "nucleos_total": len(g),
                      "nucleos_min_por_vocal": int(n.min()),
                      "distancia_a_la_mediana": round(float(dist), 4)})
    crit = pd.DataFrame(filas)
    crit["cumple_minimo"] = crit.nucleos_min_por_vocal >= MIN_NUCLEOS
    crit = crit.sort_values(["cumple_minimo", "distancia_a_la_mediana"], ascending=[False, True])
    crit["rango"] = range(1, len(crit) + 1)
    crit.to_csv(f"resultados/violin_comprobacion_{SEXO}.csv", index=False)
    H = fija or crit.iloc[0].hablante
    print(f"\n[{SEXO}] hablante: {H}")
    print(crit.to_string(index=False))

    g = m[m.hablante == H]
    print(f"Núcleos: {len(g)} | por vocal: {g.groupby('vocal').size().reindex(VOCALES).to_dict()}"
          f" | F4_tel sin valor: {int(g.F4_tel.isna().sum())}")

    # --- medianas (para describir la figura) ---------------------------------
    filas = []
    for v in VOCALES:
        for n_f in (1, 2, 3, 4):
            for cond in ("limpio", "tel"):
                x = g.loc[g.vocal == v, f"F{n_f}_{cond}"].dropna()
                filas.append({"vocal": v, "formante": f"F{n_f}", "condicion": cond, "n": len(x),
                              "mediana_Hz": round(float(x.median()), 0) if len(x) else np.nan,
                              "p25_Hz": round(float(x.quantile(.25)), 0) if len(x) else np.nan,
                              "p75_Hz": round(float(x.quantile(.75)), 0) if len(x) else np.nan})
    pd.DataFrame(filas).to_csv(f"resultados/violin_medianas_{SEXO}.csv", index=False)

    # --- figura ------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    for ax, n_f in zip(axs.ravel(), (1, 2, 3, 4)):
        for cond, desp in (("limpio", -0.19), ("tel", 0.19)):
            datos = [g.loc[g.vocal == v, f"F{n_f}_{cond}"].dropna().values for v in VOCALES]
            ok = [len(x) > 1 for x in datos]
            vp = ax.violinplot([x for x, o in zip(datos, ok) if o],
                               positions=[i + desp for i, o in enumerate(ok) if o],
                               widths=0.34, showmedians=True, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(COLORES[cond]); b.set_alpha(0.65); b.set_edgecolor("none")
            vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(1)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"/{v}/" for v in VOCALES])
        ax.set_title(f"F{n_f}"); ax.set_ylabel("Hz"); ax.grid(axis="y", alpha=.25)
        ax.margins(y=0.12)
    axs[1, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=COLORES["limpio"], alpha=.65),
                              plt.Rectangle((0, 0), 1, 1, fc=COLORES["tel"], alpha=.65)],
                     labels=["Limpio", "Telefónico"], loc="upper right", frameon=False, ncol=2)
    fig.savefig(f"resultados/figura_violin_{SEXO}.png", dpi=200)
    plt.close(fig)
    print(f"Guardado resultados/figura_violin_{SEXO}.png")
