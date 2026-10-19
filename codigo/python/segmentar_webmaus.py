"""Segmentacion fonetica con WebMAUS (BAS): alinea cada audio limpio con su
transcripcion y guarda un TextGrid con los limites de cada fono en segmentacion/.
Reanudable: salta los que ya estan. Errores -> segmentacion/_errores.log
Uso:  python codigo/python/segmentar_webmaus.py
"""
import time
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm

# Decisiones (memoria, cap. 4.3):
# - Segmentacion con WebMAUS (runMAUSBasic), igual que Lopez-Escobedo et al. (2025).
# - Solo audios LIMPIOS: los mismos limites valen para la version telefonica.
# - Idioma: spa-ES.

API = "https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSBasic"
LANG = "spa-ES"
PAUSA = 0.8

df = pd.read_csv("datos/seleccion_final.csv", dtype=str).dropna(
        subset=["archivo_procesado", "texto"])
DEST = Path("segmentacion"); DEST.mkdir(exist_ok=True)
LOG = DEST / "_errores.log"
print(f"{len(df)} audios a segmentar (reanudable, ~2-3 h)")

ok = err = 0
for row in tqdm(df.itertuples(index=False), total=len(df)):
    wav = Path(row.archivo_procesado)
    tg = DEST / (wav.stem + ".TextGrid")
    if tg.exists() and tg.stat().st_size > 0:
        ok += 1
        continue
    try:
        with open(wav, "rb") as f:
            r = requests.post(API,
                files={"SIGNAL": f},
                data={"TEXT": row.texto, "LANGUAGE": LANG, "OUTFORMAT": "TextGrid"},
                timeout=180)
        r.raise_for_status()
        if "<success>true</success>" not in r.text:
            raise RuntimeError(r.text[:300])
        link = r.text.split("<downloadLink>")[1].split("</downloadLink>")[0].strip()
        g = requests.get(link, timeout=180); g.raise_for_status()
        tg.write_text(g.text, encoding="utf-8")
        ok += 1
    except Exception as e:
        err += 1
        with open(LOG, "a", encoding="utf-8") as lf:
            lf.write(f"{wav.name}\t{type(e).__name__}: {e}\n")
        if err <= 3 or err % 50 == 0:
            tqdm.write(f"  error en {wav.name}: {e}")
    time.sleep(PAUSA)

print(f"\nTextGrids OK: {ok} / {len(df)}   errores: {err}")
if err:
    print(f"Detalle en {LOG}")