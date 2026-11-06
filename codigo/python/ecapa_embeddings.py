"""Apartado 4.5.3: embeddings de locutor con un modelo ECAPA-TDNN preentrenado (SpeechBrain, spkrec-ecapa-voxceleb).
Un embedding por grabacion y canal (limpio 16 kHz; telefonico 8 kHz remuestreado a 16 kHz).
Salida: resultados/ecapa_embeddings.npz
Uso (entorno con torch y speechbrain):  python codigo/python/ecapa_embeddings.py [n_max]
"""
import sys, time
import numpy as np
import pandas as pd
import soundfile as sf
import torch, torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else None
import os
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="C:/Temp/viu/ecapa_modelo", run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY)   # en Windows no se pueden crear enlaces simbolicos
df = pd.read_csv("datos/seleccion_final.csv", dtype=str).dropna(subset=["archivo_procesado", "archivo_telefono"])
if N_MAX: df = df.head(N_MAX)

def embedding(ruta):
    x, sr = sf.read(ruta, dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    x = torch.from_numpy(x)
    if sr != 16000: x = torchaudio.functional.resample(x, sr, 16000)
    with torch.no_grad():
        return clf.encode_batch(x.unsqueeze(0)).squeeze().numpy()

def guardar(filas):
    np.savez("resultados/ecapa_embeddings.npz", fileid=[f[0] for f in filas], hablante=[f[1] for f in filas], sesion=[f[2] for f in filas],
             limpio=np.vstack([f[3] for f in filas]), telefono=np.vstack([f[4] for f in filas]))

t0 = time.time(); filas = []
for i, r in enumerate(df.itertuples(index=False), 1):
    filas.append((r.fileid, r.hablante, r.sesion, embedding(r.archivo_procesado), embedding(r.archivo_telefono)))
    if i % 100 == 0 or i == len(df): print(f"{i}/{len(df)}  {time.time() - t0:.0f} s", flush=True)
    if i % 200 == 0 and i < len(df): guardar(filas)
guardar(filas)
print("Guardado resultados/ecapa_embeddings.npz", len(filas), "grabaciones")
