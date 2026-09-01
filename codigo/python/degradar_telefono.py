import subprocess
from pathlib import Path

SRC = Path("datos/procesados")   # audios limpios
DST = Path("datos/telefono")     # salida degradada
DST.mkdir(parents=True, exist_ok=True)

AF = "highpass=f=300,lowpass=f=3400"

wavs = sorted(SRC.glob("*.wav"))

for wav in wavs:
    out = DST / wav.name
    cmd = [
        "ffmpeg", "-y", "-i", str(wav),
        "-ac", "1",
        "-ar", "8000",
        "-af", AF,
        str(out)
    ]

    print("->", wav.name)
    subprocess.run(cmd, check=True, capture_output=True)

print(f"Hecho: {len(list(DST.glob('*.wav')))} archivos en {DST}")
