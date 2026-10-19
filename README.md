# Comparación forense de voz en español: efecto de la degradación del canal telefónico

Proyecto del Trabajo Fin de Máster (TFM) del Máster Universitario en Big Data y
Ciencia de Datos (VIU). Evalúa un sistema de comparación forense de voz basado en
formantes vocálicos y razón de verosimilitud, cuantificando el efecto de la
degradación del canal telefónico (limitación de banda, pérdida del cuarto formante)
sobre la fuerza de la evidencia, y comparándolo con un sistema de embeddings de red
neuronal (ECAPA-TDNN, pendiente de incorporar antes del depósito).

Toma como referencia metodológica el trabajo de López-Escobedo, Huerta-Pacheco y
Meza-Ruiz (2025), *The Voice of Frida Kahlo? An acoustic–phonetic approach to the case*
(IJSLL, https://doi.org/10.3138/ijsll-2024-0028).

## Estructura del proyecto

```
proyecto-voz-forense/
├── datos/            # datos crudos, procesados y telefonicos (no versionados)
├── codigo/
│   ├── python/       # scripts del pipeline, en orden de ejecucion (ver abajo)
│   └── r/            # fvclrr instalado; pendiente de usar para MVKD (trabajo futuro)
├── rasgos/           # formantes extraidos (rasgos/formantes.csv)
├── resultados/       # metricas, tablas y figuras
├── segmentacion/     # intento fallido de WebMAUS (descartado, ver mas abajo)
├── memoria/          # memoria del TFM
├── prompts/          # registro de uso de IA generativa
└── requirements.txt
```

## Datos

Se emplean los corpus públicos [OpenSLR SLR72](https://www.openslr.org/72/) (español
de Colombia) y [SLR73](https://www.openslr.org/73/) (español de Perú), publicados por
Google bajo licencia CC BY-SA 4.0 (Guevara-Rukoz et al., 2020). Los audios no se suben
al repositorio (ver `.gitignore`, por tamaño y por licencia): hay que descargarlos
aparte y colocarlos en `datos/crudos/openslr72/` y `datos/crudos/openslr73/`.

## Cómo reproducir el pipeline

1. Crear el entorno virtual e instalar dependencias:
   ```
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. Instalar [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) y añadirlo al PATH.
3. Descargar los datos (ver apartado anterior).
4. Ejecutar los scripts en este orden:

   1. `codigo/python/inventario_corpus.py` — inventario de hablantes por corpus
   2. `codigo/python/seleccionar_hablantes.py` — selección de 24 hablantes y pseudosesiones (A/B)
   3. `codigo/python/procesar_audios.py` — normalización de audio (mono, 16 kHz, 16 bits)
   4. `codigo/python/degradar_telefono.py` — condición telefónica simulada (300–3400 Hz, 8 kHz)
   5. `codigo/python/extraer_formantes.py` — segmentación local y extracción de F1–F4/F0
   6. `codigo/python/comparar_locutores.py` — sistema de razón de verosimilitud y evaluación (Cllr, EER, Tippett)
   7. `codigo/python/desglose.py` — desglose de resultados por vocal y por sexo
   8. `codigo/python/graficos.py` — figuras adicionales para la memoria

`codigo/python/segmentar_webmaus.py` documenta un intento de segmentación con el
servicio WebMAUS (el mismo método del estudio de referencia), que falló por un error
del servidor en el 100 % de las peticiones. Se conserva por transparencia
metodológica, pero no forma parte del pipeline final; su sustituto es
`extraer_formantes.py`.

## Método

Extracción de formantes (F1–F4) de las vocales /a/, /e/, /i/, /o/, /u/ mediante detección
acústica local (Parselmouth), y sistema de razón de verosimilitud mediante regresión
logística sobre diferencias de formantes entre pares de grabaciones, con validación
leave-one-speaker-out. Sustituye al método MVKD del estudio de referencia por
restricción de tiempo; una implementación de MVKD con el paquete de R `fvclrr` (ya
instalado en `codigo/r/`) queda como línea de trabajo futuro para el depósito.

## Uso de IA generativa

Este proyecto se ha desarrollado con apoyo de una herramienta de IA generativa
(Claude, Anthropic), declarado conforme a la normativa del máster. El registro
detallado de usos está en [`prompts/README.md`](prompts/README.md).

## Autora

Alexandra Santamaria Alcaraz — TFM, Máster Universitario en Big Data y Ciencia de Datos, VIU.
