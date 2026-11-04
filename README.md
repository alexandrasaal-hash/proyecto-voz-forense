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
│   └── r/            # MVKD con el paquete fvclrr (mvkd_formantes.R)
├── rasgos/           # formantes extraidos (rasgos/formantes.csv)
├── resultados/       # metricas, tablas y figuras
├── segmentacion/     # intento fallido de WebMAUS (descartado, ver mas abajo)
├── memoria/          # memoria del TFM
├── prompts/          # registro de uso de IA generativa
└── requirements.txt
```

## Entorno y reproducibilidad

El archivo `requirements.txt` fija las versiones de nueve bibliotecas de Python
documentadas en el entorno del proyecto (Python 3.11.9). Se han comprobado
las versiones de **ffmpeg 9.0.1** (compilación gyan.dev), **R 4.5.2** y
**fvclrr 1.2.0**, así como la disponibilidad de las nueve bibliotecas de Python
en el entorno original, sin conflictos detectados por `pip check`.

Además, se ha creado un entorno virtual independiente con Python 3.11.9 y se
han instalado desde cero las nueve dependencias con las versiones fijadas.
La instalación ha finalizado correctamente, `pip check` no ha encontrado
conflictos y se han importado satisfactoriamente las nueve bibliotecas.
Estas pruebas verifican la instalación y el funcionamiento básico del entorno,
pero **no equivalen a ejecutar el pipeline completo ni demuestran que las métricas
publicadas se obtuvieran originalmente con esas versiones**. La reproducción
completa de los resultados experimentales no se ha comprobado; requiere los
corpus originales, los archivos intermedios pertinentes y la ejecución del
procedimiento documentado.

Las dependencias del pipeline principal se agrupan en `requirements.txt`.
`librosa` se conserva para documentar el entorno, pero no es necesaria para los
scripts principales revisados. El intento de WebMAUS requiere `requests` (su
versión histórica no consta y no se declara como dependencia del pipeline).
Las dependencias de ECAPA-TDNN (`torch` y `speechbrain`) se incorporarán
únicamente cuando el experimento y sus versiones estén comprobados.

## Evaluación adicional por pareja (opción B, exploratoria)

Como ampliación de la Tabla 8 original, `codigo/python/comparacion_homogenea_B.py`
calcula una comparación con los mismos 22 hablantes y las mismas 484 parejas
ordenadas para MVKD y regresión logística (22 del mismo hablante, 462 de
hablantes distintos). Fusiona la información de las cinco vocales para obtener
una puntuación por pareja; la normalización y la regresión logística/fusión se
entrenan con LOSO. **No sustituye ni modifica la Tabla 8 original**.

Ejecución desde la raíz del proyecto, después de generar los archivos de
rasgos y resultados de MVKD:

```powershell
.\.venv\Scripts\python.exe codigo/python/comparacion_homogenea_B.py
```

El script genera `resultados/comparacion_B/comparacion_B_metricas.csv`,
`comparacion_B_puntuaciones.csv` y dos figuras (`figura_B_cllr_min.png` y
`figura_B_eer.png`). Las salidas de referencia están archivadas en la carpeta
`02_Proyecto/resultados/comparacion_B` de Google Drive.

**Limitación:** MVKD parte de puntuaciones precomputadas en R. Aún no se ha
verificado si el cálculo original de fondo excluye internamente al hablante
evaluado; por tanto, esta comparación es exploratoria y no debe interpretarse
como una evaluación forense independiente. El sistema principal y la fusión
se han reproducido desde `rasgos/formantes.csv` y los archivos intermedios,
pero no desde los audios originales.

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
leave-one-speaker-out. Como comparación, el método MVKD del estudio de referencia se ha
implementado con el paquete de R `fvclrr` (`codigo/r/mvkd_formantes.R`) y se evalúa con
`codigo/python/mvkd_evaluacion.py`.

## Uso de IA generativa

Este proyecto se ha desarrollado con apoyo de herramientas de IA generativa
(ChatGPT, OpenAI, y Claude, Anthropic), empleadas para consultas técnicas,
programación, depuración y revisión de la redacción. Su uso se declara en la
memoria del TFM, y el registro detallado está en
[`prompts/README.md`](prompts/README.md).

## Autora

Alexandra Santamaria Alcaraz — TFM, Máster Universitario en Big Data y Ciencia de Datos, VIU.
