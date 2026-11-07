# Comparación forense de voz en español: efecto de la degradación del canal telefónico

Proyecto del Trabajo Fin de Máster (TFM) del Máster Universitario en Big Data y
Ciencia de Datos (VIU). Evalúa un sistema de comparación forense de voz basado en
formantes vocálicos y razón de verosimilitud, cuantificando el efecto de la
degradación del canal telefónico (limitación de banda, pérdida del cuarto formante)
sobre la fuerza de la evidencia, y comparándolo con un sistema de embeddings de red
neuronal (ECAPA-TDNN, apartado 4.5.2 de la memoria).

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
├── memoria/          # memoria del TFM (no se incluye en el repositorio)
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
El extractor de embeddings de ECAPA-TDNN (`codigo/python/ecapa_embeddings.py`) necesita además
`torch`, `torchaudio` y `speechbrain`, cuyas versiones no se han fijado ni comprobado; no están en
`requirements.txt`. La evaluación posterior (`codigo/python/ecapa_evaluacion.py`) solo utiliza las
bibliotecas de `requirements.txt` y parte de los embeddings ya calculados en
`resultados/ecapa_embeddings.npz`.

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

**Alcance de la versión B original:** las puntuaciones MVKD se calcularon
previamente con fvclrr (R). Se ha auditado su algoritmo y se ha observado que,
aunque excluye observaciones de los hablantes evaluados, reutiliza una media
poblacional calculada antes de esa exclusión. Por ello esta comparación original
se conserva como referencia exploratoria. El sistema principal y la fusión se
han reproducido desde `rasgos/formantes.csv` y los archivos intermedios,
pero no desde los audios originales.

## Comprobación adicional con fondo MVKD corregido

`codigo/python/mvkd_fondo_corregido.py` reconstruye en Python las expresiones
numéricas de fvclrr y recalcula la media y las matrices del fondo únicamente con
los hablantes restantes después de excluir a los implicados en cada pareja. En
las comparaciones del mismo hablante se promedian las 21 posibles exclusiones
adicionales de otro hablante para evitar aleatoriedad. Se ha contrastado la
formulación con 9.240 puntuaciones de distintos hablantes del cálculo R original:
9.239 coinciden con precisión mejor que 1,1 × 10⁻⁹ y una satura en R por límites
numéricos. La implementación estable usa limitación explícita del rango.

Después, `codigo/python/comparacion_homogenea_B_corregida.py` aplica LOSO a
la fusión de cinco vocales y a la regresión logística con las mismas 484 parejas
ordenadas por condición; las nuevas salidas se guardan en
`resultados/comparacion_B_corregida/`. Para reproducir desde la raíz:

```powershell
.\.venv\Scripts\python.exe codigo/python/mvkd_fondo_corregido.py
.\.venv\Scripts\python.exe codigo/python/comparacion_homogenea_B_corregida.py
```

Cllr_min MVKD corregido frente a regresión logística por pareja:
- Limpio F1–F4: 0,025 frente a 0,034.
- Limpio F1–F3: 0,062 frente a 0,066.
- Telefónico F1–F3: 0,080 frente a 0,198.
- Cruzado F1–F3: 0,833 frente a 0,520.

Esto es una **evaluación de sensibilidad con implementación independiente en
Python**, no una ejecución nueva de fvclrr en R ni una validación externa con
sesiones forenses reales. La Tabla 8 y la primera comparación B se conservan.
La ejecución completa desde los audios originales sigue sin realizarse.

## Evaluación con ECAPA-TDNN y comparación de los tres sistemas (apartado 4.5.2)

`codigo/python/ecapa_embeddings.py` extrae un embedding de 192 dimensiones por grabación y canal
(limpio y telefónico) con el modelo preentrenado `speechbrain/spkrec-ecapa-voxceleb`, sin
reentrenarlo, y guarda `resultados/ecapa_embeddings.npz` (1920 grabaciones).

`codigo/python/ecapa_evaluacion.py` usa esos embeddings con el mismo diseño de la comparación por
pareja: 22 hablantes y 484 parejas. El embedding de cada sesión es la media de los embeddings
normalizados de sus 40 grabaciones, la puntuación es la similitud coseno entre las sesiones A y B, y
se convierte en razón de verosimilitud con regresión logística y validación leave-one-speaker-out.
Ejecución desde cualquier carpeta:

```powershell
.\.venv\Scripts\python.exe codigo/python/ecapa_evaluacion.py
```

Salidas en `resultados/comparacion_ecapa/`: `comparacion_tres_sistemas.csv` (Cllr, Cllr_min y EER
de los tres sistemas), `ecapa_metricas.csv`, `ecapa_puntuaciones.csv`, `ecapa_nivel_grabacion.csv`
(comprobación sin promediar las 40 grabaciones de cada sesión) y cinco figuras (`figura_tres_sistemas_*.png`
y `figura_ecapa_distribuciones.png`).

Cllr_min / EER con los mismos 22 hablantes y 484 parejas:

| Condición | Regresión logística | MVKD | ECAPA-TDNN |
|---|---|---|---|
| Limpio | 0,066 / 3,6 % | 0,062 / 0,9 % | 0,000 / 0,0 % |
| Telefónico | 0,198 / 8,1 % | 0,080 / 4,0 % | 0,000 / 0,0 % |
| Cruzado | 0,520 / 23,1 % | 0,833 / 37,0 % | 0,022 / 0,7 % |

Cautelas: las sesiones A y B son pseudosesiones de un mismo conjunto de grabaciones; la media de 40
grabaciones por sesión favorece a ECAPA (por grabación individual la EER es de 1,8 %, 5,0 % y 5,0 %);
y solo hay 22 parejas del mismo hablante, por lo que un Cllr_min de 0 indica separación perfecta en
esta muestra y no un sistema perfecto. Los formantes y ECAPA-TDNN no usan la misma cantidad de
información, por lo que no son sistemas equivalentes.

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
   9. `codigo/python/ecapa_embeddings.py` — embeddings de locutor con ECAPA-TDNN preentrenado (necesita `torch` y `speechbrain`; se tarda un buen rato)
   10. `codigo/python/ecapa_evaluacion.py` — evaluación de ECAPA-TDNN y comparación con la regresión logística y MVKD (apartado 4.5.2)
   11. `codigo/python/figura_violin_hablante.py` — gráficos de violín de F1–F4 de una mujer y de un hombre, en limpio y telefónico (apartado 4.4.1); el criterio de elección de los hablantes está explicado al inicio del programa

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
