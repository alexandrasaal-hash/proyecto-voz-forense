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
`torch`, `torchaudio` y `speechbrain`. Sus versiones (las del entorno virtual con el que se extrajeron los
embeddings) están en `requirements-ecapa.txt`, que se instala **en un entorno aparte** y no forma parte de
`requirements.txt` (torch es pesado). La evaluación posterior (`codigo/python/ecapa_evaluacion.py`) solo
utiliza las bibliotecas de `requirements.txt` (entre ellas matplotlib, para las figuras) y las comprobaciones
adicionales (`codigo/python/comprobaciones_ecapa.py`) solo numpy, pandas y scikit-learn; ambas parten de los
embeddings ya calculados en `resultados/ecapa_embeddings.npz`.

## Evaluación adicional por pareja (opción B, exploratoria)

Como ampliación de la Tabla 9 original, `codigo/python/comparacion_homogenea_B.py`
calcula una comparación con los mismos 22 hablantes y las mismas 484 parejas
ordenadas para MVKD y regresión logística (22 del mismo hablante, 462 de
hablantes distintos). Fusiona la información de las cinco vocales para obtener
una puntuación por pareja; la normalización y la regresión logística/fusión se
entrenan con LOSO. **No sustituye ni modifica la Tabla 9 original**.

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
sesiones forenses reales. La Tabla 9 y la primera comparación B se conservan.
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

### Comprobaciones adicionales de ECAPA-TDNN (apartado 4.5.2)

`codigo/python/comprobaciones_ecapa.py` reúne las comprobaciones que refuerzan la lectura de los resultados
de ECAPA-TDNN (sección «Comprobaciones adicionales del rendimiento de ECAPA-TDNN» de la memoria). **No usa los
audios**: parte de los embeddings y de las puntuaciones ya calculadas.

Entradas: `resultados/ecapa_embeddings.npz`, `resultados/mvkd_hablantes.csv`,
`resultados/comparacion_ecapa/ecapa_puntuaciones.csv` y
`resultados/comparacion_B_corregida/comparacion_B_puntuaciones.csv`.

Ejecución desde la raíz del proyecto, en el entorno de `requirements-ecapa.txt` (el mismo con el que se
generaron los CSV):

```powershell
py -m venv .venv-ecapa
.\.venv-ecapa\Scripts\python.exe -m pip install -r requirements-ecapa.txt
.\.venv-ecapa\Scripts\python.exe codigo/python/comprobaciones_ecapa.py
```

Salidas en `resultados/comparacion_ecapa/comprobaciones/`:

| Archivo | Contenido |
|---|---|
| `1_numero_grabaciones.csv` | EER y Cllr con 1, 5, 10, 20 y 40 grabaciones promediadas por sesión (30 sorteos al azar) |
| `2_intervalos_confianza_hablantes.csv` | Intervalos 95 % aproximados del Cllr, Cllr_min y EER remuestreando hablantes (2000 repeticiones); las parejas entre copias de un mismo hablante no se contabilizan como nuevas parejas positivas |
| `3_etiquetas_barajadas.csv` | Control con identidades permutadas repitiendo el procedimiento completo, validación LOSO incluida (500 permutaciones) |
| `4_diferencias_pareadas.csv` | Intervalos 95 % aproximados de la diferencia entre sistemas, con los mismos remuestreos (mismas parejas en cada remuestreo) |
| `5_permutaciones_puntuaciones_guardadas.csv` | Permutación de identidades sobre las puntuaciones ya calculadas, sin reentrenar (2000 permutaciones) |
| `6_permutaciones_variante_sin_parejas_reales.csv` | Variante de la anterior sin las 22 parejas reales en la distribución nula |

Todas las semillas aleatorias son fijas y están documentadas al inicio del programa. Las permutaciones se
generan sin puntos fijos (ningún hablante conserva su pareja real) y el p empírico es
(1 + n.º de permutaciones tan buenas o mejores que la real) / (n.º de permutaciones + 1); con 2000
permutaciones, el menor valor estimable es 1/2001 ≈ 0,0005.

**Dos niveles de reproducibilidad, que no deben confundirse:**

1. *Repetir las comprobaciones desde los embeddings guardados* (lo anterior). Comprobado: una ejecución en
   una carpeta limpia, con solo las cuatro entradas, reproduce los seis CSV byte a byte.
2. *Reproducir la extracción de los embeddings desde los audios* (`ecapa_embeddings.py`, que requiere los datos
   de `03_Datos` y el entorno de `requirements-ecapa.txt`). Solo se ha comprobado parcialmente: ocho embeddings
   recalculados (cuatro grabaciones, en limpio y telefónico) coinciden exactamente con los guardados
   (diferencia máxima 0). No se ha repetido la extracción completa de las 1920 grabaciones ni el pipeline desde
   los audios originales.

En `requirements-ecapa.txt`, `torch` y `torchaudio` figuran con las versiones del entorno original
(2.14.1 y 2.11.0); `torchaudio` no declara una dependencia estricta de `torch`, de modo que la combinación se ha
verificado solo en ese entorno (`pip check` sin conflictos, importación y remuestreo correctos y embeddings
idénticos). Que las comprobaciones sean reproducibles no elimina las limitaciones del diseño (pseudosesiones,
22 hablantes, canal telefónico simulado).

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
   9. `codigo/python/ecapa_embeddings.py` — embeddings de locutor con ECAPA-TDNN preentrenado (necesita `torch`, `torchaudio` y `speechbrain`: entorno aparte con `requirements-ecapa.txt`; se tarda un buen rato)
   10. `codigo/python/ecapa_evaluacion.py` — evaluación de ECAPA-TDNN y comparación con la regresión logística y MVKD (apartado 4.5.2)
   11. `codigo/python/figura_violin_hablante.py` — gráficos de violín de F1–F4 de una mujer y de un hombre, en limpio y telefónico (apartado 4.4.1); el criterio de elección de los hablantes está explicado al inicio del programa
   12. `codigo/python/comprobaciones_ecapa.py` — comprobaciones adicionales de ECAPA-TDNN: número de grabaciones, intervalos de confianza por hablante y controles con identidades permutadas (apartado 4.5.2); parte de los embeddings guardados
   13. `codigo/python/figura_ecapa_arquitectura.py` — diagrama simplificado de la arquitectura de ECAPA-TDNN con la configuración del modelo preentrenado (apartado 3.3.3); es un esquema, no ejecuta el modelo

`codigo/python/segmentar_webmaus.py` documenta un intento de segmentación con el
servicio WebMAUS (el mismo método del estudio de referencia), que falló por un error
del servidor en el 100 % de las peticiones. Se conserva por transparencia
metodológica, pero no forma parte del pipeline final; su sustituto es
`extraer_formantes.py`.

### Otros scripts de análisis y generación de figuras

Los siguientes programas son auxiliares: producen análisis o figuras que aparecen en la memoria
y **no son pasos obligatorios** del pipeline anterior.

| Programa | Descripción |
|---|---|
| `codigo/python/desglose_canal.py` | Diagnóstico del efecto del canal sobre los formantes: diferencia media entre dos sesiones (mismo hablante y hablantes distintos) y desplazamiento de F2 y F3 por vocal (apartado 4.4). |
| `codigo/python/f0_sexo.py` | Efecto de añadir F0 al sistema, comparación entre hablantes del mismo sexo y clasificación del sexo (apartado 4.5.3). |
| `codigo/python/figura_bayes.py` | Esquema de la relación entre probabilidades a priori, razón de verosimilitud y probabilidades a posteriori (apartado 3.2). |
| `codigo/python/figura_cronograma.py` | Cronograma del trabajo por fases (apartado 4.1). |
| `codigo/python/figura_diseno.py` | Esquema del diseño experimental, de los datos a las métricas (apartado 4.2). |
| `codigo/python/figura_espectro_vocal.py` | Espectro de una vocal /a/ con su envolvente y los cuatro primeros formantes (apartado 3.1). |
| `codigo/python/figura_formantes_banda.py` | Distribución de F1–F4 frente al límite de la banda telefónica y comparación del F4 en limpio y telefónico (apartado 3.4). |
| `codigo/python/figura_intro.py` | La misma frase en limpio y telefónico simulado, con los formantes superpuestos (Introducción). |
| `codigo/python/figura_resultados.py` | Cllr y Cllr_min por condición y curvas de Tippett (apartado 4.4). |

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
