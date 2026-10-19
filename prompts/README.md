# Registro de uso de inteligencia artificial generativa — TFM

Durante el desarrollo de este TFM he utilizado herramientas de inteligencia artificial, principalmente Claude (Anthropic) y ChatGPT (OpenAI), como apoyo en la programación, la resolución de errores, la comprensión de conceptos técnicos y la revisión de textos, incluyendo correcciones ortográficas y mejoras en la claridad de algunas explicaciones.

A continuación, se resumen por meses los principales usos de estas herramientas, indicando para qué se utilizaron y cómo se revisaron, adaptaron y comprobaron los resultados obtenidos.

---

## Agosto 2026

**Planificación y estructura del TFM.** Orientación sobre cómo abordar el trabajo:
estructura de la memoria según la plantilla de la VIU, plan de trabajo por fases,
lectura de la guía docente y del reglamento del TFM. Se generó un esqueleto de la
memoria en `.docx` con notas de contenido por sección, y un panel de seguimiento del
plan de trabajo.
Qué hice: revisé y adapté la estructura propuesta a mi caso; las decisiones de alcance
y el plan final son mías.

**Definición del tema.** A partir de la conversación con mi tutora, ayuda para entender
el planteamiento propuesto (comparación forense de voz, marco de razón de
verosimilitud) y para definir un alcance propio: el efecto de la degradación del canal
telefónico sobre la fuerza de la evidencia, tomando como referencia el paper de mi
tutora (López-Escobedo, Huerta-Pacheco y Meza-Ruiz, 2025).
Qué hice: el tema, el título, los objetivos y el alcance son decisión mía, orientada
por la propuesta de mi tutora.

**Lista de lecturas.** Búsqueda y organización de bibliografía relevante en español e
inglés sobre comparación forense de voz, agrupada por niveles de prioridad.
Qué hice: leí las fuentes y decidí cuáles añadir a Zotero.

---

## Septiembre 2026

**Entorno de trabajo.** Ayuda para instalar y configurar Python (entorno virtual), R
(paquete fvclrr), Git/GitHub, ffmpeg y Zotero, y para estructurar las carpetas del
proyecto.
Qué hice: ejecuté yo misma cada instalación y cada paso.

**Gestión de referencias en Zotero.** Ayuda para añadir referencias por DOI/identificador,
resolver fichas mal importadas y, más adelante, auditar la biblioteca frente a las citas
usadas en la memoria (script `leer_zotero.py`, que vuelca la colección para comprobar
qué falta o está mal).
Qué hice: decidí qué referencias eran relevantes y corregí las fichas señaladas como
incorrectas.

**`inventario_corpus.py`.** Script para inventariar los corpus OpenSLR (SLR72 Colombia,
SLR73 Perú): listar hablantes, sexo y número de grabaciones por hablante a partir de
los nombres de archivo y los `line_index`.
Qué hice: lo ejecuté y revisé el resumen (71 hablantes disponibles).

**`seleccionar_hablantes.py`.** Script para elegir 24 hablantes equilibrados por sexo
(12+12) y dividir cada uno en dos pseudosesiones (A/B), con semilla fija para que la
selección sea reproducible.
Qué hice: revisé el código, añadí comentarios propios con las decisiones
metodológicas, lo ejecuté y comprobé la salida (1920 grabaciones seleccionadas).

**`procesar_audios.py`.** Conversión de los 1920 audios seleccionados a formato
uniforme (mono, 16 kHz, 16 bits) con ffmpeg.
Qué hice: revisado y ejecutado por mí (1920/1920 sin errores).

**`degradar_telefono.py`.** Generación de la condición telefónica simulada (paso banda
300-3400 Hz + remuestreo a 8 kHz) con ffmpeg.
Qué hice: revisado y ejecutado por mí (1920/1920 sin errores).

**`segmentar_webmaus.py`.** Intento de segmentación fonética con el servicio WebMAUS,
el mismo método que el paper de referencia. El servicio devolvió un error de ejecución
en el 100 % de las peticiones (0/1920), por causas del servidor ajenas a este trabajo.
Qué hice: con esa información, decidí pasar a un método de segmentación local.

**`extraer_formantes.py`.** Sustituto de WebMAUS: detección local de núcleos vocálicos
(picos de intensidad sonoros) y extracción de F1-F4/F0 con Parselmouth, en condición
limpia y telefónica, con clasificación de la vocal por F1-F2. Versión reanudable
(escritura fila a fila) tras una primera interrupción.
Qué hice: revisé el método propuesto, lo ejecuté y comprobé los resultados (14.662
tokens de vocal, 0 errores).

**`comparar_locutores.py`.** Sistema de razón de verosimilitud mediante regresión
logística sobre diferencias de formantes entre pares de grabaciones, validación
leave-one-speaker-out, cálculo de Cllr/Cllr_min/EER y gráficos de Tippett. Sustituye al
método MVKD/fvclrr del paper de referencia por restricción de tiempo (decisión
documentada como comentario en el propio script y como limitación en la memoria).
Qué hice: detecté y corregí, con ayuda de la IA, un fallo en el cálculo de Cllr_min (la
recalibración isotónica no ponderaba las clases, dando un resultado imposible:
Cllr_min > Cllr). Verifiqué que Cllr_min ≤ Cllr en las cuatro condiciones antes de dar
los resultados por buenos.

**`graficos.py` y `desglose.py`.** Visualizaciones adicionales (espacio vocálico F1-F2,
ejemplo visual mismo/distinto locutor, resumen de métricas) y desglose de resultados
por vocal y por sexo (comparando todos los pares frente a solo pares del mismo sexo).
Qué hice: revisé los resultados y decidí qué interpretación dar a los números para la
memoria.

**Redacción de la memoria (capítulos 1-6, Resumen, Abstract).** Redacción de un primer
borrador de la Introducción, Objetivos, Fundamentos y trabajos relacionados, Desarrollo
del proyecto y resultados (con tablas y figuras incrustadas), Conclusión y trabajos
futuros, y Referencias, a partir de las lecturas, las decisiones metodológicas y los
resultados obtenidos durante el proyecto. Ampliación posterior de los capítulos 3 y 4
para acercar la extensión al mínimo exigido.
Qué hice: pendiente de una revisión completa por mi parte antes del depósito. Comprobé
que el documento no quedara roto tras cada cambio (estructura, tablas, figuras,
índices). El planteamiento, las decisiones metodológicas, el análisis crítico y las
conclusiones son míos.

---

## Octubre 2026

**Revisión de mi redacción (Resumen, Abstract, Introducción).** Pasé a la IA el texto
que estoy redactando yo a mano en el documento con la plantilla oficial para que lo
revisara. Señaló imprecisiones técnicas (terminología de fonética acústica, un párrafo
poco preciso sobre el desafío de la discordancia de canal), incoherencias entre el
resumen, el título y el trabajo realmente hecho, y la falta de una cita para los rangos
de F0. A petición mía, aplicó directamente en el documento las correcciones
acordadas (terminología, tiempos verbales, redacción de las grabaciones propias como
fase en curso) y sustituyó los rangos de F0 sin fuente por una formulación general,
dejando un comentario en el documento para añadirlos cuando verifique la referencia.
Qué hice: decidí qué correcciones aceptar, corregí el título y mantuve la estructura y
las ideas de mi texto. Pendiente: verificar la cita de F0 y revisar el resto de capítulos.

**`figura_intro.py`.** Script que genera la figura de la Introducción: espectrograma de
una misma grabación del corpus en condición limpia y telefónica simulada, con los
formantes superpuestos y el límite de 3400 Hz. La IA escribió el código y, a partir de
una primera versión, corrigió dos fallos (formantes dibujados en silencios y un F4
espurio en el panel telefónico). También insertó la figura, su título y su referencia
en el texto del documento, y los índices de ilustraciones y de tablas, a petición mía.
Qué hice: elegí qué mostrar, revisé la figura y decidí indicar en su pie que se ha
elaborado con apoyo de la IA.

**Revisión del Capítulo 3 y sus figuras y tabla.** Con la IA revisé, apartado por
apartado, los textos del Capítulo 3 (fundamentos y trabajos relacionados) que redacté a
mano: señaló errores de terminología y de contenido (por ejemplo, una afirmación sobre
Morrison (2011a) que no era correcta y una novedad del trabajo formulada de forma
excesiva), comprobó citas y referencias en fuentes (Crossref, páginas oficiales de la
ITU y de la UE) y añadió las referencias que faltaban a mi biblioteca de Zotero a través
de su conexión local. Para ilustrar el capítulo generó, con mis datos, los scripts
`figura_formantes_banda.py` (distribución de F1-F4 frente a la banda telefónica),
`figura_espectro_vocal.py` (espectro de una vocal con sus formantes) y `figura_bayes.py`
(esquema de la razón de verosimilitud), y una tabla comparativa de métodos; los insertó
en el documento con título y referencia cruzada, a petición mía.
Qué hice: decidí qué figuras y tabla incluir, qué afirmaciones mantener o suavizar
(incluida la reducción de menciones al trabajo de mi tutora), revisé cada figura y
mantuve mi redacción como base de todos los textos. Pendiente: leer los artículos que
cito (Zhang et al., 2013, en particular) y completar los apartados que faltan.

**Cronograma y esquema del diseño (apartados 4.1 y 4.2).** La IA generó los scripts
`figura_cronograma.py` (cronograma por fases con el estado real de cada una) y
`figura_diseno.py` (esquema ilustrado del diseño experimental, de los datos a las
métricas), una tabla con las cuatro condiciones de comparación, y los insertó en el
documento con título y referencia cruzada, ajustando su tamaño y paginación. A petición
mía, el cronograma refleja lo que realmente hice en cada mes, incluidos los intentos
fallidos y el trabajo previo a la solicitud formal.
Qué hice: decidí qué mostrar y cómo (colores, cajas, explicación de cada una),
corregí las fechas y estados del cronograma, y revisé cada figura hasta darla por buena.

**Revisión de los apartados 4.2 y 4.3 y del código.** Redacté yo los apartados 4.2
(planteamiento) y 4.3 (desarrollo) y pedí a la IA que los comprobara contra el código y
los datos. Detectó erratas de terminología (razón de probabilidad en vez de verosimilitud,
tracto vocal, género/sexo), numeración de tablas y cifras o descripciones que no
coincidían con lo que hacen los scripts (por ejemplo, el punto de medida de los
formantes o el límite de 3900 Hz en la condición telefónica). Al comparar el texto con
`comparar_locutores.py` y `desglose.py`, la IA encontró también un error en el código:
se corregía dos veces el desequilibrio entre clases (ponderación de clases y resta del
logaritmo de las probabilidades previas), lo que inflaba el Cllr. Verifiqué que, sin la
resta, el Cllr queda cerca del Cllr_min (calibración correcta), que Cllr_min y EER no
cambian y que el orden entre condiciones se mantiene; regeneré resultados y gráficos.
Además, al analizar por qué faltaba la vocal /i/ en una sesión de un hablante, la IA
midió que la región de /o/ absorbía muestras de /u/ (casi un tercio de las /o/). Decidí
incluir la /u/ en todo el trabajo: la IA modificó las regiones F1-F2 en
`extraer_formantes.py` (frontera /o/-/u/ en F1 = 450 Hz, decisión que declaro como
limitación), repitió la extracción y todos los resultados, y actualizó las cifras del
Resumen, el Abstract, los Objetivos, el 4.2 y el 4.3.
Al preparar los resultados (4.4), la IA comparó los formantes de las versiones limpia y
telefónica de los mismos núcleos vocálicos y vio que, con 5 formantes buscados en la señal
telefónica, F2 y F3 se desplazaban unas 260 y 830 Hz (más que la variación entre hablantes),
señal de que el algoritmo asignaba mal los formantes; con 4 formantes el desplazamiento
casi desaparece. La IA modificó `extraer_formantes.py` (4 formantes en el telefónico,
techo de 3900 Hz), repitió la extracción y todos los resultados y figuras, y actualizó las
cifras del Resumen, el Abstract, el 4.3 y la Ilustración 4.
Qué hice: decidí qué correcciones aceptar, mantuve mi redacción como base, decidí
incluir la /u/ y usar 4 formantes en el telefónico, y valoré los nuevos resultados antes
de aceptarlos.

**Apartado 4.4 (Resultados) y diagnóstico del canal.** A petición mía, la IA redactó un
borrador del apartado 4.4 a partir de los resultados de `metricas.csv` y de los desgloses
por vocal y por sexo, con un lenguaje sencillo, y lo insertó en el documento con cinco
tablas y dos figuras (`figura_resultados.py`: barras de Cllr y Cllr_min y curvas de
Tippett). Para explicar por qué la condición cruzada casi anula la evidencia, la IA
escribió `desglose_canal.py`, que compara la diferencia entre dos sesiones del mismo
hablante y de hablantes distintos según el canal, y el desplazamiento de F2 y F3 por vocal;
también probó, sin adoptarlo, promediar por mediana en vez de por media, que no mejoró el
resultado.
Qué hice: pedí el texto, decidí qué apartados mantener (por ejemplo, el desglose por sexo
breve, a ampliar en 4.5.4) y qué tono usar. Pendiente: revisar y ajustar la redacción.

**F0 y clasificación del sexo (apartado 4.5.4).** A petición mía, la IA escribió `f0_sexo.py`:
añade la diferencia de F0 como rasgo del sistema de comparación (con todos los pares y solo con
parejas del mismo sexo) y clasifica el sexo de cada sesión con validación leave-one-speaker-out.
También redactó un borrador del apartado 4.5.4 con las dos tablas de resultados y la advertencia de
que F0 varía más entre días que los formantes.
Qué hice: pedí ampliar el trabajo con este análisis, revisé los resultados y decidí cómo presentarlos.
Pendiente: revisar y reescribir el texto con mis palabras.

**MVKD con el paquete fvclrr (apartado 4.5.2).** A petición mía, la IA escribió `codigo/r/mvkd_formantes.R`
(MVKD por vocal con el paquete de R fvclrr, en las cuatro condiciones; hubo que usar la función MVKD
porque MVKD_loop no era compatible con la versión actual de tidyr) y `codigo/python/mvkd_evaluacion.py`
(fusión de las cinco vocales con regresión logística y leave-one-speaker-out, y comparación con la
regresión logística sobre los mismos 22 hablantes). También redactó un borrador del apartado 4.5.2.
Qué hice: pedí la comparación con el MVKD y valoré los resultados antes de incluirlos.
Pendiente: revisar y reescribir el texto con mis palabras.

