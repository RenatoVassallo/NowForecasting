# MVP: nowcast semanal del crecimiento mensual del PBI peruano

Este producto estima, cada domingo, el crecimiento mensual desestacionalizado
del PBI del mes en curso. No estima ni publica un "PBI semanal": no existe una
variable semanal observada que permita validar ese objeto.

## Alcance inicial

Solo utiliza tres bloques públicos:

1. Demanda eléctrica ejecutada diaria de COES.
2. Operaciones diarias de clientes en el LBTR del BCRP, separando valor y
   número de operaciones en moneda nacional y extranjera.
3. Un panel semanal congelado de Google Trends, únicamente si se cuenta con
   acceso autorizado y una extracción con metadatos. La API oficial está en
   alfa y es de acceso controlado, por lo que el MVP no usa scraping ni
   `pytrends`.

El modelo multibloque opera con COES y LBTR. Añade Trends solamente cuando el
archivo congelado satisface el contrato de procedencia. No se reemplazan los
datos faltantes ni se convierten series mensuales en semanales.

## Reproducción

Desde la raíz del repositorio, con el entorno configurado:

```bash
PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m weekly_activity.run download \
  --start-coes 2015-01-01 --start-lbtr 2019-01-02 --end 2026-08-17 --workers 4
PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m weekly_activity.run build
PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m weekly_activity.run evaluate
PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m weekly_activity.run nowcast --as-of 2026-08-18
PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m weekly_activity.diagnostics
```

El primer comando de COES consulta una fecha por solicitud porque esa es la
granularidad documentada del endpoint. La descarga es reanudable: cada respuesta
cruda se guarda bajo `input/weekly_activity/raw/coes/daily/`, con un manifiesto
que registra URL, hash y fecha de descarga. No borre esa carpeta si quiere una
traza reproducible. La señal de electricidad es el núcleo estable Norte, Centro
y Sur reportado por COES. Electroandes se conserva como diagnóstico de calidad,
pero no se imputa ni se suma cuando tiene intervalos faltantes.

Rutas principales:

| Artefacto | Ruta |
|---|---|
| Respuestas crudas y manifiestos | `input/weekly_activity/raw/` |
| Datos diarios limpios | `input/weekly_activity/processed/coes_daily.parquet`, `lbtr_daily.parquet` |
| Target y features | `input/weekly_activity/processed/target_monthly.parquet`, `weekly_features.parquet` |
| Diccionario y cobertura | `feature_dictionary.csv`, `output/weekly_activity/source_coverage.csv` |
| Backtest y tabla de resultados | `output/weekly_activity/backtest.parquet`, `backtest_scoreboard.csv` |
| Producto vigente | `output/weekly_activity/latest_nowcast.csv` |

## Definición del target y limitaciones

El target es `100 * Δ log(PBI mensual SA)`, construido a partir de la columna
`pbim` de `input/peru/monthly_sa_spec3.parquet`. Esa cache aplica X13 a la
historia actualmente revisada. También se impone el rezago conservador de 51
días ya usado por el proyecto para decidir qué etiquetas están disponibles en
cada origen.

Esto permite un backtest con conjunto de información coherente en predictores,
pero todavía no un ejercicio genuinamente en tiempo real. Faltan los vintages de
PBI, los vintages del ajuste estacional y timestamps históricos de la primera
publicación de COES y LBTR. Cada tabla se etiqueta
`pseudo_real_time_final_vintage_observation_clock`.

Por tanto, una mejora frente a AR en esta versión es evidencia exploratoria, no
una razón suficiente para publicar o promocionar el índice. La condición para
V2 es capturar esos vintages hacia adelante y evaluar los nowcasts congelados.

## Modelos y evaluación

Los candidatos se fijan antes de mirar resultados:

1. AR(1) del PBI mensual, con pronóstico recursivo desde la última etiqueta
   disponible.
2. Ridge con electricidad COES y controles calendario.
3. Ridge con electricidad, LBTR y controles calendario. Google Trends se suma
   solo si su snapshot completo está disponible en entrenamiento y corte actual.

El escalamiento y la selección de `alpha` de Ridge se hacen dentro de cada
ventana de entrenamiento. El backtest es expanding-window y reporta semana 1,
semana 2 y fin de mes, RMSE, MAE y precisión direccional, además de períodos
pre-2020, 2020-2021 y post-2020. Las comparaciones comunes se calculan solo
para celdas donde todos los candidatos emiten un pronóstico.

Los orígenes evaluados empiezan en 2015, cuando inicia COES. El AR no recibe
una ventaja artificial por puntuar décadas sin ningún bloque de alta frecuencia.

Los pagos LBTR no se interpretan como consumo. Son un bloque de actividad y
liquidez que debe demostrar valor predictivo fuera de muestra.

`latest_nowcast.csv` conserva los tres candidatos. En el estado actual,
`Ridge electricity` es líder de investigación en la muestra test, pero el
archivo lo etiqueta `research_mvp_not_production`: no existe una promoción
automática a un producto publicado mientras falten vintages reales.

## Figuras descriptivas de electricidad y PBI

`python -m weekly_activity.diagnostics` guarda tres gráficos en
`output/weekly_activity/diagnostics/`:

1. Electricidad semanal en nivel y PBI mensual SA, ambos indexados a 2019.
   El PBI se dibuja solamente como puntos mensuales ubicados al día 15 y unidos
   entre observaciones. No se convierte en una serie semanal.
2. Variación anual de electricidad agregada genuinamente a mes y PBI mensual.
3. Scatterplots y correlaciones por régimen, para evitar que la pandemia cree
   una correlación de muestra completa engañosa.

Las figuras son diagnósticos descriptivos, no evidencia de causalidad ni un
criterio para elegir el modelo. El archivo CSV adjunto contiene los tamaños de
muestra y correlaciones que se muestran en el tercer gráfico.

## Preparación para el nowcast trimestral general

El adaptador `weekly_activity.quarterly` construye `g_coes_yoy` usando solo
meses COES completos. No incorpora todavía una estimación parcial semanal ni
modifica el producto trimestral. El screen aislado se ejecuta así:

```bash
PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m weekly_activity.quarterly_screen --delay-days 3
```

La secuencia de integración, las pruebas requeridas y los motivos para no usar
el punto estimado semanal como regresor generado están en
`docs/weekly_activity_quarterly_integration.md`.
