# Integración de COES con el nowcast trimestral de PBI: contrato y secuencia

## Decisión actual

COES no entra todavía al producto trimestral ni a `Adaptive-IC`. El screen
exploratorio disponible al 18 de agosto de 2026 no encuentra una mejora robusta
frente al `Bridge(leaders)` vigente. La fuente se conserva porque aporta una
señal diaria pública y puede ganar valor cuando se acumulen vintages prospectivos.

## Arquitectura recomendada

No introducir como regresor un punto estimado por el MVP semanal. Eso crearía un
regresor generado, escondería su incertidumbre y permitiría doble conteo con el
PBI mensual y los indicadores domésticos ya usados por el modelo trimestral.

La ruta correcta tiene dos capas separadas:

1. **Bloque mensual completado**: `weekly_activity.quarterly` convierte la
   energía diaria COES en `g_coes_yoy`, crecimiento logarítmico anual de meses
   calendario completos. Es un candidato transparente para los modelos
   trimestrales directos.
2. **Señal parcial semanal del mes corriente**: el MVP semanal sigue siendo un
   producto de investigación independiente. Solo puede llegar al nowcast
   trimestral como un miembro separado de la combinación, con su propio
   backtest por origen, incertidumbre y disponibilidad histórica. No debe
   entrar como una columna de puntos estimados en `Bridge(leaders)`.

No hace falta construir un satélite semanal-mensual antes de probar el bloque
mensual completado. Sí será necesario para explotar formalmente la información
intramensual, pues el motor genérico actual opera sobre paneles M/Q y no conoce
la fracción de mes observada cada domingo.

## Contrato de datos ya preparado

Entrada no versionada:

```text
input/weekly_activity/processed/coes_daily.parquet
date, coes_energy_mwh, core_complete, ...
```

Adaptador rastreado:

```python
from weekly_activity.quarterly import (
    load_completed_monthly_coes_block, coes_monthly_metadata,
)

extra, audit = load_completed_monthly_coes_block()
monthly, quarterly, panel = peru_gdp.load_panel(
    extra=extra,
    extra_meta=[coes_monthly_metadata(delay_days=3)],
)
```

Salida:

```text
g_coes_yoy
```

Es la diferencia anual de logaritmos de la suma diaria de energía de COES.
Solo se genera con 13 meses calendario completos. Un mes incompleto, incluidos
los meses corrientes, es `NaN`, nunca se rellena. El adaptador devuelve además
un `audit` con días observados, días de calendario y bandera de completitud.

`delay_days=3` es una hipótesis de escenario posterior al fin de mes, no un
calendario histórico de publicación. Antes de promocionar el bloque se deben
repetir todos los resultados con 1, 3 y 7 días y registrar los timestamps
prospectivos reales en el manifiesto COES.

## Screen reproducible ya preparado

```bash
PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m weekly_activity.quarterly_screen --delay-days 3
```

El comando no modifica `pipeline/config/metadata.py`, `targets/peru_gdp.py` ni
artefactos publicados. Escribe resultados aislados en
`output/weekly_activity/quarterly_coes_screen/` y compara todos los modelos en
las mismas celdas `(trimestre, días hasta publicación)`.

Candidatos predefinidos:

- `Bridge(leaders)` y `P-MIDAS(leaders)`: referencias vigentes.
- `Bridge(leaders+COES)` y `P-MIDAS(leaders+COES)`: prueba incremental.
- `Bridge(COES)`: prueba diagnóstica aislada, nunca candidato de producción.
- `RW`: referencia de fallback.

La tabla reporta `rw_identical_share`: una participación alta revela que el
modelo está devolviendo el fallback y evita leer esos números como evidencia de
un modelo estimado.

## Puerta de promoción para Claude

1. Ejecutar el screen para demoras 1, 3 y 7, sin cambiar modelos ni la muestra.
2. Repetir con un backtest de origen exacto que use snapshots semanales
   prospectivos de COES. Mientras no existan, etiquetar todos los resultados
   como `pseudo_real_time_final_vintage_observation_clock`.
3. Exigir mejora en RMSE y MAE de `Bridge(leaders+COES)` respecto de
   `Bridge(leaders)` sobre celdas comunes, tanto en 2022+ como en la muestra
   ex-COVID, y también a 30 días o menos de la publicación trimestral.
4. Verificar que el candidato no sea principalmente idéntico a RW, que no
   degrade las bandas y que siga disponible en una corrida live exacta.
5. Solo si pasa esas condiciones, añadirlo como candidato, pero no como miembro
   de `ADAPTIVE_MEMBERS` todavía. Congelar una nueva carrera, revisar pesos y
   recalibrar bandas antes de cualquier promoción.
6. Para la señal parcial semanal, crear un nuevo miembro con contrato propio:
   corte semanal, mes objetivo, nowcast mensual, distribución y snapshot de
   features. Correr un puente trimestral separado y combinar distribuciones,
   no puntos. Esa es una extensión posterior, no un atajo.

## Riesgos que no se deben relajar

- COES no publica una historia de primera disponibilidad, por lo que no se
  puede afirmar un backtest realmente en tiempo real todavía.
- El nivel se limita al núcleo Norte-Centro-Sur. Los días parciales de
  Electroandes siguen siendo diagnóstico y nunca imputación.
- El PBI mensual, su ajuste estacional y el PBI trimestral siguen siendo
  final-vintage en los ejercicios históricos.
- COVID no se debe usar para sostener una relación predictiva. La correlación
  contemporánea de 2022-2026 entre crecimiento anual de electricidad y PBI fue
  aproximadamente cero en la inspección descriptiva.
