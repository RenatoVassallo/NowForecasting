# Diseño del MVP: nowcast semanal de actividad y PBI mensual de Perú

## Qué mide

El producto es una estimación del crecimiento mensual desestacionalizado del
PBI peruano que se actualiza al cierre de cada domingo. Es un nowcast para el
mes calendario actual. No hay un PBI semanal observado, por lo que no se usa
ese nombre ni se intenta crear una serie semanal artificial.

## Fuentes y factibilidad comprobada el 18 de agosto de 2026

| Bloque | Fuente efectiva | Cobertura comprobada | Riesgo principal | Decisión MVP |
|---|---|---:|---|---|
| Electricidad | COES, núcleo Norte-Centro-Sur de demanda ejecutada, 48 registros de media hora | consultas válidas en 2015-01-02, 2019-01-02 y agosto de 2026 | COES no expone timestamps históricos de primera publicación; Electroandes es incompleta intermitentemente | Incluir |
| Pagos | BCRP, operaciones de clientes LBTR | 2019-01-02 a 2026-08-14 en las cuatro series elegidas | pagos financieros no equivalen a consumo; API puede mostrar challenge anti-bot | Incluir |
| Búsquedas | Google Trends, Perú | no hay acceso público estable sin autorización a la API alfa | índice relativo, sensible a ventana y redescarga | Opcional, desactivado |
| Target | nivel PBI mensual BCRP/INEI, SA con X13 local | cache local hasta el último PBI publicado | revisiones y SA de muestra completa | Incluir con etiqueta pseudo-tiempo-real |

La documentación de COES confirma que la demanda histórica diaria tiene 48
intervalos de media hora y que su endpoint agregado entrega demanda ejecutada
por área. La documentación del BCRP identifica `PD38073DD`, `PD38075DD`,
`PD38077DD` y `PD38079DD` como las cuatro series de clientes seleccionadas.

## Reglas de información

- Las semanas cierran domingo. Se producen filas para cada domingo de un mes y
  una fila adicional al fin de mes.
- Cada feature se filtra con `observation_date <= cutoff_date` antes de sumar.
- Electricidad: `sum(MW media-hora) * 0.5` para Norte, Centro y Sur, sin
  rellenar días parciales. Electroandes no se usa en el nivel porque presenta
  huecos, pero su disponibilidad se registra como diagnóstico.
- LBTR: valor y número de operaciones siguen separados por moneda. Se usa
  `log1p` del promedio diario acumulado del mes, sin interpretarlo como gasto.
- Trends: solo semanas completas dentro del mes, sin interpolación. La extracción
  debe registrar tópico, geografía, ventana y timestamp de descarga.
- Etiquetas de PBI: se admite entrenamiento únicamente cuando `fin de mes + 51
  días <= fecha de corte`. Ese rezago es una hipótesis explícita del proyecto,
  no un calendario INEI observado.

## Riesgos que no deben olvidarse

1. La evaluación usa PBI y ajuste estacional final-vintage. No es una prueba de
   tiempo real genuina.
2. COES y BCRP no ofrecen una historia pública de timestamps de publicación.
   Para el backtest se usa el reloj de fecha de observación y se declara así.
3. La muestra LBTR empieza en 2019. No puede sustentar una conclusión sólida
   sobre el período pre-2020 ni justificar tuning complejo.
4. La composición de las áreas COES puede cambiar. Un cambio debe aparecer en
   `area_signature` y disparar revisión antes de mezclar niveles. La señal MVP
   se limita explícitamente al núcleo Norte-Centro-Sur.
5. Un intervalo empírico solo se publica cuando hay al menos 12 errores previos
   de la misma etapa y modelo. No se finge precisión con muestras pequeñas.

## Criterio para una V2

Proceder solo si Ridge multibloque supera de manera consistente al AR y a Ridge
electricidad en la muestra temporal de prueba y en las tres etapas de corte,
sin empeorar claramente 2020-2021. Si no ocurre, no añadir modelos complejos.
La siguiente fuente pública a evaluar sería un indicador diario de movilidad o
transacciones con timestamps históricos verificables, no más términos de
Google Trends.
