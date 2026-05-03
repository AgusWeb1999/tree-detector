# Uso Y Entrenamiento

Este proyecto permite dos flujos:

1. Usar un modelo ya entrenado para generar puntos de árboles en QGIS.
2. Entrenar o mejorar el modelo con nuevos puntos corregidos.

## Instalar

```bash
git clone https://github.com/AgusWeb1999/tree-detector.git
cd tree-detector
pip install -r requirements.txt
```

## Usar El Modelo Incluido

El repo incluye un modelo inicial:

```text
models/young_tree_model_v3.joblib
```

Para predecir en un GeoTIFF nuevo:

```bash
python3 ml_tree_detector.py predict \
  --image "/ruta/nuevo_campo.tiff" \
  --model "models/young_tree_model_v3.joblib" \
  --out-dir "/ruta/salida_prediccion" \
  --vegetation-percentile 62 \
  --min-score 0.08 \
  --prob-threshold 0.55 \
  --nms-spacing-factor 0.65
```

La salida principal es:

```text
salida_prediccion/arboles_detectados.gpkg
```

También se genera:

```text
salida_prediccion/arboles_detectados.qml
salida_prediccion/detections_preview.png
salida_prediccion/arboles_detectados.csv
```

En QGIS, abrir `arboles_detectados.gpkg`. Si no aparece amarillo, cargar `arboles_detectados.qml` como estilo de capa.

## Entrenar Un Modelo Nuevo

Se necesita:

- Un GeoTIFF georreferenciado.
- Un `.gpkg` con puntos correctos de árboles.
- La separación aproximada entre árboles en metros.

Ejemplo:

```bash
python3 ml_tree_detector.py train \
  --image "/ruta/campo.tiff" \
  --reference-points "/ruta/puntos_correctos.gpkg" \
  --out-dir "/ruta/modelo_arboles_v4" \
  --rgb-bands 1 2 3 \
  --spacing-m 1.5 \
  --vegetation-percentile 62 \
  --min-score 0.08 \
  --positive-distance-m 0.55 \
  --negative-distance-m 1.00 \
  --anchor-positive-fraction 0.05
```

El modelo entrenado queda en:

```text
modelo_arboles_v4/young_tree_model.joblib
```

## Mejorar Iterativamente

1. Ejecutar `predict`.
2. Abrir el `.gpkg` en QGIS.
3. Borrar falsos positivos.
4. Agregar árboles faltantes.
5. Guardar como `puntos_corregidos.gpkg`.
6. Reentrenar con ese `.gpkg`.

Ese ciclo produce modelos cada vez más adaptados a la finca, resolución, iluminación, época y tamaño de copa.

## Parámetros Útiles

- `--prob-threshold`: sube/baja exigencia del modelo.
  - Más alto: menos puntos, mayor precisión.
  - Más bajo: más puntos, mayor recall.
- `--vegetation-percentile`: controla qué tan verde debe ser un candidato inicial.
- `--min-score`: controla fuerza mínima del candidato visual.
- `--nms-spacing-factor`: controla separación mínima entre puntos finales.
- `--spacing-m`: distancia aproximada entre árboles.

## Datos Pesados

No se recomienda versionar GeoTIFFs ni `.gpkg` productivos en el repo. Para compartir datasets reales, usar almacenamiento externo o GitHub Releases privados.
