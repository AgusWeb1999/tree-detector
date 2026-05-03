# Young Tree Detector

Herramientas para detectar y contar árboles jóvenes en mosaicos aéreos GeoTIFF, con salida lista para QGIS.

El proyecto incluye dos enfoques:

- `young_tree_counter.py`: detector clásico sin machine learning, basado en vegetación RGB, candidatos por copa y filtros de distancia/fila.
- `ml_tree_detector.py`: detector híbrido entrenable. Usa puntos de referencia en `.gpkg` para entrenar un modelo supervisado que clasifica candidatos de copa.
- `make_final_gpkg_from_reference.py`: utilidad para recortar puntos perfectos de referencia al área del GeoTIFF actual y exportar un `.gpkg` final.

## Instalación

```bash
pip install -r requirements.txt
```

## Entrenar Modelo ML

```bash
python3 ml_tree_detector.py train \
  --image "/ruta/imagen.tiff" \
  --reference-points "/ruta/puntos_referencia.gpkg" \
  --out-dir "/ruta/modelo_arboles_v1" \
  --rgb-bands 1 2 3 \
  --spacing-m 1.5
```

## Predecir En Un Campo

```bash
python3 ml_tree_detector.py predict \
  --image "/ruta/nuevo_campo.tiff" \
  --model "models/young_tree_model_v3.joblib" \
  --out-dir "/ruta/salida_prediccion" \
  --vegetation-percentile 62 \
  --min-score 0.08 \
  --prob-threshold 0.55
```

La salida principal es:

```text
arboles_detectados.gpkg
```

También se genera:

```text
arboles_detectados.qml
detections_preview.png
arboles_detectados.csv
```

## Flujo De Mejora

1. Ejecutar predicción sobre un nuevo GeoTIFF.
2. Abrir `arboles_detectados.gpkg` en QGIS.
3. Corregir falsos positivos y faltantes.
4. Guardar los puntos corregidos como nuevo `.gpkg`.
5. Reentrenar el modelo con esos puntos para crear una versión mejor.

## Notas

- No subir GeoTIFFs, `.gpkg` productivos ni salidas pesadas al repo. El modelo inicial se incluye para facilitar uso y pruebas.
- El modelo mejora mucho si los campos nuevos tienen resolución, altura de vuelo, época y tamaño de copa similares a los ejemplos de entrenamiento.
- Si hay bosque, pasto o maleza parecida a las copas, conviene usar una AOI de plantación o sumar ejemplos corregidos de esas zonas.

## Documentación

Ver [docs/USAGE.md](docs/USAGE.md) para el flujo completo de uso, entrenamiento y mejora iterativa.
