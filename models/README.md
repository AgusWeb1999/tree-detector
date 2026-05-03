# Modelos

Este directorio contiene el modelo inicial entrenado durante el desarrollo:

```text
young_tree_model_v3.joblib
```

Metadatos:

```text
model_meta_v3.json
training_report_v3.json
```

El modelo fue entrenado con puntos de referencia de árboles jóvenes sobre un mosaico RGB georreferenciado. Está pensado como punto de partida, no como modelo universal.

Para usarlo:

```bash
python3 ml_tree_detector.py predict \
  --image "/ruta/nuevo_campo.tiff" \
  --model "models/young_tree_model_v3.joblib" \
  --out-dir "/ruta/salida"
```

Para mejores resultados, reentrenar o ajustar con puntos corregidos del campo objetivo.
