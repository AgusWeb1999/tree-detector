#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import fiona
import rasterio
from shapely.geometry import Point, mapping, shape


REFERENCE_GPKG = Path("/Users/agusmazzini/Downloads/Proyecto conteo/Cañada Brava - Puntos.gpkg")
INPUT_TIF = Path("/Users/agusmazzini/Downloads/CANIADA BRAVA-MOSAICO-240326-PRENDIMIENTO-PARTE1DE2.tiff")
OUTPUT_DIR = Path("/Users/agusmazzini/Downloads/Proyecto conteo")
OUTPUT_GPKG = OUTPUT_DIR / "Cañada Brava - arboles jovenes PARTE1.gpkg"
OUTPUT_QML = OUTPUT_DIR / "Cañada Brava - arboles jovenes PARTE1.qml"
OUTPUT_SUMMARY = OUTPUT_DIR / "Cañada Brava - arboles jovenes PARTE1 resumen.json"


QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <renderer-v2 type="singleSymbol" enableorderby="0" referencescale="-1" forceraster="0" symbollevels="0">
    <symbols>
      <symbol type="marker" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer enabled="1" pass="0" class="SimpleMarker" locked="0">
          <Option type="Map">
            <Option name="angle" type="QString" value="0"/>
            <Option name="color" type="QString" value="255,230,0,255"/>
            <Option name="horizontal_anchor_point" type="QString" value="1"/>
            <Option name="joinstyle" type="QString" value="bevel"/>
            <Option name="name" type="QString" value="circle"/>
            <Option name="offset" type="QString" value="0,0"/>
            <Option name="offset_map_unit_scale" type="QString" value="3x:0,0,0,0,0,0"/>
            <Option name="offset_unit" type="QString" value="MM"/>
            <Option name="outline_color" type="QString" value="0,0,0,255"/>
            <Option name="outline_style" type="QString" value="solid"/>
            <Option name="outline_width" type="QString" value="0.25"/>
            <Option name="outline_width_map_unit_scale" type="QString" value="3x:0,0,0,0,0,0"/>
            <Option name="outline_width_unit" type="QString" value="MM"/>
            <Option name="scale_method" type="QString" value="diameter"/>
            <Option name="size" type="QString" value="1.8"/>
            <Option name="size_map_unit_scale" type="QString" value="3x:0,0,0,0,0,0"/>
            <Option name="size_unit" type="QString" value="MM"/>
            <Option name="vertical_anchor_point" type="QString" value="1"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    layers = fiona.listlayers(REFERENCE_GPKG)
    if not layers:
        raise RuntimeError(f"No layers found in {REFERENCE_GPKG}")
    layer = layers[0]

    with rasterio.open(INPUT_TIF) as tif:
        bounds = tif.bounds
        crs = tif.crs
        transform = tif.transform
        width = tif.width
        height = tif.height

    schema = {
        "geometry": "Point",
        "properties": {
            "id": "int",
            "x_pixel": "float",
            "y_pixel": "float",
            "fuente": "str:32",
        },
    }

    total = 0
    inside = 0
    with fiona.open(REFERENCE_GPKG, layer=layer) as src:
        if src.crs and crs and src.crs.to_string() != crs.to_string():
            raise RuntimeError(f"CRS mismatch: points={src.crs}, tif={crs}")
        with fiona.open(
            OUTPUT_GPKG,
            mode="w",
            driver="GPKG",
            layer="arboles_jovenes",
            schema=schema,
            crs=crs,
        ) as dst:
            inv = ~transform
            for feat in src:
                total += 1
                geom = shape(feat.geometry)
                if not isinstance(geom, Point):
                    continue
                x, y = geom.x, geom.y
                if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
                    continue
                px, py = inv * (x, y)
                if not (0 <= px < width and 0 <= py < height):
                    continue
                inside += 1
                dst.write({
                    "geometry": mapping(geom),
                    "properties": {
                        "id": inside,
                        "x_pixel": float(px),
                        "y_pixel": float(py),
                        "fuente": "referencia",
                    },
                })

    OUTPUT_QML.write_text(QML, encoding="utf-8")
    summary = {
        "reference_gpkg": str(REFERENCE_GPKG),
        "input_tif": str(INPUT_TIF),
        "output_gpkg": str(OUTPUT_GPKG),
        "output_qml": str(OUTPUT_QML),
        "reference_points_total": total,
        "points_inside_tif": inside,
        "crs": crs.to_string() if crs else None,
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
