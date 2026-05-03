#!/usr/bin/env python3
"""
Trainable young-tree detector for georeferenced drone RGB GeoTIFFs.

This is a hybrid detector:
  1. Classical CV proposes green/crown-like candidates.
  2. A supervised sklearn model classifies candidate patches as tree/non-tree.
  3. Detections are merged and exported as a QGIS-ready GeoPackage.

It is designed for iterative improvement:
  - Add/correct points in QGIS.
  - Retrain with the updated GPKG.
  - Predict again on the same or a similar mosaic.

Dependencies:
  rasterio, fiona, shapely, numpy, scipy, scikit-image, scikit-learn, joblib, opencv-python
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import fiona
import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from shapely.geometry import Point, mapping, shape
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


EPS = 1e-6


QML_YELLOW_POINTS = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <renderer-v2 type="singleSymbol" enableorderby="0" referencescale="-1" forceraster="0" symbollevels="0">
    <symbols>
      <symbol type="marker" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer enabled="1" pass="0" class="SimpleMarker" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="255,230,0,255"/>
            <Option name="outline_color" type="QString" value="0,0,0,255"/>
            <Option name="outline_width" type="QString" value="0.25"/>
            <Option name="outline_width_unit" type="QString" value="MM"/>
            <Option name="name" type="QString" value="circle"/>
            <Option name="size" type="QString" value="1.8"/>
            <Option name="size_unit" type="QString" value="MM"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""


@dataclass
class ModelMeta:
    pixel_size_m: float
    spacing_m: float
    spacing_px: float
    crown_radius_px: float
    min_crown_radius_px: float
    max_crown_radius_px: float
    rgb_bands: tuple[int, int, int]
    feature_names: list[str]
    crs: str | None


@dataclass
class Candidate:
    x: float
    y: float
    score: float
    area_px: float
    radius_px: float


def robust_rescale(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    ok = np.isfinite(arr)
    if not np.any(ok):
        return np.zeros(arr.shape, dtype=np.float32)
    p1, p2 = np.nanpercentile(arr[ok], [low, high])
    if p2 <= p1:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - p1) / (p2 - p1), 0, 1).astype(np.float32)


def pixel_size(transform: Affine) -> float:
    return float((math.hypot(transform.a, transform.d) + math.hypot(transform.b, transform.e)) / 2.0)


def read_rgb(src: rasterio.io.DatasetReader, window: Window, rgb_bands: tuple[int, int, int]) -> np.ndarray:
    arr = src.read(list(rgb_bands), window=window, boundless=True, fill_value=0).astype(np.float32)
    return np.stack([robust_rescale(arr[i]) for i in range(3)], axis=-1)


def rgb_indices(rgb: np.ndarray) -> dict[str, np.ndarray]:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    exg = 2 * green - red - blue
    gli = (2 * green - red - blue) / (2 * green + red + blue + EPS)
    brightness = (red + green + blue) / 3.0
    saturation = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    green_dom = green - np.maximum(red, blue)
    veg = robust_rescale(0.46 * robust_rescale(exg) + 0.34 * robust_rescale(gli) + 0.14 * robust_rescale(green_dom) + 0.06 * robust_rescale(saturation))
    return {
        "red": red,
        "green": green,
        "blue": blue,
        "exg": robust_rescale(exg),
        "gli": robust_rescale(gli),
        "brightness": brightness,
        "saturation": saturation,
        "green_dom": robust_rescale(green_dom),
        "veg": veg,
    }


def load_reference_pixels(gpkg: Path, src: rasterio.io.DatasetReader) -> np.ndarray:
    layers = fiona.listlayers(gpkg)
    if not layers:
        raise ValueError(f"No layers found in {gpkg}")
    inv = ~src.transform
    pts = []
    with fiona.open(gpkg, layer=layers[0]) as ref:
        if ref.crs and src.crs and ref.crs.to_string() != src.crs.to_string():
            raise ValueError(f"CRS mismatch: reference={ref.crs}, raster={src.crs}")
        for feat in ref:
            geom = shape(feat.geometry)
            if not isinstance(geom, Point):
                continue
            px, py = inv * (geom.x, geom.y)
            if 0 <= px < src.width and 0 <= py < src.height:
                pts.append((px, py))
    if not pts:
        raise ValueError("No reference points fall inside the raster.")
    return np.array(pts, dtype=np.float32)


def estimate_spacing_from_points(points_px: np.ndarray, fallback_px: float) -> float:
    if len(points_px) < 8:
        return fallback_px
    tree = cKDTree(points_px)
    d, _ = tree.query(points_px, k=min(4, len(points_px)))
    nn = d[:, 1]
    nn = nn[np.isfinite(nn)]
    if len(nn) == 0:
        return fallback_px
    return float(np.nanmedian(nn))


def make_meta(src: rasterio.io.DatasetReader, ref_points_px: np.ndarray | None, spacing_m: float | None, rgb_bands: tuple[int, int, int]) -> ModelMeta:
    px_m = pixel_size(src.transform)
    fallback_spacing_px = (spacing_m or 1.5) / px_m
    spacing_px = estimate_spacing_from_points(ref_points_px, fallback_spacing_px) if ref_points_px is not None else fallback_spacing_px
    spacing_m_final = spacing_px * px_m
    crown = float(np.clip(spacing_px * 0.25, 4, 40))
    return ModelMeta(
        pixel_size_m=px_m,
        spacing_m=spacing_m_final,
        spacing_px=spacing_px,
        crown_radius_px=crown,
        min_crown_radius_px=max(2.0, spacing_px * 0.12),
        max_crown_radius_px=max(6.0, spacing_px * 0.45),
        rgb_bands=rgb_bands,
        feature_names=[],
        crs=src.crs.to_string() if src.crs else None,
    )


def candidate_mask(idx: dict[str, np.ndarray], meta: ModelMeta, vegetation_percentile: float) -> np.ndarray:
    veg = idx["veg"]
    bright = idx["brightness"]
    finite = veg[np.isfinite(veg)]
    otsu = threshold_otsu(finite) if finite.size > 64 else 0.5
    veg_thr = max(float(np.nanpercentile(veg, vegetation_percentile)), float(otsu * 0.92))
    bright_thr = min(float(np.nanpercentile(bright, 8)), float(np.nanmedian(bright) * 0.70))
    mask = (veg >= veg_thr) & (bright >= bright_thr)
    r = max(1, int(round(meta.min_crown_radius_px * 0.25)))
    min_obj = max(5, int(math.pi * (meta.min_crown_radius_px * 0.45) ** 2))
    mask = binary_opening(mask, disk(r))
    mask = binary_closing(mask, disk(r))
    mask = remove_small_objects(mask, min_size=min_obj)
    mask = remove_small_holes(mask, area_threshold=max(8, min_obj))
    return mask.astype(bool)


def response_from_indices(idx: dict[str, np.ndarray], mask: np.ndarray, meta: ModelMeta) -> np.ndarray:
    veg = idx["veg"]
    small = max(0.8, meta.crown_radius_px * 0.22)
    large = max(small + 0.5, meta.crown_radius_px * 1.1)
    contrast = ndi.gaussian_filter(veg, small) - ndi.gaussian_filter(veg, large)
    dist = ndi.distance_transform_edt(mask)
    resp = 0.72 * robust_rescale(contrast) + 0.28 * robust_rescale(dist)
    resp[~mask] = 0
    return robust_rescale(resp)


def propose_candidates_for_window(
    src: rasterio.io.DatasetReader,
    window: Window,
    meta: ModelMeta,
    vegetation_percentile: float,
    min_score: float,
    valid_box: tuple[int, int, int, int] | None = None,
) -> tuple[list[Candidate], dict[str, np.ndarray]]:
    rgb = read_rgb(src, window, meta.rgb_bands)
    idx = rgb_indices(rgb)
    mask = candidate_mask(idx, meta, vegetation_percentile)
    resp = response_from_indices(idx, mask, meta)
    labels, nlab = ndi.label(mask)
    sizes = np.bincount(labels.ravel()) if nlab else np.array([0])
    min_area = math.pi * meta.min_crown_radius_px**2 * 0.45
    max_area = math.pi * meta.max_crown_radius_px**2 * 2.5
    min_dist = max(2, int(meta.min_crown_radius_px * 1.1))
    threshold = max(min_score, float(np.nanmax(resp) * 0.25))
    coords = peak_local_max(resp, min_distance=min_dist, threshold_abs=threshold, exclude_border=False)
    if valid_box is None:
        valid_box = (0, 0, int(window.width), int(window.height))
    x0, y0, x1, y1 = valid_box
    best: dict[int, Candidate] = {}
    for y, x in coords:
        if x < x0 or x >= x1 or y < y0 or y >= y1:
            continue
        lab = int(labels[y, x])
        if lab <= 0:
            continue
        area = float(sizes[lab])
        if area < min_area or area > max_area:
            continue
        score = float(resp[y, x])
        radius = float(np.clip(math.sqrt(area / math.pi) * 0.70, meta.min_crown_radius_px, meta.max_crown_radius_px))
        cand = Candidate(float(x + window.col_off), float(y + window.row_off), score, area, radius)
        old = best.get(lab)
        if old is None or cand.score > old.score:
            best[lab] = cand
    return list(best.values()), idx


def feature_names() -> list[str]:
    names = []
    for band in ["red", "green", "blue", "exg", "gli", "brightness", "saturation", "green_dom", "veg"]:
        names.extend([f"{band}_mean", f"{band}_std", f"{band}_p25", f"{band}_p75", f"{band}_center"])
    names.extend(["score", "area_px", "radius_px", "local_contrast", "green_ratio"])
    return names


def point_features_from_window(
    idx: dict[str, np.ndarray],
    local_x: float,
    local_y: float,
    meta: ModelMeta,
    score: float = 0.0,
    area_px: float = 0.0,
    radius_px: float | None = None,
) -> np.ndarray:
    r = int(max(3, round(meta.crown_radius_px)))
    x = int(round(local_x))
    y = int(round(local_y))
    h, w = idx["veg"].shape
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    feats: list[float] = []
    for name in ["red", "green", "blue", "exg", "gli", "brightness", "saturation", "green_dom", "veg"]:
        patch = idx[name][y0:y1, x0:x1]
        center = idx[name][min(max(y, 0), h - 1), min(max(x, 0), w - 1)]
        if patch.size == 0:
            vals = [0, 0, 0, 0, float(center)]
        else:
            vals = [float(np.nanmean(patch)), float(np.nanstd(patch)), float(np.nanpercentile(patch, 25)), float(np.nanpercentile(patch, 75)), float(center)]
        feats.extend(vals)
    veg = idx["veg"]
    inner = veg[max(0, y - r // 2): min(h, y + r // 2 + 1), max(0, x - r // 2): min(w, x + r // 2 + 1)]
    outer = veg[y0:y1, x0:x1]
    local_contrast = float(np.nanmean(inner) - np.nanmean(outer)) if inner.size and outer.size else 0.0
    green_ratio = float(idx["green"][min(max(y, 0), h - 1), min(max(x, 0), w - 1)] / (idx["red"][min(max(y, 0), h - 1), min(max(x, 0), w - 1)] + idx["blue"][min(max(y, 0), h - 1), min(max(x, 0), w - 1)] + EPS))
    feats.extend([score, area_px, radius_px or meta.crown_radius_px, local_contrast, green_ratio])
    return np.array(feats, dtype=np.float32)


def iter_tiles(width: int, height: int, tile_size: int, overlap: int):
    stride = tile_size - overlap
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            w = min(tile_size, width - col)
            h = min(tile_size, height - row)
            left = 0 if col == 0 else overlap // 2
            top = 0 if row == 0 else overlap // 2
            right = int(w) if col + w >= width else int(w - overlap // 2)
            bottom = int(h) if row + h >= height else int(h - overlap // 2)
            yield Window(col, row, w, h), (left, top, right, bottom)


def extract_training_features(
    src: rasterio.io.DatasetReader,
    reference_px: np.ndarray,
    meta: ModelMeta,
    max_positive: int,
    negative_ratio: float,
    tile_size: int,
    vegetation_percentile: float,
    min_score: float,
    positive_distance_m: float,
    negative_distance_m: float,
    anchor_positive_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    if len(reference_px) > max_positive:
        pos_idx = rng.choice(len(reference_px), size=max_positive, replace=False)
        positives = reference_px[pos_idx]
    else:
        positives = reference_px
    X: list[np.ndarray] = []
    y: list[int] = []
    half = int(max(tile_size // 2, meta.crown_radius_px * 4))
    pos_tree = cKDTree(reference_px)

    # First train on the same object distribution used at prediction time:
    # proposed candidates. Candidates near reference points are positives;
    # candidates far from any reference point are hard negatives.
    candidate_positive_limit = max_positive
    candidate_negative_limit = int(max_positive * negative_ratio)
    candidate_pos_count = 0
    candidate_neg_count = 0
    positive_distance_px = positive_distance_m / meta.pixel_size_m
    negative_distance_px = negative_distance_m / meta.pixel_size_m
    seen_tiles: set[tuple[int, int]] = set()
    for px, py in positives:
        if candidate_pos_count >= candidate_positive_limit:
            break
        col = int(np.clip(px - half, 0, max(0, src.width - tile_size)))
        row = int(np.clip(py - half, 0, max(0, src.height - tile_size)))
        key = (col, row)
        if key in seen_tiles:
            continue
        seen_tiles.add(key)
        win = Window(col, row, min(tile_size, src.width - col), min(tile_size, src.height - row))
        cands, idx = propose_candidates_for_window(
            src,
            win,
            meta,
            max(40.0, vegetation_percentile - 28.0),
            max(0.01, min_score * 0.10),
        )
        for c in cands:
            dist, _ = pos_tree.query([c.x, c.y], k=1)
            if dist <= positive_distance_px:
                X.append(point_features_from_window(idx, c.x - win.col_off, c.y - win.row_off, meta, c.score, c.area_px, c.radius_px))
                y.append(1)
                candidate_pos_count += 1
            elif dist >= negative_distance_px and candidate_neg_count < candidate_negative_limit:
                X.append(point_features_from_window(idx, c.x - win.col_off, c.y - win.row_off, meta, c.score, c.area_px, c.radius_px))
                y.append(0)
                candidate_neg_count += 1

    # Add a small number of exact reference points as anchors, but keep them from
    # dominating training. Prediction classifies proposed candidates, not perfect
    # manual points.
    anchor_count = min(len(positives), int(max(0, candidate_pos_count) * anchor_positive_fraction))
    if candidate_pos_count == 0:
        anchor_count = min(len(positives), max_positive)
    anchor_points = positives
    if len(anchor_points) > anchor_count:
        anchor_points = anchor_points[rng.choice(len(anchor_points), size=anchor_count, replace=False)]
    for px, py in anchor_points:
        col = int(np.clip(px - half, 0, max(0, src.width - tile_size)))
        row = int(np.clip(py - half, 0, max(0, src.height - tile_size)))
        win = Window(col, row, min(tile_size, src.width - col), min(tile_size, src.height - row))
        rgb = read_rgb(src, win, meta.rgb_bands)
        idx = rgb_indices(rgb)
        X.append(point_features_from_window(idx, px - win.col_off, py - win.row_off, meta, score=1.0))
        y.append(1)

    target_neg = int(len(positives) * negative_ratio)
    negs: list[tuple[float, float, float, float, float]] = []
    total_tiles = math.ceil(src.width / tile_size) * math.ceil(src.height / tile_size)
    max_tiles = min(total_tiles, 140)
    tile_ids = rng.choice(total_tiles, size=max_tiles, replace=False)
    tile_list = list(iter_tiles(src.width, src.height, tile_size, 0))
    for tid in tile_ids:
        if len(negs) >= target_neg:
            break
        win, _ = tile_list[int(tid)]
        cands, _ = propose_candidates_for_window(src, win, meta, vegetation_percentile, min_score * 0.75)
        for c in cands:
            dist, _ = pos_tree.query([c.x, c.y], k=1)
            if dist > negative_distance_px:
                negs.append((c.x, c.y, c.score, c.area_px, c.radius_px))
                if len(negs) >= target_neg:
                    break

    while len(negs) < target_neg:
        px = float(rng.uniform(0, src.width))
        py = float(rng.uniform(0, src.height))
        dist, _ = pos_tree.query([px, py], k=1)
        if dist > negative_distance_px:
            negs.append((px, py, 0.0, 0.0, meta.crown_radius_px))

    for px, py, score, area, radius in negs:
        col = int(np.clip(px - half, 0, max(0, src.width - tile_size)))
        row = int(np.clip(py - half, 0, max(0, src.height - tile_size)))
        win = Window(col, row, min(tile_size, src.width - col), min(tile_size, src.height - row))
        rgb = read_rgb(src, win, meta.rgb_bands)
        idx = rgb_indices(rgb)
        X.append(point_features_from_window(idx, px - win.col_off, py - win.row_off, meta, score=score, area_px=area, radius_px=radius))
        y.append(0)

    return np.vstack(X), np.array(y, dtype=np.uint8)


def train(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.image) as src:
        reference_px = load_reference_pixels(args.reference_points, src)
        meta = make_meta(src, reference_px, args.spacing_m, tuple(args.rgb_bands))
        meta.feature_names = feature_names()
        X, y = extract_training_features(
            src,
            reference_px,
            meta,
            args.max_positive,
            args.negative_ratio,
            args.sample_tile_size,
            args.vegetation_percentile,
            args.min_score,
            args.positive_distance_m,
            args.negative_distance_m,
            args.anchor_positive_fraction,
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=7, stratify=y)
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=7,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    bundle = {"model": model, "meta": asdict(meta), "report": report}
    joblib.dump(bundle, out_dir / "young_tree_model.joblib")
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "model_meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    print(json.dumps({"model": str(out_dir / "young_tree_model.joblib"), "samples": int(len(y)), "positives": int(y.sum()), "negatives": int((y == 0).sum()), "report": report}, indent=2))


def spatial_nms(cands: list[tuple[Candidate, float]], min_dist: float) -> list[tuple[Candidate, float]]:
    kept: list[tuple[Candidate, float]] = []
    for cand, prob in sorted(cands, key=lambda v: v[1], reverse=True):
        if not kept:
            kept.append((cand, prob))
            continue
        pts = np.array([(k[0].x, k[0].y) for k in kept], dtype=np.float32)
        if np.all(np.hypot(pts[:, 0] - cand.x, pts[:, 1] - cand.y) >= min_dist):
            kept.append((cand, prob))
    return sorted(kept, key=lambda v: (v[0].y, v[0].x))


def export_gpkg(path: Path, detections: list[tuple[Candidate, float]], src: rasterio.io.DatasetReader, layer: str = "arboles_detectados") -> None:
    if path.exists():
        path.unlink()
    schema = {
        "geometry": "Point",
        "properties": {
            "id": "int",
            "prob": "float",
            "score": "float",
            "x_pixel": "float",
            "y_pixel": "float",
            "radius_px": "float",
        },
    }
    with fiona.open(path, "w", driver="GPKG", layer=layer, schema=schema, crs=src.crs) as dst:
        for i, (cand, prob) in enumerate(detections, 1):
            x, y = src.transform * (cand.x + 0.5, cand.y + 0.5)
            dst.write({
                "geometry": mapping(Point(x, y)),
                "properties": {
                    "id": i,
                    "prob": float(prob),
                    "score": float(cand.score),
                    "x_pixel": float(cand.x),
                    "y_pixel": float(cand.y),
                    "radius_px": float(cand.radius_px),
                },
            })


def render_preview(src: rasterio.io.DatasetReader, detections: list[tuple[Candidate, float]], out: Path, rgb_bands: tuple[int, int, int], max_size: int = 5000) -> None:
    scale = min(1.0, max_size / max(src.width, src.height))
    out_w = max(1, int(src.width * scale))
    out_h = max(1, int(src.height * scale))
    arr = src.read(list(rgb_bands), out_shape=(3, out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
    rgb = np.stack([robust_rescale(arr[i]) for i in range(3)], axis=-1)
    bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    for cand, prob in detections:
        r = 3 if prob < 0.85 else 4
        cv2.circle(bgr, (int(round(cand.x * scale)), int(round(cand.y * scale))), r, (0, 255, 255), -1)
        cv2.circle(bgr, (int(round(cand.x * scale)), int(round(cand.y * scale))), r + 1, (0, 0, 0), 1)
    cv2.imwrite(str(out), bgr)


def predict(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    meta = ModelMeta(**bundle["meta"])
    detections: list[tuple[Candidate, float]] = []
    proposed_total = 0
    max_prob_seen = 0.0
    with rasterio.open(args.image) as src:
        total = math.ceil(src.width / (args.tile_size - args.tile_overlap)) * math.ceil(src.height / (args.tile_size - args.tile_overlap))
        for i, (win, valid) in enumerate(iter_tiles(src.width, src.height, args.tile_size, args.tile_overlap), 1):
            cands, idx = propose_candidates_for_window(src, win, meta, args.vegetation_percentile, args.min_score, valid)
            proposed_total += len(cands)
            if cands:
                X = np.vstack([
                    point_features_from_window(idx, c.x - win.col_off, c.y - win.row_off, meta, c.score, c.area_px, c.radius_px)
                    for c in cands
                ])
                probs = model.predict_proba(X)[:, 1]
                max_prob_seen = max(max_prob_seen, float(np.max(probs)))
                for cand, prob in zip(cands, probs):
                    if prob >= args.prob_threshold:
                        detections.append((cand, float(prob)))
            if i == 1 or i % 25 == 0:
                print(f"Processed tile {i}/{total}; proposed={proposed_total}; accepted before NMS={len(detections)}; max_prob={max_prob_seen:.3f}")

        print(f"Proposed candidates: {proposed_total}")
        print(f"Accepted before NMS: {len(detections)}")
        print(f"Max probability seen: {max_prob_seen:.4f}")
        final = spatial_nms(detections, meta.spacing_px * args.nms_spacing_factor)
        print(f"Final detections after NMS: {len(final)}")
        export_gpkg(args.out_dir / "arboles_detectados.gpkg", final, src)
        (args.out_dir / "arboles_detectados.qml").write_text(QML_YELLOW_POINTS, encoding="utf-8")
        render_preview(src, final, args.out_dir / "detections_preview.png", meta.rgb_bands)

        with (args.out_dir / "arboles_detectados.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x_pixel", "y_pixel", "prob", "score"])
            for n, (cand, prob) in enumerate(final, 1):
                writer.writerow([n, f"{cand.x:.2f}", f"{cand.y:.2f}", f"{prob:.4f}", f"{cand.score:.4f}"])

    print(f"Outputs written to: {args.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and run an ML-assisted young tree detector.")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--image", type=Path, required=True)
    tr.add_argument("--reference-points", type=Path, required=True)
    tr.add_argument("--out-dir", type=Path, required=True)
    tr.add_argument("--rgb-bands", type=int, nargs=3, default=(1, 2, 3))
    tr.add_argument("--spacing-m", type=float, default=None)
    tr.add_argument("--max-positive", type=int, default=7000)
    tr.add_argument("--negative-ratio", type=float, default=2.0)
    tr.add_argument("--sample-tile-size", type=int, default=512)
    tr.add_argument("--vegetation-percentile", type=float, default=76)
    tr.add_argument("--min-score", type=float, default=0.22)
    tr.add_argument("--n-estimators", type=int, default=220)
    tr.add_argument("--positive-distance-m", type=float, default=0.55)
    tr.add_argument("--negative-distance-m", type=float, default=1.00)
    tr.add_argument("--anchor-positive-fraction", type=float, default=0.08)
    tr.set_defaults(func=train)

    pr = sub.add_parser("predict")
    pr.add_argument("--image", type=Path, required=True)
    pr.add_argument("--model", type=Path, required=True)
    pr.add_argument("--out-dir", type=Path, required=True)
    pr.add_argument("--tile-size", type=int, default=1536)
    pr.add_argument("--tile-overlap", type=int, default=256)
    pr.add_argument("--vegetation-percentile", type=float, default=76)
    pr.add_argument("--min-score", type=float, default=0.22)
    pr.add_argument("--prob-threshold", type=float, default=0.70)
    pr.add_argument("--nms-spacing-factor", type=float, default=0.55)
    pr.set_defaults(func=predict)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
