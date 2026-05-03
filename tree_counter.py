#!/usr/bin/env python3
"""
Detect and count orchard trees in large aerial GeoTIFFs without machine learning.

Pipeline:
  1. Stream the GeoTIFF by tiles, never loading the full raster at native size.
  2. Build a vegetation/crown-likelihood image from RGB or NIR bands.
  3. Auto-calibrate crown radius and tree spacing from sample windows.
  4. Detect local maxima inside conservative vegetation masks.
  5. Merge overlapping tile detections with non-maximum suppression.
  6. Filter candidates using size/score and row/spacing regularity.
  7. Export CSV, GeoJSON points, GeoJSON crown circles, and annotated previews.

Dependencies:
  pip install rasterio numpy scipy scikit-image opencv-python

Example:
  python tree_counter.py input.tif --out-dir out --rgb-bands 1 2 3

For NIR imagery:
  python tree_counter.py input.tif --out-dir out --rgb-bands 1 2 3 --nir-band 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import blob_log, peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects


EPS = 1e-6


@dataclass
class DetectorConfig:
    tile_size: int = 1024
    tile_overlap: int = 160
    sample_windows: int = 9
    preview_max_size: int = 4500
    rgb_bands: tuple[int, int, int] = (1, 2, 3)
    nir_band: int | None = None
    red_edge_band: int | None = None
    min_crown_radius_px: float | None = None
    max_crown_radius_px: float | None = None
    expected_spacing_px: float | None = None
    vegetation_percentile: float = 62.0
    min_brightness_percentile: float = 8.0
    max_shadow_fraction: float = 0.55
    min_component_area_factor: float = 0.18
    max_component_area_factor: float = 4.50
    peak_threshold_rel: float = 0.14
    min_score: float = 0.20
    row_filter: bool = True
    row_angle_tolerance_deg: float = 18.0
    row_spacing_tolerance: float = 0.38
    min_row_support: int = 1
    nms_distance_factor: float = 0.72
    circle_segments: int = 24
    write_annotated_tif: bool = False
    debug: bool = False


@dataclass
class Calibration:
    crown_radius_px: float
    min_crown_radius_px: float
    max_crown_radius_px: float
    spacing_px: float
    row_angle_rad: float | None
    vegetation_threshold: float
    brightness_threshold: float


@dataclass
class Detection:
    x: float
    y: float
    score: float
    radius_px: float
    tile_id: int
    support: int = 0
    component_area: float = 0.0


def robust_rescale(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    good = np.isfinite(arr)
    if not np.any(good):
        return np.zeros(arr.shape, dtype=np.float32)
    p1, p2 = np.nanpercentile(arr[good], [low, high])
    if p2 <= p1:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - p1) / (p2 - p1), 0, 1).astype(np.float32)


def read_bands_normalized(src: rasterio.io.DatasetReader, window: Window, cfg: DetectorConfig) -> tuple[np.ndarray, np.ndarray | None]:
    bands = list(cfg.rgb_bands)
    if cfg.nir_band is not None:
        bands.append(cfg.nir_band)
    data = src.read(bands, window=window, boundless=True, fill_value=0).astype(np.float32)
    rgb = np.stack([robust_rescale(data[i]) for i in range(3)], axis=-1)
    nir = robust_rescale(data[3]) if cfg.nir_band is not None else None
    return rgb, nir


def vegetation_index(rgb: np.ndarray, nir: np.ndarray | None) -> np.ndarray:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    if nir is not None:
        ndvi = (nir - red) / (nir + red + EPS)
        exg = 2 * green - red - blue
        return robust_rescale(0.75 * robust_rescale(ndvi) + 0.25 * robust_rescale(exg))
    exg = 2 * green - red - blue
    gli = (2 * green - red - blue) / (2 * green + red + blue + EPS)
    brightness = (red + green + blue) / 3
    green_contrast = green - np.maximum(red, blue)
    return robust_rescale(0.50 * robust_rescale(exg) + 0.35 * robust_rescale(gli) + 0.15 * robust_rescale(green_contrast - 0.15 * brightness))


def make_crown_mask(index: np.ndarray, rgb: np.ndarray, calib: Calibration | None, cfg: DetectorConfig) -> np.ndarray:
    brightness = rgb.mean(axis=-1)
    finite_idx = index[np.isfinite(index)]
    if finite_idx.size > 16 and float(np.nanmax(finite_idx) - np.nanmin(finite_idx)) > EPS:
        otsu_thr = float(threshold_otsu(finite_idx))
    else:
        otsu_thr = 0.5
    if calib is None:
        veg_thr = max(float(np.nanpercentile(index, cfg.vegetation_percentile)), otsu_thr * 0.92)
        bright_thr = min(
            float(np.nanpercentile(brightness, cfg.min_brightness_percentile)),
            float(np.nanmedian(brightness) * 0.55),
        )
        radius = cfg.min_crown_radius_px or 4
    else:
        veg_thr = max(calib.vegetation_threshold, otsu_thr * 0.92)
        bright_thr = min(calib.brightness_threshold, float(np.nanmedian(brightness) * 0.55))
        radius = max(2, calib.min_crown_radius_px)

    mask = (index >= veg_thr) & (brightness >= bright_thr)
    morph_r = max(1, int(round(radius * 0.20)))
    min_obj = max(8, int(math.pi * (radius * 0.45) ** 2))
    mask = binary_opening(mask, disk(morph_r))
    mask = binary_closing(mask, disk(morph_r))
    mask = remove_small_objects(mask, min_size=min_obj)
    mask = remove_small_holes(mask, area_threshold=max(16, int(min_obj * 0.75)))
    return mask.astype(bool)


def response_image(index: np.ndarray, mask: np.ndarray, radius_px: float) -> np.ndarray:
    sigma_small = max(1.0, radius_px * 0.22)
    sigma_large = max(sigma_small + 0.5, radius_px * 0.95)
    smooth = ndi.gaussian_filter(index, sigma=sigma_small)
    background = ndi.gaussian_filter(index, sigma=sigma_large)
    resp = smooth - background
    resp[~mask] = 0
    dist = ndi.distance_transform_edt(mask)
    dist = ndi.gaussian_filter(dist, sigma=max(0.6, radius_px * 0.08))
    combined = 0.72 * robust_rescale(resp) + 0.28 * robust_rescale(dist)
    combined[~mask] = 0
    return robust_rescale(combined)


def estimate_from_candidates(points: np.ndarray, radii: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if len(points) < 8:
        return None, None, None
    if len(radii) > 0:
        median_radius = float(np.nanmedian(radii))
        keep_distance = max(3.0, median_radius * 1.65)
        order = np.argsort(-scores)
        kept_idx: list[int] = []
        kept_points: list[np.ndarray] = []
        for idx in order:
            p = points[idx]
            if not kept_points or np.all(np.linalg.norm(np.vstack(kept_points) - p, axis=1) >= keep_distance):
                kept_idx.append(int(idx))
                kept_points.append(p)
        if len(kept_idx) >= 8:
            points = points[kept_idx]
            radii = radii[kept_idx]
            scores = scores[kept_idx]
    median_radius = float(np.nanmedian(radii)) if len(radii) else 2.0
    tree = cKDTree(points)
    dists, idx = tree.query(points, k=min(7, len(points)))
    nn = dists[:, 1:].reshape(-1)
    nn = nn[(nn > max(2, median_radius * 2.4)) & np.isfinite(nn)]
    if len(nn) == 0:
        spacing = None
    else:
        spacing = float(np.nanpercentile(nn, 35))
    if len(radii) > 0:
        weights = np.clip(scores, 0.05, 1.0)
        radius = float(np.average(radii, weights=weights))
    else:
        radius = None
    angle = estimate_row_angle(points, spacing) if spacing else None
    return radius, spacing, angle


def estimate_row_angle(points: np.ndarray, spacing_px: float | None) -> float | None:
    if spacing_px is None or len(points) < 12:
        return None
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=max(12.0, spacing_px * 1.65), output_type="ndarray")
    if len(pairs) < 6:
        return None
    vec = points[pairs[:, 1]] - points[pairs[:, 0]]
    dist = np.linalg.norm(vec, axis=1)
    keep = (dist > spacing_px * 0.55) & (dist < spacing_px * 1.55)
    vec = vec[keep]
    if len(vec) < 6:
        return None
    angles = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), np.pi)
    doubled = np.exp(2j * angles)
    mean = np.angle(np.mean(doubled)) / 2.0
    return float(np.mod(mean, np.pi))


def sample_windows(src: rasterio.io.DatasetReader, cfg: DetectorConfig) -> list[Window]:
    width, height = src.width, src.height
    size = min(cfg.tile_size, width, height)
    if cfg.sample_windows <= 1:
        return [Window(max(0, (width - size) // 2), max(0, (height - size) // 2), size, size)]

    grid_n = int(math.ceil(math.sqrt(cfg.sample_windows)))
    xs = np.linspace(size // 2, width - size // 2, grid_n)
    ys = np.linspace(size // 2, height - size // 2, grid_n)
    windows: list[Window] = []
    for y in ys:
        for x in xs:
            if len(windows) >= cfg.sample_windows:
                break
            col = int(np.clip(x - size / 2, 0, max(0, width - size)))
            row = int(np.clip(y - size / 2, 0, max(0, height - size)))
            windows.append(Window(col, row, size, size))
    return windows


def auto_calibrate(src: rasterio.io.DatasetReader, cfg: DetectorConfig) -> Calibration:
    all_points: list[np.ndarray] = []
    all_radii: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    veg_values: list[float] = []
    bright_values: list[float] = []

    fallback_radius = cfg.min_crown_radius_px or 8.0
    for win in sample_windows(src, cfg):
        rgb, nir = read_bands_normalized(src, win, cfg)
        idx = vegetation_index(rgb, nir)
        brightness = rgb.mean(axis=-1)
        finite_idx = idx[np.isfinite(idx)]
        if finite_idx.size > 16 and float(np.nanmax(finite_idx) - np.nanmin(finite_idx)) > EPS:
            local_otsu = float(threshold_otsu(finite_idx))
        else:
            local_otsu = 0.5
        veg_values.append(max(float(np.nanpercentile(idx, cfg.vegetation_percentile)), local_otsu * 0.92))
        bright_values.append(min(
            float(np.nanpercentile(brightness, cfg.min_brightness_percentile)),
            float(np.nanmedian(brightness) * 0.55),
        ))
        tmp_calib = Calibration(fallback_radius, fallback_radius * 0.55, fallback_radius * 1.8, fallback_radius * 4, None, veg_values[-1], bright_values[-1])
        mask = make_crown_mask(idx, rgb, tmp_calib, cfg)

        labels, nlab = ndi.label(mask)
        if nlab > 0:
            sizes = np.bincount(labels.ravel())
            candidate_labels = np.where((sizes >= 20) & (sizes <= max(80, math.pi * (fallback_radius * 3.2) ** 2)))[0]
            candidate_labels = candidate_labels[candidate_labels != 0]
            if len(candidate_labels):
                centers = ndi.center_of_mass(mask, labels, candidate_labels)
                centers_arr = np.array([(c[1] + win.col_off, c[0] + win.row_off) for c in centers if np.isfinite(c[0]) and np.isfinite(c[1])], dtype=np.float32)
                comp_areas = sizes[candidate_labels[: len(centers_arr)]].astype(np.float32)
                comp_radii = np.sqrt(comp_areas / math.pi) * 0.72
                if len(centers_arr):
                    all_points.append(centers_arr)
                    all_radii.append(comp_radii)
                    all_scores.append(np.full(len(centers_arr), 0.65, dtype=np.float32))

        min_sigma = max(1.2, (cfg.min_crown_radius_px or 3.0) / math.sqrt(2))
        max_sigma = max(min_sigma + 1.0, (cfg.max_crown_radius_px or 18.0) / math.sqrt(2))
        blobs = blob_log(idx * mask, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=8, threshold=0.035, overlap=0.55)
        if len(blobs) == 0:
            continue
        yx = blobs[:, :2]
        radii = blobs[:, 2] * math.sqrt(2)
        response = response_image(idx, mask, float(np.nanmedian(radii)))
        yi = np.clip(np.round(yx[:, 0]).astype(int), 0, response.shape[0] - 1)
        xi = np.clip(np.round(yx[:, 1]).astype(int), 0, response.shape[1] - 1)
        scores = response[yi, xi]
        keep = scores >= max(0.08, np.nanpercentile(scores, 45))
        pts = np.column_stack([yx[keep, 1] + win.col_off, yx[keep, 0] + win.row_off])
        if len(pts):
            all_points.append(pts)
            all_radii.append(radii[keep])
            all_scores.append(scores[keep])

    if all_points:
        points = np.vstack(all_points)
        radii = np.concatenate(all_radii)
        scores = np.concatenate(all_scores)
        radius, spacing, angle = estimate_from_candidates(points, radii, scores)
    else:
        radius, spacing, angle = None, None, None

    crown_radius = float(cfg.min_crown_radius_px or radius or 8.0)
    if radius is not None and cfg.min_crown_radius_px is None:
        crown_radius = float(np.clip(radius, 3.0, 40.0))
    min_radius = float(cfg.min_crown_radius_px or max(2.5, crown_radius * 0.55))
    max_radius = float(cfg.max_crown_radius_px or max(min_radius + 1, crown_radius * 1.85))
    spacing_px = float(cfg.expected_spacing_px or spacing or crown_radius * 3.5)
    veg_thr = float(np.nanmedian(veg_values)) if veg_values else 0.45
    bright_thr = float(np.nanmedian(bright_values)) if bright_values else 0.08
    return Calibration(crown_radius, min_radius, max_radius, spacing_px, angle, veg_thr, bright_thr)


def iter_tiles(width: int, height: int, tile_size: int, overlap: int) -> Iterable[tuple[int, Window, tuple[int, int, int, int]]]:
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile_size")
    tile_id = 0
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            w = min(tile_size, width - col)
            h = min(tile_size, height - row)
            window = Window(col, row, w, h)
            left_margin = 0 if col == 0 else overlap // 2
            top_margin = 0 if row == 0 else overlap // 2
            right_margin = 0 if col + w >= width else overlap // 2
            bottom_margin = 0 if row + h >= height else overlap // 2
            valid = (left_margin, top_margin, int(w - right_margin), int(h - bottom_margin))
            yield tile_id, window, valid
            tile_id += 1


def component_area_at(mask: np.ndarray, y: int, x: int) -> int:
    labels, _ = ndi.label(mask)
    lab = labels[y, x]
    if lab == 0:
        return 0
    return int(np.sum(labels == lab))


def detect_tile(src: rasterio.io.DatasetReader, window: Window, valid_box: tuple[int, int, int, int], tile_id: int, calib: Calibration, cfg: DetectorConfig) -> list[Detection]:
    rgb, nir = read_bands_normalized(src, window, cfg)
    idx = vegetation_index(rgb, nir)
    mask = make_crown_mask(idx, rgb, calib, cfg)
    resp = response_image(idx, mask, calib.crown_radius_px)

    min_distance = max(2, int(round(calib.min_crown_radius_px * 0.85)))
    threshold_abs = max(cfg.min_score, float(np.nanmax(resp) * cfg.peak_threshold_rel))
    coords = peak_local_max(resp, min_distance=min_distance, threshold_abs=threshold_abs, exclude_border=False)

    min_area = math.pi * (calib.crown_radius_px ** 2) * cfg.min_component_area_factor
    max_area = math.pi * (calib.crown_radius_px ** 2) * cfg.max_component_area_factor
    detections: list[Detection] = []
    x0, y0, x1, y1 = valid_box

    labeled, _ = ndi.label(mask)
    component_sizes = np.bincount(labeled.ravel())
    for y, x in coords:
        if x < x0 or x >= x1 or y < y0 or y >= y1:
            continue
        lab = labeled[y, x]
        area = float(component_sizes[lab]) if lab > 0 else 0.0
        if area < min_area or area > max_area:
            continue
        score = float(resp[y, x])
        if score < cfg.min_score:
            continue
        radius = float(np.clip(math.sqrt(max(area, 1.0) / math.pi) * 0.72, calib.min_crown_radius_px, calib.max_crown_radius_px))
        detections.append(Detection(float(x + window.col_off), float(y + window.row_off), score, radius, tile_id, 0, area))
    return detections


def nms_detections(dets: list[Detection], min_dist: float) -> list[Detection]:
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    tree_points: list[tuple[float, float]] = []
    for det in ordered:
        if not kept:
            kept.append(det)
            tree_points.append((det.x, det.y))
            continue
        pts = np.array(tree_points, dtype=np.float32)
        dist = np.hypot(pts[:, 0] - det.x, pts[:, 1] - det.y)
        if np.all(dist >= min_dist):
            kept.append(det)
            tree_points.append((det.x, det.y))
    return sorted(kept, key=lambda d: (d.y, d.x))


def angular_difference(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * 2 * (a - b))) / 2)


def row_regular_filter(dets: list[Detection], calib: Calibration, cfg: DetectorConfig) -> list[Detection]:
    if not cfg.row_filter or len(dets) < 12:
        return dets
    points = np.array([(d.x, d.y) for d in dets], dtype=np.float32)
    spacing = calib.spacing_px
    angle = calib.row_angle_rad if calib.row_angle_rad is not None else estimate_row_angle(points, spacing)
    if angle is None:
        return dets

    tree = cKDTree(points)
    radius = max(spacing * (1.0 + cfg.row_spacing_tolerance), calib.max_crown_radius_px * 3.0)
    pairs = tree.query_ball_point(points, r=radius)
    kept: list[Detection] = []
    angle_tol = math.radians(cfg.row_angle_tolerance_deg)
    for i, neigh in enumerate(pairs):
        support = 0
        for j in neigh:
            if i == j:
                continue
            dx, dy = points[j] - points[i]
            dist = math.hypot(float(dx), float(dy))
            if dist < spacing * (1.0 - cfg.row_spacing_tolerance) or dist > spacing * (1.0 + cfg.row_spacing_tolerance):
                continue
            theta = math.atan2(float(dy), float(dx)) % math.pi
            row_match = angular_difference(np.array([theta]), angle)[0] <= angle_tol
            cross_row_match = angular_difference(np.array([theta]), (angle + math.pi / 2) % math.pi)[0] <= angle_tol
            if row_match or cross_row_match:
                support += 1
        dets[i].support = support
        if support >= cfg.min_row_support or dets[i].score >= max(0.58, cfg.min_score + 0.22):
            kept.append(dets[i])
    return kept


def pixel_to_world(transform: Affine, x: float, y: float) -> tuple[float, float]:
    wx, wy = transform * (x + 0.5, y + 0.5)
    return float(wx), float(wy)


def transform_radius(transform: Affine, radius_px: float) -> float:
    xres = math.hypot(transform.a, transform.d)
    yres = math.hypot(transform.b, transform.e)
    return float(radius_px * (abs(xres) + abs(yres)) / 2.0)


def circle_polygon(cx: float, cy: float, radius: float, segments: int) -> list[list[float]]:
    pts = []
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        pts.append([cx + math.cos(a) * radius, cy + math.sin(a) * radius])
    return pts


def export_csv(path: Path, dets: list[Detection], transform: Affine) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pixel_x", "pixel_y", "x", "y", "score", "radius_px", "support", "component_area_px"])
        for i, d in enumerate(dets, start=1):
            wx, wy = pixel_to_world(transform, d.x, d.y)
            writer.writerow([i, f"{d.x:.3f}", f"{d.y:.3f}", f"{wx:.8f}", f"{wy:.8f}", f"{d.score:.4f}", f"{d.radius_px:.3f}", d.support, f"{d.component_area:.1f}"])


def export_geojson_points(path: Path, dets: list[Detection], src: rasterio.io.DatasetReader) -> None:
    features = []
    for i, d in enumerate(dets, start=1):
        wx, wy = pixel_to_world(src.transform, d.x, d.y)
        features.append({
            "type": "Feature",
            "properties": {"id": i, "score": d.score, "radius_px": d.radius_px, "support": d.support},
            "geometry": {"type": "Point", "coordinates": [wx, wy]},
        })
    collection = {"type": "FeatureCollection", "name": "tree_points", "crs": crs_obj(src), "features": features}
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def export_geojson_circles(path: Path, dets: list[Detection], src: rasterio.io.DatasetReader, cfg: DetectorConfig) -> None:
    features = []
    for i, d in enumerate(dets, start=1):
        wx, wy = pixel_to_world(src.transform, d.x, d.y)
        radius = transform_radius(src.transform, d.radius_px)
        features.append({
            "type": "Feature",
            "properties": {"id": i, "score": d.score, "radius_px": d.radius_px, "radius_map": radius},
            "geometry": {"type": "Polygon", "coordinates": [circle_polygon(wx, wy, radius, cfg.circle_segments)]},
        })
    collection = {"type": "FeatureCollection", "name": "tree_crowns", "crs": crs_obj(src), "features": features}
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def crs_obj(src: rasterio.io.DatasetReader) -> dict | None:
    if src.crs is None:
        return None
    return {"type": "name", "properties": {"name": src.crs.to_string()}}


def render_preview(src: rasterio.io.DatasetReader, dets: list[Detection], path: Path, cfg: DetectorConfig, calib: Calibration) -> None:
    scale = min(1.0, cfg.preview_max_size / max(src.width, src.height))
    out_w = max(1, int(src.width * scale))
    out_h = max(1, int(src.height * scale))
    rgb = src.read(list(cfg.rgb_bands), out_shape=(3, out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
    img = np.stack([robust_rescale(rgb[i]) for i in range(3)], axis=-1)
    img8 = (img * 255).astype(np.uint8)
    img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)

    for d in dets:
        cx = int(round(d.x * scale))
        cy = int(round(d.y * scale))
        rad = max(2, int(round(max(d.radius_px, calib.crown_radius_px * 0.65) * scale)))
        cv2.circle(img8, (cx, cy), rad, (0, 255, 255), max(1, int(round(2 * scale))))
        cv2.circle(img8, (cx, cy), 1, (0, 0, 255), -1)
    cv2.imwrite(str(path), img8)


def write_annotated_geotiff(src: rasterio.io.DatasetReader, dets: list[Detection], path: Path, cfg: DetectorConfig, calib: Calibration) -> None:
    profile = src.profile.copy()
    profile.update(count=3, dtype="uint8", compress="deflate", photometric="RGB")
    points = np.array([(d.x, d.y, max(d.radius_px, calib.crown_radius_px * 0.65)) for d in dets], dtype=np.float32)
    tree = cKDTree(points[:, :2]) if len(points) else None
    with rasterio.open(path, "w", **profile) as dst:
        for _, win, _ in iter_tiles(src.width, src.height, cfg.tile_size, 0):
            rgb, _ = read_bands_normalized(src, win, cfg)
            img = (rgb * 255).astype(np.uint8)
            if tree is not None:
                cx = win.col_off + win.width / 2
                cy = win.row_off + win.height / 2
                search_r = math.hypot(win.width, win.height) / 2 + calib.max_crown_radius_px + 4
                idxs = tree.query_ball_point([cx, cy], r=search_r)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for idx in idxs:
                    x, y, r = points[idx]
                    lx = int(round(x - win.col_off))
                    ly = int(round(y - win.row_off))
                    if -r <= lx < win.width + r and -r <= ly < win.height + r:
                        cv2.circle(bgr, (lx, ly), int(round(r)), (0, 255, 255), 2)
                        cv2.circle(bgr, (lx, ly), 1, (0, 0, 255), -1)
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dst.write(np.moveaxis(img, -1, 0), window=win)


def save_debug_masks(src: rasterio.io.DatasetReader, out_dir: Path, cfg: DetectorConfig, calib: Calibration) -> None:
    win = sample_windows(src, cfg)[0]
    rgb, nir = read_bands_normalized(src, win, cfg)
    idx = vegetation_index(rgb, nir)
    mask = make_crown_mask(idx, rgb, calib, cfg)
    resp = response_image(idx, mask, calib.crown_radius_px)
    cv2.imwrite(str(out_dir / "debug_vegetation_index.png"), (idx * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "debug_crown_mask.png"), (mask.astype(np.uint8) * 255))
    cv2.imwrite(str(out_dir / "debug_peak_response.png"), (resp * 255).astype(np.uint8))


def run(input_tif: Path, out_dir: Path, cfg: DetectorConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(input_tif) as src:
        if max(cfg.rgb_bands) > src.count or (cfg.nir_band is not None and cfg.nir_band > src.count):
            raise ValueError(f"Band selection exceeds raster band count ({src.count}).")

        print("Auto-calibrating crown size, spacing, and row direction...")
        calib = auto_calibrate(src, cfg)
        (out_dir / "calibration.json").write_text(json.dumps(asdict(calib), indent=2), encoding="utf-8")
        print(json.dumps(asdict(calib), indent=2))

        detections: list[Detection] = []
        total_tiles = math.ceil(src.width / max(1, cfg.tile_size - cfg.tile_overlap)) * math.ceil(src.height / max(1, cfg.tile_size - cfg.tile_overlap))
        for tile_id, win, valid in iter_tiles(src.width, src.height, cfg.tile_size, cfg.tile_overlap):
            detections.extend(detect_tile(src, win, valid, tile_id, calib, cfg))
            if tile_id % 25 == 0:
                print(f"Processed tile {tile_id + 1}/{total_tiles}; raw detections={len(detections)}")

        print(f"Raw detections before merge: {len(detections)}")
        merge_distance = min(calib.spacing_px * cfg.nms_distance_factor, calib.crown_radius_px * 2.2)
        merged = nms_detections(detections, merge_distance)
        print(f"After non-maximum suppression: {len(merged)}")
        filtered = row_regular_filter(merged, calib, cfg)
        print(f"After row/spacing filter: {len(filtered)}")

        export_csv(out_dir / "trees.csv", filtered, src.transform)
        export_geojson_points(out_dir / "trees.geojson", filtered, src)
        export_geojson_circles(out_dir / "tree_crowns.geojson", filtered, src, cfg)
        render_preview(src, filtered, out_dir / "detections_preview.png", cfg, calib)
        if cfg.write_annotated_tif:
            write_annotated_geotiff(src, filtered, out_dir / "detections_annotated.tif", cfg, calib)
        if cfg.debug:
            save_debug_masks(src, out_dir, cfg, calib)

        summary = {
            "input": str(input_tif),
            "count": len(filtered),
            "outputs": {
                "csv": "trees.csv",
                "points_geojson": "trees.geojson",
                "crowns_geojson": "tree_crowns.geojson",
                "preview_png": "detections_preview.png",
                "annotated_tif": "detections_annotated.tif" if cfg.write_annotated_tif else None,
            },
            "calibration": asdict(calib),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Final count: {len(filtered)}")
        print(f"Outputs written to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and count orchard trees in large GeoTIFFs without ML.")
    parser.add_argument("input_tif", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("tree_count_out"))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=160)
    parser.add_argument("--rgb-bands", type=int, nargs=3, default=(1, 2, 3), metavar=("R", "G", "B"))
    parser.add_argument("--nir-band", type=int, default=None)
    parser.add_argument("--min-crown-radius-px", type=float, default=None)
    parser.add_argument("--max-crown-radius-px", type=float, default=None)
    parser.add_argument("--expected-spacing-px", type=float, default=None)
    parser.add_argument("--vegetation-percentile", type=float, default=62.0)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--disable-row-filter", action="store_true")
    parser.add_argument("--min-row-support", type=int, default=1)
    parser.add_argument("--write-annotated-tif", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DetectorConfig(
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        rgb_bands=tuple(args.rgb_bands),
        nir_band=args.nir_band,
        min_crown_radius_px=args.min_crown_radius_px,
        max_crown_radius_px=args.max_crown_radius_px,
        expected_spacing_px=args.expected_spacing_px,
        vegetation_percentile=args.vegetation_percentile,
        min_score=args.min_score,
        row_filter=not args.disable_row_filter,
        min_row_support=args.min_row_support,
        write_annotated_tif=args.write_annotated_tif,
        debug=args.debug,
    )
    run(args.input_tif, args.out_dir, cfg)


if __name__ == "__main__":
    main()
