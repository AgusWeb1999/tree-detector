#!/usr/bin/env python3
"""
Count young planted trees in RGB aerial GeoTIFFs without machine learning.

This version is tuned for orchards/plantations where trees are small, planted
in rows, and have a known approximate spacing in meters. It rejects forest and
grass mainly by requiring both crown-like green blobs and local row/spacing
support.

Outputs:
  - trees.csv
  - trees.geojson
  - detections_preview.png with yellow dots
  - optional detections_points.tif with yellow dots georeferenced
  - calibration.json

Dependencies:
  pip install rasterio numpy scipy scikit-image opencv-python
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects


EPS = 1e-6


@dataclass
class Config:
    rgb_bands: tuple[int, int, int] = (1, 2, 3)
    tile_size: int = 1536
    tile_overlap: int = 256
    expected_spacing_m: float = 1.5
    spacing_tolerance: float = 0.45
    min_row_support: int = 2
    vegetation_percentile: float = 78.0
    min_score: float = 0.36
    peak_threshold_rel: float = 0.28
    min_crown_radius_m: float | None = None
    max_crown_radius_m: float | None = None
    preview_max_size: int = 5000
    dot_radius_px: int = 4
    write_points_tif: bool = False
    debug: bool = False
    nms_spacing_factor: float = 0.62


@dataclass
class Calibration:
    pixel_size_m: float
    spacing_px: float
    min_crown_radius_px: float
    max_crown_radius_px: float
    crown_radius_px: float
    vegetation_threshold: float
    brightness_threshold: float


@dataclass
class Detection:
    x: float
    y: float
    score: float
    radius_px: float
    support: int = 0
    component_area_px: float = 0.0


def robust_rescale(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    ok = np.isfinite(arr)
    if not np.any(ok):
        return np.zeros(arr.shape, dtype=np.float32)
    p1, p2 = np.nanpercentile(arr[ok], [low, high])
    if p2 <= p1:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - p1) / (p2 - p1), 0, 1).astype(np.float32)


def pixel_size_from_transform(transform: Affine) -> float:
    xres = math.hypot(transform.a, transform.d)
    yres = math.hypot(transform.b, transform.e)
    px = (abs(xres) + abs(yres)) / 2.0
    if px <= 0:
        raise ValueError("Could not infer pixel size from GeoTIFF transform.")
    return float(px)


def read_rgb(src: rasterio.io.DatasetReader, window: Window, cfg: Config) -> np.ndarray:
    arr = src.read(list(cfg.rgb_bands), window=window, boundless=True, fill_value=0).astype(np.float32)
    return np.stack([robust_rescale(arr[i]) for i in range(3)], axis=-1)


def vegetation_index(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    exg = 2.0 * green - red - blue
    gli = (2.0 * green - red - blue) / (2.0 * green + red + blue + EPS)
    green_chroma = green - np.maximum(red, blue)
    saturation_like = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    return robust_rescale(
        0.46 * robust_rescale(exg)
        + 0.34 * robust_rescale(gli)
        + 0.14 * robust_rescale(green_chroma)
        + 0.06 * robust_rescale(saturation_like)
    )


def crown_mask(index: np.ndarray, rgb: np.ndarray, calib: Calibration, cfg: Config) -> np.ndarray:
    brightness = rgb.mean(axis=-1)
    finite = index[np.isfinite(index)]
    if finite.size > 32 and float(np.nanmax(finite) - np.nanmin(finite)) > EPS:
        otsu_thr = float(threshold_otsu(finite))
    else:
        otsu_thr = calib.vegetation_threshold
    veg_thr = max(calib.vegetation_threshold, otsu_thr * 0.95)
    bright_thr = min(calib.brightness_threshold, float(np.nanmedian(brightness) * 0.70))

    mask = (index >= veg_thr) & (brightness >= bright_thr)
    r = max(1, int(round(calib.min_crown_radius_px * 0.25)))
    min_obj = max(6, int(math.pi * (calib.min_crown_radius_px * 0.45) ** 2))
    mask = binary_opening(mask, disk(r))
    mask = binary_closing(mask, disk(r))
    mask = remove_small_objects(mask, min_size=min_obj)
    mask = remove_small_holes(mask, area_threshold=max(8, min_obj))
    return mask.astype(bool)


def response_image(index: np.ndarray, mask: np.ndarray, calib: Calibration) -> np.ndarray:
    sigma_small = max(0.8, calib.crown_radius_px * 0.22)
    sigma_large = max(sigma_small + 0.5, calib.crown_radius_px * 1.10)
    local_contrast = ndi.gaussian_filter(index, sigma_small) - ndi.gaussian_filter(index, sigma_large)
    distance = ndi.distance_transform_edt(mask)
    resp = 0.70 * robust_rescale(local_contrast) + 0.30 * robust_rescale(distance)
    resp[~mask] = 0
    return robust_rescale(resp)


def sample_windows(src: rasterio.io.DatasetReader, tile_size: int) -> list[Window]:
    size = min(tile_size, src.width, src.height)
    centers = [
        (0.50, 0.50),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
    ]
    windows = []
    for fx, fy in centers:
        col = int(np.clip(src.width * fx - size / 2, 0, max(0, src.width - size)))
        row = int(np.clip(src.height * fy - size / 2, 0, max(0, src.height - size)))
        windows.append(Window(col, row, size, size))
    return windows


def auto_calibrate(src: rasterio.io.DatasetReader, cfg: Config) -> Calibration:
    pixel_size_m = pixel_size_from_transform(src.transform)
    spacing_px = cfg.expected_spacing_m / pixel_size_m
    min_r_px = (cfg.min_crown_radius_m / pixel_size_m) if cfg.min_crown_radius_m else max(2.0, spacing_px * 0.12)
    max_r_px = (cfg.max_crown_radius_m / pixel_size_m) if cfg.max_crown_radius_m else max(min_r_px + 1.0, spacing_px * 0.42)
    crown_r_px = float(np.clip(spacing_px * 0.25, min_r_px, max_r_px))

    veg_thresholds: list[float] = []
    bright_thresholds: list[float] = []
    for win in sample_windows(src, cfg.tile_size):
        rgb = read_rgb(src, win, cfg)
        idx = vegetation_index(rgb)
        brightness = rgb.mean(axis=-1)
        finite = idx[np.isfinite(idx)]
        if finite.size > 32 and float(np.nanmax(finite) - np.nanmin(finite)) > EPS:
            otsu_thr = float(threshold_otsu(finite))
        else:
            otsu_thr = 0.5
        veg_thresholds.append(max(float(np.nanpercentile(idx, cfg.vegetation_percentile)), otsu_thr * 0.95))
        bright_thresholds.append(min(float(np.nanpercentile(brightness, 8)), float(np.nanmedian(brightness) * 0.70)))

    return Calibration(
        pixel_size_m=pixel_size_m,
        spacing_px=float(spacing_px),
        min_crown_radius_px=float(min_r_px),
        max_crown_radius_px=float(max_r_px),
        crown_radius_px=float(crown_r_px),
        vegetation_threshold=float(np.nanmedian(veg_thresholds)),
        brightness_threshold=float(np.nanmedian(bright_thresholds)),
    )


def iter_tiles(width: int, height: int, tile_size: int, overlap: int) -> Iterable[tuple[Window, tuple[int, int, int, int]]]:
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile-overlap must be smaller than tile-size.")
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            w = min(tile_size, width - col)
            h = min(tile_size, height - row)
            win = Window(col, row, w, h)
            left = 0 if col == 0 else overlap // 2
            top = 0 if row == 0 else overlap // 2
            right = int(w) if col + w >= width else int(w - overlap // 2)
            bottom = int(h) if row + h >= height else int(h - overlap // 2)
            yield win, (left, top, right, bottom)


def detect_tile(src: rasterio.io.DatasetReader, win: Window, valid: tuple[int, int, int, int], calib: Calibration, cfg: Config) -> list[Detection]:
    rgb = read_rgb(src, win, cfg)
    idx = vegetation_index(rgb)
    mask = crown_mask(idx, rgb, calib, cfg)
    resp = response_image(idx, mask, calib)

    min_area = math.pi * (calib.min_crown_radius_px ** 2) * 0.65
    max_area = math.pi * (calib.max_crown_radius_px ** 2) * 2.30
    labels, nlab = ndi.label(mask)
    sizes = np.bincount(labels.ravel()) if nlab else np.array([0])

    min_dist = max(2, int(round(calib.min_crown_radius_px * 1.35)))
    threshold = max(cfg.min_score, float(np.nanmax(resp) * cfg.peak_threshold_rel))
    coords = peak_local_max(resp, min_distance=min_dist, threshold_abs=threshold, exclude_border=False)

    x0, y0, x1, y1 = valid
    best_by_component: dict[int, Detection] = {}
    free: list[Detection] = []
    for y, x in coords:
        if x < x0 or x >= x1 or y < y0 or y >= y1:
            continue
        lab = int(labels[y, x])
        area = float(sizes[lab]) if lab > 0 else 0.0
        if lab == 0 or area < min_area or area > max_area:
            continue
        score = float(resp[y, x])
        radius = float(np.clip(math.sqrt(area / math.pi) * 0.70, calib.min_crown_radius_px, calib.max_crown_radius_px))
        det = Detection(float(x + win.col_off), float(y + win.row_off), score, radius, 0, area)
        current = best_by_component.get(lab)
        if current is None or det.score > current.score:
            best_by_component[lab] = det

    out = list(best_by_component.values()) + free
    # Hard cap per tile based on the physical planting density. This is not the
    # final count; it prevents grass texture from flooding later stages.
    expected_cells = (win.width * win.height) / max(1.0, (calib.spacing_px * 0.70) ** 2)
    cap = max(25, int(expected_cells * 1.6))
    return sorted(out, key=lambda d: d.score, reverse=True)[:cap]


def nms(dets: list[Detection], distance_px: float) -> list[Detection]:
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda v: v.score, reverse=True):
        if not kept:
            kept.append(d)
            continue
        pts = np.array([(k.x, k.y) for k in kept], dtype=np.float32)
        if np.all(np.hypot(pts[:, 0] - d.x, pts[:, 1] - d.y) >= distance_px):
            kept.append(d)
    return sorted(kept, key=lambda d: (d.y, d.x))


def angle_diff_pi(a: np.ndarray | float, b: float) -> np.ndarray | float:
    return np.abs(np.angle(np.exp(2j * (np.asarray(a) - b))) / 2.0)


def dominant_row_angle(pts: np.ndarray, calib: Calibration, cfg: Config) -> float | None:
    if len(pts) < 20:
        return None
    tree = cKDTree(pts)
    pairs = tree.query_pairs(calib.spacing_px * (1.0 + cfg.spacing_tolerance), output_type="ndarray")
    if len(pairs) < 20:
        return None
    vec = pts[pairs[:, 1]] - pts[pairs[:, 0]]
    dist = np.linalg.norm(vec, axis=1)
    keep = (dist >= calib.spacing_px * (1.0 - cfg.spacing_tolerance)) & (dist <= calib.spacing_px * (1.0 + cfg.spacing_tolerance))
    vec = vec[keep]
    if len(vec) < 20:
        return None
    angles = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), np.pi)
    hist, edges = np.histogram(angles, bins=36, range=(0, np.pi))
    best = int(np.argmax(hist))
    return float((edges[best] + edges[best + 1]) / 2.0)


def row_filter(dets: list[Detection], calib: Calibration, cfg: Config) -> list[Detection]:
    if len(dets) < 8:
        return dets
    pts = np.array([(d.x, d.y) for d in dets], dtype=np.float32)
    row_angle = dominant_row_angle(pts, calib, cfg)
    if row_angle is None:
        return []

    tree = cKDTree(pts)
    search = calib.spacing_px * (1.0 + cfg.spacing_tolerance)
    min_d = calib.spacing_px * (1.0 - cfg.spacing_tolerance)
    max_d = calib.spacing_px * (1.0 + cfg.spacing_tolerance)
    angle_tol = math.radians(14)
    neighbors = tree.query_ball_point(pts, r=search)

    kept: list[Detection] = []
    for i, ids in enumerate(neighbors):
        forward = 0
        backward = 0
        row_support = 0
        cross_support = 0
        for j in ids:
            if i == j:
                continue
            dx, dy = pts[j] - pts[i]
            dist = math.hypot(float(dx), float(dy))
            if not (min_d <= dist <= max_d):
                continue
            theta = math.atan2(float(dy), float(dx)) % math.pi
            if angle_diff_pi(theta, row_angle) <= angle_tol:
                row_support += 1
                signed = math.cos(math.atan2(float(dy), float(dx)) - row_angle) * dist
                if signed >= 0:
                    forward += 1
                else:
                    backward += 1
            elif angle_diff_pi(theta, (row_angle + math.pi / 2.0) % math.pi) <= angle_tol:
                cross_support += 1

        support = row_support + min(cross_support, 1)
        dets[i].support = support
        # Young planted trees should sit on a coherent row. Requiring either
        # neighbors on both row directions, or one row neighbor plus one adjacent
        # row neighbor, removes most forest/pasture texture.
        if (forward >= 1 and backward >= 1) or (row_support >= 1 and cross_support >= 1 and support >= cfg.min_row_support):
            kept.append(dets[i])
    return kept


def pixel_to_world(transform: Affine, x: float, y: float) -> tuple[float, float]:
    wx, wy = transform * (x + 0.5, y + 0.5)
    return float(wx), float(wy)


def export_csv(path: Path, dets: list[Detection], transform: Affine) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pixel_x", "pixel_y", "x", "y", "score", "radius_px", "support", "component_area_px"])
        for i, d in enumerate(dets, 1):
            wx, wy = pixel_to_world(transform, d.x, d.y)
            writer.writerow([i, f"{d.x:.2f}", f"{d.y:.2f}", f"{wx:.6f}", f"{wy:.6f}", f"{d.score:.4f}", f"{d.radius_px:.2f}", d.support, f"{d.component_area_px:.1f}"])


def export_geojson(path: Path, dets: list[Detection], src: rasterio.io.DatasetReader) -> None:
    features = []
    for i, d in enumerate(dets, 1):
        wx, wy = pixel_to_world(src.transform, d.x, d.y)
        features.append({
            "type": "Feature",
            "properties": {"id": i, "score": d.score, "support": d.support, "radius_px": d.radius_px},
            "geometry": {"type": "Point", "coordinates": [wx, wy]},
        })
    crs = {"type": "name", "properties": {"name": src.crs.to_string()}} if src.crs else None
    path.write_text(json.dumps({"type": "FeatureCollection", "crs": crs, "features": features}, indent=2), encoding="utf-8")


def render_preview(src: rasterio.io.DatasetReader, dets: list[Detection], out: Path, cfg: Config) -> None:
    scale = min(1.0, cfg.preview_max_size / max(src.width, src.height))
    out_w = max(1, int(src.width * scale))
    out_h = max(1, int(src.height * scale))
    arr = src.read(list(cfg.rgb_bands), out_shape=(3, out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
    rgb = np.stack([robust_rescale(arr[i]) for i in range(3)], axis=-1)
    bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    dot_r = max(2, int(round(cfg.dot_radius_px)))
    for d in dets:
        cv2.circle(bgr, (int(round(d.x * scale)), int(round(d.y * scale))), dot_r, (0, 255, 255), -1)
        cv2.circle(bgr, (int(round(d.x * scale)), int(round(d.y * scale))), dot_r + 1, (0, 0, 0), 1)
    cv2.imwrite(str(out), bgr)


def write_points_tif(src: rasterio.io.DatasetReader, dets: list[Detection], out: Path, cfg: Config) -> None:
    profile = src.profile.copy()
    profile.update(count=3, dtype="uint8", compress="deflate", photometric="RGB", BIGTIFF="YES")
    pts = np.array([(d.x, d.y) for d in dets], dtype=np.float32)
    tree = cKDTree(pts) if len(pts) else None
    with rasterio.open(out, "w", **profile) as dst:
        for win, _ in iter_tiles(src.width, src.height, cfg.tile_size, 0):
            rgb = read_rgb(src, win, cfg)
            bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            if tree is not None:
                cx = win.col_off + win.width / 2
                cy = win.row_off + win.height / 2
                ids = tree.query_ball_point([cx, cy], r=math.hypot(win.width, win.height) / 2 + 16)
                for idx in ids:
                    x, y = pts[idx]
                    lx = int(round(x - win.col_off))
                    ly = int(round(y - win.row_off))
                    if -8 <= lx < win.width + 8 and -8 <= ly < win.height + 8:
                        cv2.circle(bgr, (lx, ly), cfg.dot_radius_px, (0, 255, 255), -1)
                        cv2.circle(bgr, (lx, ly), cfg.dot_radius_px + 1, (0, 0, 0), 1)
            dst.write(np.moveaxis(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), -1, 0), window=win)


def save_debug(src: rasterio.io.DatasetReader, out_dir: Path, cfg: Config, calib: Calibration) -> None:
    win = sample_windows(src, cfg.tile_size)[0]
    rgb = read_rgb(src, win, cfg)
    idx = vegetation_index(rgb)
    mask = crown_mask(idx, rgb, calib, cfg)
    resp = response_image(idx, mask, calib)
    cv2.imwrite(str(out_dir / "debug_vegetation_index.png"), (idx * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "debug_crown_mask.png"), (mask.astype(np.uint8) * 255))
    cv2.imwrite(str(out_dir / "debug_peak_response.png"), (resp * 255).astype(np.uint8))


def run(input_tif: Path, out_dir: Path, cfg: Config) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(input_tif) as src:
        if max(cfg.rgb_bands) > src.count:
            raise ValueError(f"RGB band selection exceeds raster band count ({src.count}).")
        calib = auto_calibrate(src, cfg)
        (out_dir / "calibration.json").write_text(json.dumps(asdict(calib), indent=2), encoding="utf-8")
        print(json.dumps(asdict(calib), indent=2))

        all_dets: list[Detection] = []
        stride = cfg.tile_size - cfg.tile_overlap
        total_tiles = math.ceil(src.width / stride) * math.ceil(src.height / stride)
        for i, (win, valid) in enumerate(iter_tiles(src.width, src.height, cfg.tile_size, cfg.tile_overlap), 1):
            all_dets.extend(detect_tile(src, win, valid, calib, cfg))
            if i == 1 or i % 25 == 0:
                print(f"Processed tile {i}/{total_tiles}; raw candidates={len(all_dets)}")

        print(f"Raw candidates before merge: {len(all_dets)}")
        merged = nms(all_dets, max(calib.min_crown_radius_px * 1.8, calib.spacing_px * cfg.nms_spacing_factor))
        print(f"After NMS: {len(merged)}")
        filtered = row_filter(merged, calib, cfg)
        print(f"After row/spacing filter: {len(filtered)}")

        export_csv(out_dir / "trees.csv", filtered, src.transform)
        export_geojson(out_dir / "trees.geojson", filtered, src)
        render_preview(src, filtered, out_dir / "detections_preview.png", cfg)
        if cfg.write_points_tif:
            write_points_tif(src, filtered, out_dir / "detections_points.tif", cfg)
        if cfg.debug:
            save_debug(src, out_dir, cfg, calib)

        summary = {"count": len(filtered), "calibration": asdict(calib)}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Final count: {len(filtered)}")
        print(f"Outputs written to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect young planted trees in RGB GeoTIFFs without ML.")
    p.add_argument("input_tif", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("young_tree_out"))
    p.add_argument("--rgb-bands", nargs=3, type=int, default=(1, 2, 3), metavar=("R", "G", "B"))
    p.add_argument("--tile-size", type=int, default=1536)
    p.add_argument("--tile-overlap", type=int, default=256)
    p.add_argument("--expected-spacing-m", type=float, default=1.5)
    p.add_argument("--spacing-tolerance", type=float, default=0.45)
    p.add_argument("--min-row-support", type=int, default=2)
    p.add_argument("--vegetation-percentile", type=float, default=78.0)
    p.add_argument("--min-score", type=float, default=0.36)
    p.add_argument("--peak-threshold-rel", type=float, default=0.28)
    p.add_argument("--min-crown-radius-m", type=float, default=None)
    p.add_argument("--max-crown-radius-m", type=float, default=None)
    p.add_argument("--dot-radius-px", type=int, default=4)
    p.add_argument("--write-points-tif", action="store_true")
    p.add_argument("--nms-spacing-factor", type=float, default=0.62)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        rgb_bands=tuple(args.rgb_bands),
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        expected_spacing_m=args.expected_spacing_m,
        spacing_tolerance=args.spacing_tolerance,
        min_row_support=args.min_row_support,
        vegetation_percentile=args.vegetation_percentile,
        min_score=args.min_score,
        peak_threshold_rel=args.peak_threshold_rel,
        min_crown_radius_m=args.min_crown_radius_m,
        max_crown_radius_m=args.max_crown_radius_m,
        dot_radius_px=args.dot_radius_px,
        write_points_tif=args.write_points_tif,
        debug=args.debug,
        nms_spacing_factor=args.nms_spacing_factor,
    )
    run(args.input_tif, args.out_dir, cfg)


if __name__ == "__main__":
    main()
