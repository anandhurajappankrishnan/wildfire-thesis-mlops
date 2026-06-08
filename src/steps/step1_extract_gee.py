"""
Extract MODIS (NDVI/EVI), ERA5-Land weather, MCD64A1 burned area, and FIRMS
7-day forward fire labels for each region in config/study_areas.geojson.

Unit of analysis: persistent 0.1 deg x 0.1 deg areal cells (not point samples).
Each cell x date aggregates layer means; label = 1 if any forward FIRMS fire in cell.

Cell placement uses leak-free historical MCD64A1 burn climatology (prior to study window).

Outputs: data/raw/gee_modis_era5_burnedarea.csv (or --output-csv override)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from env_setup import load_project_env  # noqa: E402
from ml_eval import stable_int_hash  # noqa: E402


def load_regions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    regions = []
    for feat in payload.get("features", []):
        props = feat.get("properties", {})
        name = props.get("name") or props.get("label") or "unknown"
        label = props.get("label") or name.replace("_", " ").title()
        geom = feat["geometry"]
        west, south, east, north = _bbox_from_polygon(geom)
        regions.append(
            {
                "country": name,
                "label": label,
                "geometry": ee.Geometry(geom),
                "bbox": (west, south, east, north),
            }
        )
    if not regions:
        raise ValueError(f"No features found in {path}")
    return regions


def _bbox_from_polygon(geom: dict) -> tuple[float, float, float, float]:
    ring = geom["coordinates"][0]
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    return min(lons), min(lats), max(lons), max(lats)


def make_cell_id(lat_center: float, lon_center: float, cell_size: float) -> str:
    lat_snapped = round(round(lat_center / cell_size) * cell_size, 10)
    lon_snapped = round(round(lon_center / cell_size) * cell_size, 10)
    return f"{lat_snapped:.1f}_{lon_snapped:.1f}"


def generate_cell_catalog(
    west: float, south: float, east: float, north: float, cell_size: float
) -> list[dict]:
    """All 0.1 deg cell centers covering the region bounding box."""
    half = cell_size / 2.0
    cells: list[dict] = []
    lat = south + half
    while lat < north - 1e-9:
        lon = west + half
        while lon < east - 1e-9:
            cid = make_cell_id(lat, lon, cell_size)
            cells.append(
                {
                    "cell_id": cid,
                    "lat_center": round(lat, 6),
                    "lon_center": round(lon, 6),
                }
            )
            lon += cell_size
        lat += cell_size
    return cells


def cells_to_feature_collection(cells: list[dict], cell_size: float) -> ee.FeatureCollection:
    half = cell_size / 2.0
    features = []
    for c in cells:
        lat, lon = c["lat_center"], c["lon_center"]
        rect = ee.Geometry.Rectangle([lon - half, lat - half, lon + half, lat + half])
        features.append(
            ee.Feature(
                rect,
                {
                    "cell_id": c["cell_id"],
                    "lat_center": lat,
                    "lon_center": lon,
                    "sample_type": c.get("sample_type", "unknown"),
                },
            )
        )
    return ee.FeatureCollection(features)


def build_modis_veg(d: ee.Date, aoi: ee.Geometry) -> ee.Image:
    start = d.advance(-32, "day")
    end = d.advance(1, "day")
    img = (
        ee.ImageCollection("MODIS/061/MOD13A1")
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(["NDVI", "EVI"])
        .median()
    )
    return img.multiply(0.0001).rename(["ndvi", "evi"]).clip(aoi)


def build_era5_daily(d: ee.Date, aoi: ee.Geometry) -> ee.Image:
    d0 = d
    d1 = d.advance(1, "day")
    first = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(d0, d1).first()
    temp_c = first.select("temperature_2m").subtract(273.15).rename("temperature_2m")
    precip = first.select("total_precipitation_sum").rename("total_precipitation")
    dew_c = first.select("dewpoint_temperature_2m").subtract(273.15).rename("dewpoint_temperature_2m")
    u = first.select("u_component_of_wind_10m")
    v = first.select("v_component_of_wind_10m")
    wind_speed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed")
    return ee.Image.cat([temp_c, precip, dew_c, wind_speed]).clip(aoi)


def build_cell_topography(aoi: ee.Geometry) -> ee.Image:
    """Static DEM-derived elevation and slope (time-invariant per cell)."""
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi)
    slope = ee.Terrain.slope(dem).rename("slope")
    return dem.rename("elevation").addBands(slope)


def reduce_cell_topography(
    cells_fc: ee.FeatureCollection,
    aoi: ee.Geometry,
    scale_m: int,
    tile_scale: int,
) -> dict[str, dict[str, float]]:
    topo = build_cell_topography(aoi)
    reduced = topo.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.mean(),
        scale=scale_m,
        tileScale=tile_scale,
    )
    out: dict[str, dict[str, float]] = {}
    for feat in reduced.getInfo().get("features", []):
        props = feat.get("properties", {})
        cid = props.get("cell_id")
        if cid:
            out[cid] = {
                "elevation": props.get("elevation"),
                "slope": props.get("slope"),
            }
    return out


def build_firms_forward_horizon(d: ee.Date, aoi: ee.Geometry, horizon_days: int) -> ee.Image:
    start = d.advance(1, "day")
    end = d.advance(horizon_days + 1, "day")
    firms = ee.ImageCollection("FIRMS").filterDate(start, end).filterBounds(aoi)
    return (
        firms.select("T21")
        .max()
        .gt(325)
        .focal_max(radius=3000, units="meters")
        .rename("firms_fire_7d")
        .unmask(0)
        .clip(aoi)
        .float()
    )


def build_burn_monthly(d: ee.Date, aoi: ee.Geometry) -> ee.Image:
    month_start = d.advance(0, "month")
    month_end = month_start.advance(1, "month")
    burn = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(month_start, month_end)
        .filterBounds(aoi)
        .select(["BurnDate"])
        .first()
    )
    return burn.select("BurnDate").gt(0).rename("burned_area").clip(aoi).float()


def build_historical_burn_count_image(aoi: ee.Geometry, study_start: date, prior_years: int) -> ee.Image:
    """
    Leakage-safe cell-selection climatology from MCD64A1 BurnDate ONLY.

    Uses burn history strictly BEFORE data.start_date. Never uses FIRMS forward labels
    or any observation from within the study / prediction window.
    """
    prior_end = ee.Date(study_start.isoformat())
    prior_start = prior_end.advance(-int(prior_years), "year")
    burns = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(prior_start, prior_end)
        .filterBounds(aoi)
        .select("BurnDate")
    )
    return burns.map(lambda img: img.gt(0)).sum().rename("burn_count").clip(aoi)


def score_cells_historical_fire(
    cells_fc: ee.FeatureCollection,
    aoi: ee.Geometry,
    study_start: date,
    prior_years: int,
    scale_m: int,
    tile_scale: int,
) -> dict[str, float]:
    burn_img = build_historical_burn_count_image(aoi, study_start, prior_years)
    reduced = burn_img.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.max().setOutputs(["burn_count"]),
        scale=scale_m,
        tileScale=tile_scale,
    )
    scores: dict[str, float] = {}
    for feat in reduced.getInfo().get("features", []):
        props = feat.get("properties", {})
        cid = props.get("cell_id")
        if cid:
            scores[cid] = float(props.get("burn_count") or props.get("max") or 0)
    return scores


def select_fire_aware_cells(
    catalog: list[dict],
    burn_scores: dict[str, float],
    n_cells: int,
    fire_weighted_fraction: float,
    seed: int,
) -> list[dict]:
    """Pick persistent cells: fire_weighted_fraction from high prior-burn cells, rest background."""
    rng = np.random.RandomState(seed)
    n_fire = max(1, int(round(n_cells * fire_weighted_fraction)))
    n_bg = max(0, n_cells - n_fire)

    ranked = sorted(catalog, key=lambda c: burn_scores.get(c["cell_id"], 0.0), reverse=True)
    fire_pool = [c for c in ranked if burn_scores.get(c["cell_id"], 0.0) > 0]
    if len(fire_pool) >= n_fire:
        fire_selected = fire_pool[:n_fire]
    else:
        fire_selected = ranked[:n_fire]

    fire_ids = {c["cell_id"] for c in fire_selected}
    bg_pool = [c for c in catalog if c["cell_id"] not in fire_ids]
    if len(bg_pool) >= n_bg:
        idx = rng.choice(len(bg_pool), size=n_bg, replace=False)
        bg_selected = [bg_pool[i] for i in idx]
    else:
        bg_selected = bg_pool

    out = []
    for c in fire_selected:
        out.append({**c, "sample_type": "fire_prone"})
    for c in bg_selected:
        out.append({**c, "sample_type": "background"})
    return out


def stable_region_seed(country: str, base_seed: int) -> int:
    return (base_seed + stable_int_hash(country, 10_000)) % (2**31 - 1)


def aggregate_cells_one_date(
    obs_date: date,
    aoi: ee.Geometry,
    cells_fc: ee.FeatureCollection,
    scale_m: int,
    tile_scale: int,
    country: str,
    region_label: str,
    horizon_days: int,
    topo_by_id: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    d = ee.Date(obs_date.isoformat())
    veg = build_modis_veg(d, aoi)
    met = build_era5_daily(d, aoi)
    burn = build_burn_monthly(d, aoi)
    firms = build_firms_forward_horizon(d, aoi, horizon_days)

    mean_img = ee.Image.cat([veg, met, burn])
    mean_reduced = mean_img.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.mean(),
        scale=scale_m,
        tileScale=tile_scale,
    )
    max_reduced = firms.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.max().setOutputs(["firms_fire_7d"]),
        scale=scale_m,
        tileScale=tile_scale,
    )

    mean_by_id: dict[str, dict] = {}
    for feat in mean_reduced.getInfo().get("features", []):
        props = feat.get("properties", {})
        cid = props.get("cell_id")
        if cid:
            mean_by_id[cid] = props

    rows: list[dict] = []
    for feat in max_reduced.getInfo().get("features", []):
        props = feat.get("properties", {})
        cid = props.get("cell_id")
        if not cid:
            continue
        m = mean_by_id.get(cid, {})
        topo = (topo_by_id or {}).get(cid, {})
        rows.append(
            {
                "obs_date": obs_date.isoformat(),
                "country": country,
                "region_label": region_label,
                "cell_id": cid,
                "latitude": props.get("lat_center"),
                "longitude": props.get("lon_center"),
                "ndvi": m.get("ndvi"),
                "evi": m.get("evi"),
                "temperature_2m": m.get("temperature_2m"),
                "total_precipitation": m.get("total_precipitation"),
                "dewpoint_temperature_2m": m.get("dewpoint_temperature_2m"),
                "wind_speed": m.get("wind_speed"),
                "elevation": topo.get("elevation"),
                "slope": topo.get("slope"),
                "burned_area": m.get("burned_area"),
                "firms_fire_7d": props.get("firms_fire_7d") or props.get("max"),
                "sample_type": props.get("sample_type", "unknown"),
                "source_system": "GEE",
            }
        )
    return rows


def resolve_cells_per_region(gee_cfg: dict, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    return int(
        gee_cfg.get("cells_per_region")
        or gee_cfg.get("grid_points_per_region")
        or gee_cfg.get("grid_points", 300)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEE areal-cell extraction (fire-aware persistent grid)")
    p.add_argument("--output-csv", type=str, default=None)
    p.add_argument("--grid-points", type=int, default=None, help="Alias: cells_per_region")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--max-dates", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_project_env()
    project_id = os.getenv("EE_PROJECT_ID")
    if not project_id:
        raise SystemExit("Set EE_PROJECT_ID in .env (copy from .env.example).")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    gee_cfg = cfg["data"].get("gee", {})
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    ee.Initialize(project=project_id)
    regions = load_regions(ROOT / cfg["data"]["aoi_geojson_path"])

    study_start = datetime.strptime(cfg["data"]["start_date"], "%Y-%m-%d").date()
    extract_start = datetime.strptime(args.start_date or cfg["data"]["start_date"], "%Y-%m-%d").date()
    extract_end = datetime.strptime(args.end_date or cfg["data"]["end_date"], "%Y-%m-%d").date()

    cell_size = float(gee_cfg.get("cell_size_deg", 0.1))
    stride = int(gee_cfg.get("temporal_stride_days", 7))
    n_cells = resolve_cells_per_region(gee_cfg, args.grid_points)
    fire_frac = float(gee_cfg.get("fire_weighted_fraction", 0.7))
    prior_years = int(gee_cfg.get("fire_prior_years", 4))
    max_dates = args.max_dates if args.max_dates is not None else gee_cfg.get("max_dates")
    tile_scale = int(gee_cfg.get("tile_scale", 4))
    scale_m = int(cfg["data"]["gee_scale_meters"])
    seed = int(cfg["project"]["seed"])
    horizon_days = int(cfg["data"].get("prediction_horizon_days", 7))

    date_index = pd.date_range(extract_start, extract_end, freq=f"{stride}D")
    dates = [d.date() for d in date_index]
    if max_dates is not None:
        dates = dates[: int(max_dates)]

    n_fire = max(1, int(round(n_cells * fire_frac)))
    n_bg = max(0, n_cells - n_fire)
    print(
        f"Areal cells ({cell_size} deg): {n_fire} fire-prone + {n_bg} background "
        f"per region x {len(dates)} dates"
    )
    print(
        f"  Cell selection: MCD64A1 {prior_years}y prior to {study_start} only "
        f"(no FIRMS / no in-window leakage)"
    )

    all_rows: list[dict] = []
    total_jobs = len(regions) * len(dates)
    job = 0

    for region in regions:
        country = region["country"]
        region_seed = stable_region_seed(country, seed)
        west, south, east, north = region["bbox"]
        catalog = generate_cell_catalog(west, south, east, north, cell_size)
        print(f"Region: {region['label']} — {len(catalog)} candidate cells in bbox")

        catalog_fc = cells_to_feature_collection(catalog, cell_size)
        burn_scores = score_cells_historical_fire(
            catalog_fc, region["geometry"], study_start, prior_years, scale_m, tile_scale
        )
        selected = select_fire_aware_cells(catalog, burn_scores, n_cells, fire_frac, region_seed)
        cells_fc = cells_to_feature_collection(selected, cell_size)
        topo_by_id = reduce_cell_topography(cells_fc, region["geometry"], scale_m, tile_scale)
        print(
            f"  Selected {len(selected)} persistent cells "
            f"(seed={region_seed}, fire-prone={sum(1 for c in selected if c['sample_type']=='fire_prone')})"
        )

        for d in dates:
            job += 1
            if job % 10 == 0 or job == 1:
                print(f"  GEE {job}/{total_jobs}: {region['label']} @ {d}")
            try:
                all_rows.extend(
                    aggregate_cells_one_date(
                        d,
                        region["geometry"],
                        cells_fc,
                        scale_m,
                        tile_scale,
                        country,
                        region["label"],
                        horizon_days,
                        topo_by_id,
                    )
                )
            except Exception as exc:
                print(f"  Warning: {region['label']} @ {d} failed: {exc}")

    df = pd.DataFrame(all_rows)
    out = Path(args.output_csv) if args.output_csv else raw_dir / "gee_modis_era5_burnedarea.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows across {len(regions)} regions -> {out}")

    if len(df) and "cell_id" in df.columns:
        cell_obs = df.groupby(["country", "cell_id"]).size()
        print(
            f"  Unique cells: {len(cell_obs)} | obs/cell min={cell_obs.min()} "
            f"mean={cell_obs.mean():.1f} max={cell_obs.max()}"
        )

    if "sample_type" in df.columns and "firms_fire_7d" in df.columns:
        pos = pd.to_numeric(df["firms_fire_7d"], errors="coerce").fillna(0) > 0.5
        by_type = df.assign(_pos=pos).groupby("sample_type")["_pos"].agg(["count", "sum", "mean"])
        print(f"  FIRMS cell positives by sample_type:\n{by_type.to_string()}")

    if "firms_fire_7d" in df.columns:
        firms = (pd.to_numeric(df["firms_fire_7d"], errors="coerce").fillna(0) > 0.5).sum()
        print(f"  FIRMS 7-day forward fire labels: {firms} ({100 * firms / max(len(df), 1):.2f}% positive)")


if __name__ == "__main__":
    main()
