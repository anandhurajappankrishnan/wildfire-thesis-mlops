"""
Extract MODIS (NDVI/EVI), ERA5-Land weather, MCD64A1 burned area, and FIRMS
7-day forward fire labels for each region in config/study_areas.geojson.

Outputs: data/raw/gee_modis_era5_burnedarea.csv

Requires:
  - .env with EE_PROJECT_ID
  - earthengine authenticate
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import ee
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from env_setup import load_project_env  # noqa: E402


def load_regions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    regions = []
    for feat in payload.get("features", []):
        props = feat.get("properties", {})
        name = props.get("name") or props.get("label") or "unknown"
        label = props.get("label") or name.replace("_", " ").title()
        regions.append(
            {
                "country": name,
                "label": label,
                "geometry": ee.Geometry(feat["geometry"]),
            }
        )
    if not regions:
        raise ValueError(f"No features found in {path}")
    return regions


def build_modis_veg(d: ee.Date, aoi: ee.Geometry) -> ee.Image:
    start = d.advance(-16, "day")
    end = d.advance(16, "day")
    img = (
        ee.ImageCollection("MODIS/061/MOD13A1")
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(["NDVI", "EVI"])
        .median()
    )
    scaled = img.multiply(0.0001).rename(["ndvi", "evi"])
    return scaled.clip(aoi)


def build_era5_daily(d: ee.Date, aoi: ee.Geometry) -> ee.Image:
    d0 = d
    d1 = d.advance(1, "day")
    first = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(d0, d1).first()
    temp_k = first.select("temperature_2m")
    temp_c = temp_k.subtract(273.15).rename("temperature_2m")
    precip = first.select("total_precipitation_sum").rename("total_precipitation")
    dew_k = first.select("dewpoint_temperature_2m")
    dew_c = dew_k.subtract(273.15).rename("dewpoint_temperature_2m")
    return ee.Image.cat([temp_c, precip, dew_c]).clip(aoi)


def build_firms_forward_horizon(d: ee.Date, aoi: ee.Geometry, horizon_days: int) -> ee.Image:
    """1 where FIRMS detects active fire within horizon_days after obs_date."""
    start = d.advance(1, "day")
    end = d.advance(horizon_days + 1, "day")
    firms = ee.ImageCollection("FIRMS").filterDate(start, end).filterBounds(aoi)
    fire_img = (
        firms.select("T21")
        .max()
        .gt(325)
        .focal_max(radius=3000, units="meters")
        .rename("firms_fire_7d")
        .unmask(0)
        .clip(aoi)
        .float()
    )
    return fire_img


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
    has_fire = burn.select("BurnDate").gt(0).rename("burned_area")
    return has_fire.clip(aoi).float()


def sample_one_date(
    obs_date: date,
    aoi: ee.Geometry,
    points_fc: ee.FeatureCollection,
    scale_m: int,
    tile_scale: int,
    country: str,
    region_label: str,
    horizon_days: int,
) -> list[dict]:
    d = ee.Date(obs_date.isoformat())
    veg = build_modis_veg(d, aoi)
    met = build_era5_daily(d, aoi)
    burn = build_burn_monthly(d, aoi)
    firms = build_firms_forward_horizon(d, aoi, horizon_days)
    combined = ee.Image.cat([veg, met, burn, firms])
    reduced = combined.reduceRegions(
        collection=points_fc,
        reducer=ee.Reducer.mean(),
        scale=scale_m,
        tileScale=tile_scale,
    )
    info = reduced.getInfo()
    rows: list[dict] = []
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        lon, lat = coords[0], coords[1]
        rows.append(
            {
                "obs_date": obs_date.isoformat(),
                "country": country,
                "region_label": region_label,
                "latitude": lat,
                "longitude": lon,
                "ndvi": props.get("ndvi"),
                "evi": props.get("evi"),
                "temperature_2m": props.get("temperature_2m"),
                "total_precipitation": props.get("total_precipitation"),
                "dewpoint_temperature_2m": props.get("dewpoint_temperature_2m"),
                "burned_area": props.get("burned_area"),
                "firms_fire_7d": props.get("firms_fire_7d"),
                "source_system": "GEE",
            }
        )
    return rows


def main() -> None:
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
    start = datetime.strptime(cfg["data"]["start_date"], "%Y-%m-%d").date()
    end = datetime.strptime(cfg["data"]["end_date"], "%Y-%m-%d").date()
    stride = int(gee_cfg.get("temporal_stride_days", 7))
    n_points = int(gee_cfg.get("grid_points_per_region", gee_cfg.get("grid_points", 150)))
    max_dates = gee_cfg.get("max_dates")
    tile_scale = int(gee_cfg.get("tile_scale", 4))
    scale_m = int(cfg["data"]["gee_scale_meters"])
    seed = int(cfg["project"]["seed"])
    horizon_days = int(cfg["data"].get("prediction_horizon_days", 7))

    date_index = pd.date_range(start, end, freq=f"{stride}D")
    dates = [d.date() for d in date_index]
    if max_dates is not None:
        dates = dates[: int(max_dates)]

    all_rows: list[dict] = []
    total_jobs = len(regions) * len(dates)
    job = 0

    for region in regions:
        country = region["country"]
        region_seed = seed + abs(hash(country)) % 10000
        points_fc = ee.FeatureCollection.randomPoints(region["geometry"], n_points, region_seed)
        print(f"Region: {region['label']} ({n_points} points, {len(dates)} dates)")

        for i, d in enumerate(dates):
            job += 1
            if job % 10 == 0 or job == 1:
                print(f"  GEE {job}/{total_jobs}: {region['label']} @ {d}")
            try:
                all_rows.extend(
                    sample_one_date(
                        d,
                        region["geometry"],
                        points_fc,
                        scale_m,
                        tile_scale,
                        country,
                        region["label"],
                        horizon_days,
                    )
                )
            except Exception as exc:
                print(f"  Warning: {region['label']} @ {d} failed: {exc}")

    df = pd.DataFrame(all_rows)
    out = raw_dir / "gee_modis_era5_burnedarea.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows across {len(regions)} regions -> {out}")
    if "burned_area" in df.columns:
        fires = (pd.to_numeric(df["burned_area"], errors="coerce").fillna(0) > 0.5).sum()
        print(f"  MCD64 burn pixels in raw extract: {fires}")
    if "firms_fire_7d" in df.columns:
        firms = (pd.to_numeric(df["firms_fire_7d"], errors="coerce").fillna(0) > 0.5).sum()
        print(f"  FIRMS 7-day forward fire labels: {firms}")


if __name__ == "__main__":
    main()
