"""Fetch Copernicus GLO-30 DEM clips around bridge sites.

The Copernicus 30 m DEM is public on AWS S3, no credentials needed.
Tiles are 1x1 degree, named by the latitude/longitude of their
southwest corner, e.g. Copernicus_DSM_COG_10_S02_00_E029_00_DEM.
"""

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/"
    "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif"
)


def tile_url(lat: float, lon: float) -> str:
    """URL of the 1-degree DEM tile containing (lat, lon)."""
    lat_sw = math.floor(lat)
    lon_sw = math.floor(lon)
    lat_str = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{abs(lat_sw):02d}"
    lon_str = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{abs(lon_sw):03d}"
    return TILE_URL.format(lat=lat_str, lon=lon_str)


def fetch_clip(lat: float, lon: float, half_size_m: float = 500) -> tuple[np.ndarray, rasterio.Affine]:
    """Read a square DEM window centered on (lat, lon) straight from S3.

    Returns the elevation array (meters) and its affine transform.
    half_size_m is half the side length of the square, in meters.
    """
    # ~111,320 m per degree latitude; longitude shrinks by cos(lat)
    dlat = half_size_m / 111_320
    dlon = half_size_m / (111_320 * math.cos(math.radians(lat)))
    with rasterio.open(tile_url(lat, lon)) as src:
        window = from_bounds(
            lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform
        )
        data = src.read(1, window=window)
        transform = src.window_transform(window)
    return data, transform


def save_clip(lat: float, lon: float, name: str, out_dir: Path, half_size_m: float = 500) -> Path:
    """Fetch a clip and save it as a small GeoTIFF under out_dir."""
    data, transform = fetch_clip(lat, lon, half_size_m)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.tif"
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return out_path


def profile_through(
    data: np.ndarray,
    transform: rasterio.Affine,
    lat: float,
    lon: float,
    azimuth_deg: float,
    length_m: float = 400,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Elevation cross-section through (lat, lon) along a compass bearing.

    Returns (distance_m, elevation_m) where distance is centered on the
    site: negative on one side, positive on the other. Sweep azimuth_deg
    to find the direction that crosses the valley.
    """
    az = math.radians(azimuth_deg)
    dist = np.linspace(-length_m / 2, length_m / 2, n_points)
    dlat = dist * math.cos(az) / 111_320
    dlon = dist * math.sin(az) / (111_320 * math.cos(math.radians(lat)))
    lats = lat + dlat
    lons = lon + dlon
    inv = ~transform
    cols, rows = inv * (lons, lats)
    rows = np.clip(rows.astype(int), 0, data.shape[0] - 1)
    cols = np.clip(cols.astype(int), 0, data.shape[1] - 1)
    return dist, data[rows, cols].astype(float)


def width_at_height(
    dist: np.ndarray,
    elev: np.ndarray,
    height_above_floor: float = 20,
    search_radius_m: float = 150,
) -> tuple[float, float, float]:
    """Pit width at a fixed height above the pit floor.

    Finds the lowest point within search_radius_m of the site (x = 0),
    draws a horizontal line height_above_floor meters above it, and
    measures where the terrain first crosses that line on each side.

    Returns (width_m, left_x, right_x). Width is nan when the terrain
    never reaches the line on one side (pit shallower than the height).
    """
    near = np.abs(dist) <= search_radius_m
    floor_idx = np.where(near)[0][np.argmin(elev[near])]
    line = elev[floor_idx] + height_above_floor

    above = elev >= line
    left_side = np.where(above[:floor_idx])[0]
    right_side = np.where(above[floor_idx:])[0]
    if len(left_side) == 0 or len(right_side) == 0:
        return float("nan"), float("nan"), float("nan")
    left_x = dist[left_side[-1]]
    right_x = dist[floor_idx + right_side[0]]
    return right_x - left_x, left_x, right_x


def best_crossing(
    data: np.ndarray,
    transform: rasterio.Affine,
    lat: float,
    lon: float,
    height_above_floor: float = 20,
    length_m: float = 900,
    n_points: int = 450,
) -> dict:
    """Scan every bearing (0-179) and pick the narrowest valid crossing.

    The perpendicular crossing of the channel is the narrowest one, so
    the minimum width over all bearings is the span estimate and its
    bearing is the crossing direction. No eyeballing required.
    """
    best = {"bearing": None, "width_m": float("inf")}
    for az in range(0, 180, 5):
        dist, elev = profile_through(
            data, transform, lat, lon, az, length_m=length_m, n_points=n_points
        )
        width, left_x, right_x = width_at_height(dist, elev, height_above_floor)
        if not math.isnan(width) and width < best["width_m"]:
            best = {
                "bearing": az,
                "width_m": width,
                "left_x": left_x,
                "right_x": right_x,
                "dist": dist,
                "elev": elev,
            }
    if best["bearing"] is None:
        return {"bearing": None, "width_m": float("nan")}
    return best
