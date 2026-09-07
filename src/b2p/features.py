"""Terrain feature extraction for labeled sites.

Reads from locally downloaded Copernicus tiles in data/dem_tiles/ so a
full-country extraction touches the network zero times.
"""

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from b2p.dem import best_crossing, profile_through

TILE_DIR = Path("data/dem_tiles")


def local_tile(lat: float, lon: float) -> Path:
    lat_sw, lon_sw = math.floor(lat), math.floor(lon)
    lat_str = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{abs(lat_sw):02d}"
    lon_str = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{abs(lon_sw):03d}"
    return TILE_DIR / f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM.tif"


def local_clip(lat: float, lon: float, half_size_m: float = 800):
    """Like dem.fetch_clip but from a local tile. Returns (array, transform)."""
    dlat = half_size_m / 111_320
    dlon = half_size_m / (111_320 * math.cos(math.radians(lat)))
    with rasterio.open(local_tile(lat, lon)) as src:
        window = from_bounds(
            lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform
        )
        data = src.read(1, window=window)
        transform = src.window_transform(window)
    return data, transform


def site_features(lat: float, lon: float) -> dict:
    """Terrain features for one site. Every value derives from the DEM.

    Sites near tile edges get whatever the window overlaps; a clip that
    comes back empty yields NaNs for the caller to drop.
    """
    data, transform = local_clip(lat, lon)
    if data.size == 0:
        return {}
    px = 30.0  # approximate pixel size in meters at this latitude
    gy, gx = np.gradient(data.astype(float), px)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    cy, cx = data.shape[0] // 2, data.shape[1] // 2
    site_elev = float(data[cy, cx])
    r = 5  # ~150 m neighborhood around the site
    local = data[max(0, cy - r):cy + r, max(0, cx - r):cx + r].astype(float)

    cross = best_crossing(data, transform, lat, lon, height_above_floor=10)
    feats = {
        "elev_m": site_elev,
        "relief_m": float(data.max() - data.min()),
        "valley_depth_m": site_elev - float(local.min()),
        "local_slope_deg": float(slope[max(0, cy - r):cy + r, max(0, cx - r):cx + r].mean()),
        "area_slope_deg": float(slope.mean()),
        "gap_width_m": cross["width_m"],
        "roughness_m": float(np.std(local)),
    }
    return feats
