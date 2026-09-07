"""Sample Fika's WaterNet water-probability raster at site locations.

WaterNet (CC-BY 4.0, cite Fika/B2P) publishes 20 m water-probability
COGs on Source Cooperative, tiled as zoom-6 XYZ map tiles named
{x}_{y}.tif. Values are 0-255 probability of a waterway.

Feature: waternet_prob = max value in a ~5x5 window (~100 m) around
the site, using the same snap-to-channel logic as flow accumulation:
GPS jitter should not zero out a site next to a detected stream.
Run: uv run python -m b2p.extract_waternet [csv ...]
"""

import math
import sys

import numpy as np
import pandas as pd
import rasterio

URL = "/vsicurl/https://data.source.coop/fika/waternet/raster/{x}_{y}.tif"
Z = 6


def tile_xy(lat: float, lon: float) -> tuple[int, int]:
    n = 2 ** Z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def sample(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    df = df.assign(_tile=[tile_xy(la, lo) for la, lo in zip(df.lat, df.lon)])
    for (x, y), grp in df.groupby("_tile"):
        try:
            src = rasterio.open(URL.format(x=x, y=y))
        except rasterio.errors.RasterioIOError:
            print(f"tile {x}_{y}: missing, {len(grp)} sites skipped")
            continue
        with src:
            inv = ~src.transform
            data = None  # windowed reads per site keep transfers small
            for idx, la, lo in zip(grp.index, grp.lat, grp.lon):
                cx, cy = inv * (lo, la)
                c, r = int(cx), int(cy)
                win = rasterio.windows.Window(c - 2, r - 2, 5, 5)
                try:
                    arr = src.read(1, window=win, boundless=True, fill_value=0)
                except Exception:
                    continue
                out[idx] = float(arr.max())
        print(f"tile {x}_{y}: {len(grp)} sites sampled")
    return out


def main():
    paths = sys.argv[1:] or ["data/rwanda_features.csv", "data/backlog_scored.csv"]
    for path in paths:
        df = pd.read_csv(path)
        df["waternet_prob"] = sample(df)
        df.to_csv(path, index=False)
        print(f"{path}: valid {df.waternet_prob.notna().sum()}/{len(df)}")
        if "label" in df:
            print(df.groupby("label")["waternet_prob"].mean().round(1))


if __name__ == "__main__":
    main()
