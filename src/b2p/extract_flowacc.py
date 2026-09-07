"""DEM-derived flow accumulation for the Rwanda box.

Mosaics the six local tiles, downsamples to ~90 m (memory and speed;
channel-finding survives coarsening far better than span measurement
did), runs pit-fill + D8 flow direction + accumulation with pysheds,
then samples upstream area at each site. Snaps to the max within a
3x3 window because GPS points rarely sit exactly on the channel cell.
Run: uv run python -m b2p.extract_flowacc
"""

import numpy as np

# pysheds 1.x still calls np.in1d, removed in NumPy 2.x.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import pandas as pd
import rasterio
from pysheds.grid import Grid
from rasterio.enums import Resampling
from rasterio.merge import merge

from b2p.features import TILE_DIR

COARSE = 3  # 30 m -> 90 m
TMP = "data/rwanda_dem_90m.tif"


def build_mosaic():
    srcs = [rasterio.open(p) for p in sorted(TILE_DIR.glob("*.tif"))]
    data, transform = merge(srcs)
    h, w = data.shape[1] // COARSE, data.shape[2] // COARSE
    out = np.empty((1, h, w), dtype="float32")
    with rasterio.open(
        "/vsimem/full.tif", "w", driver="GTiff", height=data.shape[1],
        width=data.shape[2], count=1, dtype="float32", crs=srcs[0].crs,
        transform=transform,
    ) as tmp:
        tmp.write(data[0].astype("float32"), 1)
    with rasterio.open("/vsimem/full.tif") as tmp:
        out = tmp.read(
            out_shape=(1, h, w), resampling=Resampling.average
        )
        coarse_transform = tmp.transform * tmp.transform.scale(
            tmp.width / w, tmp.height / h
        )
    with rasterio.open(
        TMP, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs=srcs[0].crs, transform=coarse_transform,
        nodata=-9999,
    ) as dst:
        dst.write(out[0], 1)
    for s in srcs:
        s.close()
    print("mosaic:", out.shape)


def main():
    build_mosaic()
    grid = Grid.from_raster(TMP)
    dem = grid.read_raster(TMP)
    filled = grid.fill_depressions(grid.fill_pits(dem))
    inflated = grid.resolve_flats(filled)
    fdir = grid.flowdir(inflated)
    acc = grid.accumulation(fdir)
    print("flow accumulation done, max cells:", int(np.nanmax(acc)))

    df = pd.read_csv("data/rwanda_features.csv")
    cell_km2 = (90 * 90) / 1e6
    inv = ~grid.affine
    vals = []
    for lat, lon in zip(df.lat, df.lon):
        col, row = inv * (lon, lat)
        r, c = int(row), int(col)
        if 0 <= r < acc.shape[0] and 0 <= c < acc.shape[1]:
            win = acc[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            vals.append(float(np.nanmax(win)) * cell_km2)
        else:
            vals.append(float("nan"))
    df["upstream_km2"] = vals
    df.to_csv("data/rwanda_features.csv", index=False)
    print("merged upstream_km2, valid:", df.upstream_km2.notna().sum())


if __name__ == "__main__":
    main()
