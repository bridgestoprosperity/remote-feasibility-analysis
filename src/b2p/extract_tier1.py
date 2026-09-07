"""Tier 1 hydrology features: HAND and channel slope, Rwanda box.

HAND (height above nearest drainage): the site's elevation minus the
elevation of the channel cell its water drains to. Low HAND = flood
exposed banks, directly related to freeboard requirements.

Channel slope: elevation drop per meter along the channel through the
site's snapped channel cell, over roughly 10 cells (~900 m) of flow
path. Steep = fast water and scour; flat = meandering flood plain.

Both derive from the existing 90 m mosaic. Run after extract_flowacc.
Run: uv run python -m b2p.extract_tier1
"""

import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import pandas as pd
from pysheds.grid import Grid

from b2p.extract_flowacc import TMP as DEM90

CHANNEL_MIN_CELLS = 25  # ~0.2 km^2 upstream at 90 m: a modest stream
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
OFFSETS = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1),
           4: (1, 0), 8: (1, -1), 16: (1, -1), 32: (0, -1)}
OFFSETS[8] = (1, -1)
OFFSETS[16] = (0, -1)
OFFSETS[32] = (-1, -1)


def main():
    grid = Grid.from_raster(DEM90)
    dem = grid.read_raster(DEM90)
    filled = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(filled)
    acc = grid.accumulation(fdir)
    channels = acc > CHANNEL_MIN_CELLS  # stays a pysheds Raster
    hand = grid.compute_hand(fdir, filled, channels, dirmap=DIRMAP)
    print("HAND grid done")

    dem_a = np.asarray(filled)
    fdir_a = np.asarray(fdir)
    acc_a = np.asarray(acc)
    inv = ~grid.affine

    def snap(lat, lon):
        col, row = inv * (lon, lat)
        r, c = int(row), int(col)
        if not (0 <= r < acc_a.shape[0] and 0 <= c < acc_a.shape[1]):
            return None
        window = [(rr, cc)
                  for rr in range(max(0, r - 1), min(acc_a.shape[0], r + 2))
                  for cc in range(max(0, c - 1), min(acc_a.shape[1], c + 2))]
        return max(window, key=lambda rc: acc_a[rc])

    def walk_down(r, c, steps):
        path = [(r, c)]
        for _ in range(steps):
            d = int(fdir_a[r, c])
            if d not in OFFSETS:
                break
            dr, dc = OFFSETS[d]
            r, c = r + dr, c + dc
            if not (0 <= r < dem_a.shape[0] and 0 <= c < dem_a.shape[1]):
                break
            path.append((r, c))
        return path

    def walk_up(r, c, steps):
        # follow the highest-accumulation upstream neighbor each step
        path = [(r, c)]
        for _ in range(steps):
            best = None
            for d, (dr, dc) in OFFSETS.items():
                ur, uc = r - dr, c - dc
                if not (0 <= ur < dem_a.shape[0] and 0 <= uc < dem_a.shape[1]):
                    continue
                if int(fdir_a[ur, uc]) == d:
                    if best is None or acc_a[ur, uc] > acc_a[best]:
                        best = (ur, uc)
            if best is None:
                break
            r, c = best
            path.append((r, c))
        return path

    df = pd.read_csv("data/rwanda_features.csv")
    hand_a = np.asarray(hand)
    hands, slopes = [], []
    for lat, lon in zip(df.lat, df.lon):
        rc = snap(lat, lon)
        if rc is None:
            hands.append(np.nan)
            slopes.append(np.nan)
            continue
        col, row = inv * (lon, lat)
        r0, c0 = int(row), int(col)
        h = hand_a[r0, c0]
        hands.append(float(h) if np.isfinite(h) else np.nan)
        down = walk_down(*rc, 5)
        up = walk_up(*rc, 5)
        top, bot = up[-1], down[-1]
        dist = 90.0 * (len(up) + len(down) - 2)
        if dist > 0:
            drop = dem_a[top] - dem_a[bot]
            slopes.append(max(float(drop), 0.0) / dist)
        else:
            slopes.append(np.nan)
    df["hand_m"] = hands
    df["channel_slope"] = slopes
    df.to_csv("data/rwanda_features.csv", index=False)
    print("hand_m valid:", df.hand_m.notna().sum(),
          "| channel_slope valid:", df.channel_slope.notna().sum())
    print(df.groupby("label")[["hand_m", "channel_slope"]].mean().round(4))


if __name__ == "__main__":
    main()
