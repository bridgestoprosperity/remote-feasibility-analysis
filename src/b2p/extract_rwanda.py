"""Batch-extract terrain features for the Rwanda-box labeled sites.

Writes data/rwanda_features.csv: one row per site, terrain features
plus the label and a distance-to-nearest-built-bridge process feature.
Run: uv run python -m b2p.extract_rwanda
"""

import numpy as np
import pandas as pd

from b2p.features import site_features


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    df = pd.read_csv("data/labeled_clean.csv")
    box = df[
        df["GPS (Latitude)"].between(-3, -1)
        & df["GPS (Longitude)"].between(28, 31)
    ].copy()
    print(f"{len(box)} sites in Rwanda box")

    rows = []
    for i, r in enumerate(box.itertuples()):
        lat = getattr(r, "_5")  # GPS (Latitude)
        lon = getattr(r, "_6")  # GPS (Longitude)
        feats = site_features(lat, lon)
        feats.update(
            project_code=getattr(r, "_1"),
            lat=lat,
            lon=lon,
            label=r.label,
        )
        rows.append(feats)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(box)}")

    out = pd.DataFrame(rows)

    # Process feature: distance to nearest OTHER built bridge. Leaks
    # operational history on purpose; the with/without comparison shows
    # how much "feasibility" is really "where Fika already works".
    built = out[out.label == 1]
    dists = []
    for r in out.itertuples():
        d = haversine_km(r.lat, r.lon, built.lat.values, built.lon.values)
        d = d[d > 0]  # exclude self
        dists.append(d.min() if len(d) else np.nan)
    out["dist_nearest_built_km"] = dists

    out.to_csv("data/rwanda_features.csv", index=False)
    n_ok = out.drop(columns=["project_code"]).notna().all(axis=1).sum()
    print(f"wrote data/rwanda_features.csv: {len(out)} rows, {n_ok} fully valid")


if __name__ == "__main__":
    main()
