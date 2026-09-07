"""Sample SoilGrids 250 m layers at the Rwanda-box sites.

Reads remote cloud-optimized rasters with range requests; no full
downloads. Layers chosen for foundation/anchor relevance: clay and
sand fraction (cohesion), bulk density (bearing), and depth to
bedrock (anchor depth).
Run: uv run python -m b2p.extract_soil
"""

import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

LAYERS = {
    "clay_pct": "/vsicurl/https://files.isric.org/soilgrids/latest/data/clay/clay_0-5cm_mean.vrt",
    "sand_pct": "/vsicurl/https://files.isric.org/soilgrids/latest/data/sand/sand_0-5cm_mean.vrt",
    "bulk_density": "/vsicurl/https://files.isric.org/soilgrids/latest/data/bdod/bdod_0-5cm_mean.vrt",
    "bedrock_depth_cm": "/vsicurl/https://files.isric.org/soilgrids/former/2017-03-10/data/BDTICM_M_250m_ll.tif",
}


def main():
    df = pd.read_csv("data/rwanda_features.csv")
    lons, lats = df.lon.tolist(), df.lat.tolist()
    for name, url in LAYERS.items():
        with rasterio.open(url) as src:
            if src.crs.to_epsg() == 4326:
                coords = list(zip(lons, lats))
            else:
                xs, ys = warp_transform("EPSG:4326", src.crs, lons, lats)
                coords = list(zip(xs, ys))
            vals = [v[0] for v in src.sample(coords)]
        nodata = {None, -32768, 0} if name != "bedrock_depth_cm" else {None, -32768}
        df[name] = [float(v) if v not in nodata else float("nan") for v in vals]
        print(name, "done, valid:", df[name].notna().sum())
    df.to_csv("data/rwanda_features.csv", index=False)
    print("merged into data/rwanda_features.csv")


if __name__ == "__main__":
    main()
