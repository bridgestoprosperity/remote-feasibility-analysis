"""Private interactive map: pan/zoom explorer for the model's outputs.

Writes a single self-contained HTML file. The basemap is an embedded
hillshade rendered from the local DEM, not a tile service, so opening
the file sends no coordinates anywhere (Leaflet itself loads from a
CDN). Layers: scored river network (image), backlog sites and surveyed
sites (interactive markers with tooltips).

Output is figures/private/ (gitignored): the site layers plot raw
coordinates.

Run: uv run python -m b2p.make_explorer
"""

import base64
import io
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource
from rasterio.features import geometry_mask

from b2p.extract_flowacc import TMP as DEM90
from b2p.make_map import BLUE, ORANGE, SEQ_CMAP

OUT = Path("figures/private/feasibility_explorer.html")


def png_b64(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    rw = gpd.read_file("zip://data/ne_countries.zip")
    rw = rw[rw.NAME == "Rwanda"]

    with rasterio.open(DEM90) as src:
        dem = src.read(1)
        t = src.transform
        inside = ~geometry_mask(
            rw.geometry.buffer(0.01), out_shape=src.shape, transform=t
        )
    h, w = dem.shape
    west, north = t.c, t.f
    east, south = t.c + w * t.a, t.f + h * t.e
    bounds = [[south, west], [north, east]]

    shade = LightSource(azdeg=315, altdeg=45).hillshade(dem, vert_exag=2)
    base = 0.80 + 0.17 * shade
    hill = np.repeat(base[:, :, None], 3, axis=2)
    hill_rgba = np.dstack([hill, np.where(inside, 1.0, 0.35)])
    hill_b64 = png_b64(hill_rgba[::2, ::2])  # ~1200x1800, plenty for a basemap

    net_rgba = np.zeros((h, w, 4))
    net_path = Path("data/network_scored.csv")
    if net_path.exists():
        net = pd.read_csv(net_path)
        inv = ~t
        cols, rows = inv * (net.lon.values, net.lat.values)
        rows, cols = rows.astype(int), cols.astype(int)
        colors = SEQ_CMAP(net.p_feasible.values)
        big = net.upstream_km2.values >= 50
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                sel = big if (dr, dc) != (0, 0) else slice(None)
                rr = np.clip(rows[sel] + dr, 0, h - 1)
                cc = np.clip(cols[sel] + dc, 0, w - 1)
                net_rgba[rr, cc] = colors[sel]
    else:
        print("data/network_scored.csv missing; network layer will be empty")
    net_b64 = png_b64(net_rgba)

    backlog = pd.read_csv("data/backlog_scored.csv")
    bl = [
        [round(r.lat, 6), round(r.lon, 6), round(r.p_feasible, 3),
         round(r.upstream_km2, 1), int(r.project_code)]
        for r in backlog.itertuples()
    ]
    surveyed = pd.read_csv("data/rwanda_features.csv")
    sv = [
        [round(r.lat, 6), round(r.lon, 6), int(r.label),
         round(r.upstream_km2, 1)]
        for r in surveyed.itertuples()
    ]
    ramp = [SEQ_CMAP(x / 10) for x in range(11)]
    ramp_hex = ["#%02x%02x%02x" % tuple(int(c * 255) for c in col[:3]) for col in ramp]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fika feasibility explorer (PRIVATE)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ margin: 0; height: 100%; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  #map {{ height: 100%; background: #fcfcfb; }}
  .banner {{ position: fixed; top: 0; left: 50%; transform: translateX(-50%);
    z-index: 1000; background: #0b0b0b; color: #fff; padding: 4px 14px;
    font-size: 12px; border-radius: 0 0 6px 6px; }}
  .legend {{ position: fixed; bottom: 18px; left: 12px; z-index: 1000;
    background: #fcfcfbee; border-radius: 8px; padding: 10px 12px;
    font-size: 12px; color: #52514e; box-shadow: 0 1px 4px rgba(11,11,11,.15); }}
  .legend .swatch {{ display: inline-block; width: 12px; height: 12px;
    border-radius: 50%; margin-right: 6px; vertical-align: -1px; }}
  .legend .rampbar {{ display: flex; height: 10px; width: 140px;
    border-radius: 3px; overflow: hidden; margin: 4px 0 2px; }}
  .legend .rampbar div {{ flex: 1; }}
  .leaflet-tooltip {{ font-size: 12px; }}
</style>
</head>
<body>
<div class="banner">PRIVATE: raw site coordinates. Do not share or screenshot outside the team.</div>
<div id="map"></div>
<div class="legend">
  <div><b style="color:#0b0b0b">Model P(feasible)</b></div>
  <div class="rampbar">{"".join(f'<div style="background:{c}"></div>' for c in ramp_hex)}</div>
  <div>0 &nbsp;&middot;&nbsp; 0.5 &nbsp;&middot;&nbsp; 1</div>
  <div style="margin-top:6px"><span class="swatch" style="background:{BLUE}"></span>surveyed: feasible</div>
  <div><span class="swatch" style="background:{ORANGE}"></span>surveyed: not feasible</div>
</div>
<script>
const bounds = {json.dumps(bounds)};
const map = L.map('map', {{ preferCanvas: true, minZoom: 8, maxZoom: 15 }});
map.fitBounds(bounds);
L.imageOverlay('data:image/png;base64,{hill_b64}', bounds).addTo(map);
const network = L.imageOverlay('data:image/png;base64,{net_b64}', bounds, {{ opacity: 0.95 }}).addTo(map);

const ramp = {json.dumps(ramp_hex)};
const rampColor = p => ramp[Math.max(0, Math.min(10, Math.round(p * 10)))];

const backlog = L.layerGroup();
{json.dumps(bl)}.forEach(([lat, lon, p, up, code]) => {{
  L.circleMarker([lat, lon], {{
    radius: 6, color: '#fcfcfb', weight: 1.5, fillColor: rampColor(p), fillOpacity: 1
  }}).bindTooltip(
    `<b>Backlog ${{code}}</b><br>P(feasible): <b>${{p}}</b><br>upstream: ${{up}} km&sup2;`
  ).addTo(backlog);
}});
backlog.addTo(map);

const surveyed = L.layerGroup();
{json.dumps(sv)}.forEach(([lat, lon, label, up]) => {{
  L.circleMarker([lat, lon], {{
    radius: 4, color: '#fcfcfb', weight: 1,
    fillColor: label ? '{BLUE}' : '{ORANGE}', fillOpacity: 0.9
  }}).bindTooltip(
    `<b>${{label ? 'Feasible' : 'Not feasible'}}</b> (surveyed)<br>upstream: ${{up}} km&sup2;`
  ).addTo(surveyed);
}});

L.control.layers(null, {{
  'Scored river network': network,
  'Backlog (unsurveyed, scored)': backlog,
  'Surveyed sites (ground truth)': surveyed,
}}, {{ collapsed: false }}).addTo(map);
</script>
</body>
</html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
