"""National feasibility surface: the scored river network over hillshade.

Input is data/network_scored.csv from score_network.py. Every plotted
value derives from public data, so this figure is shareable, unlike the
site-level maps in figures/private/.

Run: uv run python -m b2p.make_network_map
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource, LinearSegmentedColormap, TwoSlopeNorm
from rasterio.features import geometry_mask

from b2p.extract_flowacc import TMP as DEM90
from b2p.make_map import BORDER, INK, INK_2, MUTED, SEQ_CMAP, SURFACE

OUT = Path("figures/rwanda_feasibility_network.png")
OUT_DIV = Path("figures/rwanda_feasibility_network_diverging.png")
THICKEN_KM2 = 50  # rivers this large paint a 3x3 block instead of a 2x2 one
VMAX = 0.6  # 90th percentile of surface P is 0.30; values >= 0.6 saturate

# Diverging version: neutral at the cutoff, red below, blue above.
# 0.24 is the training base rate: above it, a spot rates better than the
# average candidate Fika surveys. Set to 0.5 for "more likely than not".
CUTOFF = 0.24
DIV_CMAP = LinearSegmentedColormap.from_list("div_red_blue", [
    "#b23230", "#e34948", "#f2a09b", "#f0efec",
    "#9ec5f4", "#3987e5", "#0d366b",
])


def main():
    net = pd.read_csv("data/network_scored.csv")
    rw = gpd.read_file("zip://data/ne_countries.zip")
    rw = rw[rw.NAME == "Rwanda"]

    with rasterio.open(DEM90) as src:
        dem = src.read(1)
        transform = src.transform
        inside = ~geometry_mask(
            rw.geometry.buffer(0.01), out_shape=src.shape, transform=transform
        )

    # Hillshade, compressed to a light gray band so the ramp pops
    shade = LightSource(azdeg=315, altdeg=45).hillshade(dem, vert_exag=2)
    base = 0.80 + 0.17 * shade
    img = np.repeat(base[:, :, None], 3, axis=2)
    img[~inside] = tuple(int(SURFACE[i:i + 2], 16) / 255 for i in (1, 3, 5))

    inv = ~transform
    cols, rows = inv * (net.lon.values, net.lat.values)
    rows, cols = rows.astype(int), cols.astype(int)
    big = net.upstream_km2.values >= THICKEN_KM2
    h, w = dem.shape

    def render(out, colors, sm, ticks, tick_labels, caption_line3):
        painted = img.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                sel = big if (dr < 0 or dc < 0) else slice(None)
                rr = np.clip(rows[sel] + dr, 0, h - 1)
                cc = np.clip(cols[sel] + dc, 0, w - 1)
                painted[rr, cc] = colors[sel]

        x0, y0, x1, y1 = rw.total_bounds
        m = 0.06
        fig, ax = plt.subplots(figsize=(12, 10.5), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.imshow(
            painted,
            extent=(transform.c, transform.c + w * transform.a,
                    transform.f + h * transform.e, transform.f),
            origin="upper", interpolation="nearest",
        )
        rw.boundary.plot(ax=ax, color=BORDER, linewidth=1.2)
        ax.set_xlim(x0 - m, x1 + m)
        ax.set_ylim(y0 - m, y1 + m)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("model P(feasible)", color=INK_2, fontsize=10)
        cbar.set_ticks(ticks, labels=tick_labels)
        cbar.ax.tick_params(color=BORDER, labelcolor=MUTED, labelsize=9)
        cbar.outline.set_visible(False)

        ax.set_title(
            "Trail bridge feasibility along Rwanda's river network",
            fontsize=15, color=INK, loc="left", pad=12,
        )
        fig.text(
            0.06, 0.045,
            f"Model predictions at {len(net):,} channel cells (90 m grid, at least 1 km$^2$ upstream).\n"
            "Derived entirely from public data (Copernicus DEM, SoilGrids): no survey sites shown.\n"
            f"{caption_line3} Screening aid, not an engineering assessment.",
            fontsize=9, color=MUTED, ha="left",
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {out} ({len(net):,} cells)")

    p = net.p_feasible.values
    render(
        OUT,
        SEQ_CMAP(np.clip(p / VMAX, 0, 1))[:, :3],
        plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=plt.Normalize(0, VMAX)),
        [0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "$\\geq$0.6"],
        "Wider lines mark rivers draining over 50 km$^2$.",
    )

    # Blue arm saturates at VMAX like the sequential map; nearly all
    # above-cutoff cells sit in 0.24-0.45, so an arm to 1.0 leaves them pale.
    div_norm = TwoSlopeNorm(vmin=0, vcenter=CUTOFF, vmax=VMAX)
    render(
        OUT_DIV,
        DIV_CMAP(div_norm(np.clip(p, 0, VMAX)))[:, :3],
        plt.cm.ScalarMappable(cmap=DIV_CMAP, norm=div_norm),
        [0, CUTOFF, 0.4, VMAX], ["0", f"{CUTOFF}\nbase rate", "0.4", f"$\\geq${VMAX}"],
        f"Blue: rates above the average surveyed candidate (P $\\geq$ {CUTOFF}). Red: below. Gray: near the line.",
    )


if __name__ == "__main__":
    main()
