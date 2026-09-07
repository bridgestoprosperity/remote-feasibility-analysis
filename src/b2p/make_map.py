"""Site-level feasibility map. PRIVATE: plots raw coordinates.

Output goes to figures/private/ (gitignored). The shareable REPORT.md
figure stays hex-aggregated; this map exists for internal inspection.
Run: uv run python -m b2p.make_map
"""

from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

OUT = Path("figures/private/feasibility_map_sites.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
BORDER = "#c3c2b7"
BLUE = "#2a78d6"    # categorical slot 1: feasible
ORANGE = "#eb6834"  # categorical slot 2: not feasible
SEQ_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)


def uganda_mask(lat: pd.Series, lon: pd.Series) -> pd.Series:
    # Same split rule as test_uganda.py
    return (
        lat.between(-1.5, 4.5) & lon.between(29, 35.5)
        & ~(lat.between(-3, -1) & lon.between(28, 31))
    )


def style_axis(ax, title, bounds):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_anchor("N")  # top-align the aspect-shrunk axes so titles line up
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)


def pad_bounds(gdf, margin=0.12):
    x0, y0, x1, y1 = gdf.total_bounds
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def main():
    countries = gpd.read_file("zip://data/ne_countries.zip")
    rwanda = countries[countries.NAME == "Rwanda"]
    uganda = countries[countries.NAME == "Uganda"]

    # The modeling set (1322 sites with complete features), not raw labels,
    # so the map's n matches the quotable metrics.
    rw = pd.read_csv("data/rwanda_features.csv")
    backlog = pd.read_csv("data/backlog_scored.csv")

    bundle = joblib.load("data/feasibility_model.joblib")
    feats = bundle["features"]
    u = pd.read_csv("data/uganda_features.csv").dropna(subset=feats + ["label"])
    u["p"] = bundle["model"].predict_proba(u[feats].values)[:, 1]
    auc = roc_auc_score(u.label, u.p)

    # Cover border-river sites that fall just outside the outline
    x0, y0, x1, y1 = rwanda.total_bounds
    lons = pd.concat([rw.lon, backlog.lon])
    lats = pd.concat([rw.lat, backlog.lat])
    rw_bounds = (
        min(x0, lons.min()) - 0.08, min(y0, lats.min()) - 0.08,
        max(x1, lons.max()) + 0.08, max(y1, lats.max()) + 0.08,
    )
    ug_bounds = pad_bounds(uganda)

    # Width ratios equal each panel's lon/lat aspect so every axes fills
    # its box at a common height and the three titles align.
    rw_aspect = (rw_bounds[2] - rw_bounds[0]) / (rw_bounds[3] - rw_bounds[1])
    ug_aspect = (ug_bounds[2] - ug_bounds[0]) / (ug_bounds[3] - ug_bounds[1])
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 5.8), width_ratios=[rw_aspect, rw_aspect, ug_aspect],
        facecolor=SURFACE, constrained_layout=True,
    )
    ax_lab, ax_bkl, ax_ug = axes

    for ax in (ax_lab, ax_bkl):
        rwanda.boundary.plot(ax=ax, color=BORDER, linewidth=1)
    uganda.boundary.plot(ax=ax_ug, color=BORDER, linewidth=1)

    style_axis(ax_lab, f"Surveyed sites: engineer verdict (n={len(rw)})", rw_bounds)
    for val, color, z in [(0, ORANGE, 2), (1, BLUE, 3)]:
        sub = rw[rw.label == val]
        ax_lab.scatter(
            sub.lon, sub.lat, s=16, c=color,
            edgecolors=SURFACE, linewidths=0.5, zorder=z,
        )
    ax_lab.legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=7, mfc=BLUE, mec=SURFACE,
                   label="feasible"),
            Line2D([], [], ls="", marker="o", ms=7, mfc=ORANGE, mec=SURFACE,
                   label="not feasible"),
        ],
        loc="lower left", frameon=False, fontsize=9, labelcolor=INK_2,
    )

    style_axis(ax_bkl, f"Unsurveyed backlog: model P(feasible) (n={len(backlog)})", rw_bounds)
    b = backlog.sort_values("p_feasible")  # high scores draw on top
    sc = ax_bkl.scatter(
        b.lon, b.lat, s=24, c=b.p_feasible, cmap=SEQ_CMAP, vmin=0, vmax=1,
        edgecolors=SURFACE, linewidths=0.5, zorder=3,
    )

    style_axis(
        ax_ug,
        f"Uganda transfer test: frozen Rwanda model (n={len(u)}, AUC {auc:.2f})",
        ug_bounds,
    )
    for val, marker in [(0, "v"), (1, "o")]:
        sub = u[u.label == val].sort_values("p")
        ax_ug.scatter(
            sub.lon, sub.lat, s=28, c=sub.p, cmap=SEQ_CMAP, vmin=0, vmax=1,
            marker=marker, edgecolors=SURFACE, linewidths=0.5, zorder=3,
        )
    ax_ug.legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=7, mfc=MUTED, mec=SURFACE,
                   label="actually feasible"),
            Line2D([], [], ls="", marker="v", ms=7, mfc=MUTED, mec=SURFACE,
                   label="actually not feasible"),
        ],
        loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2,
    )

    cbar = fig.colorbar(sc, ax=axes, shrink=0.65, pad=0.01)
    cbar.set_label("P(feasible)", color=INK_2, fontsize=9)
    cbar.ax.tick_params(color=BORDER, labelcolor=MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "Trail bridge feasibility, site level. PRIVATE: raw coordinates, do not distribute.",
        fontsize=13, color=INK, x=0.01, ha="left",
    )
    fig.text(
        0.01, -0.03,
        "Left, center: Rwanda (surveyed ground truth; backlog scored by the model). "
        "Right: Uganda labels never seen in training, scored by the frozen Rwanda model; "
        "color is the prediction, shape is the engineer verdict. "
        "Sources: Copernicus DEM, SoilGrids, Fika site data.",
        fontsize=8.5, color=MUTED, ha="left",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}  (Rwanda labeled {len(rw)}, backlog {len(backlog)}, "
          f"Uganda {len(u)}, transfer AUC {auc:.3f})")


if __name__ == "__main__":
    main()
