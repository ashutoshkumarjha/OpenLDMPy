"""Cartographic map rendering (originally deferred, migrated later):
colored classified-raster maps with a legend, scale bar, north
arrow, and title.

Port of R's ``genrateMap`` (Rasterise_dev_68akj.r lines 2418-2456), via
matplotlib + rasterio instead of R's ``raster``/``sp``/``grid`` plotting
stack — same visual elements (discrete legend, scale bar, north arrow,
title, small credit line in the corner), redesigned around matplotlib's own
idioms rather than a literal transcription (R's own `sp` scale-bar/north-
arrow helpers have no direct Python equivalent worth reproducing
line-for-line).

``makePalette``/``makePaletteVRT``/``writePaletteVRT`` (GDAL VRT color-table
XML generation) and ``randomizeFileName`` are not ported — they were R-side
plumbing for a GDAL VRT-based rendering path this module doesn't use
(matplotlib takes the color list directly); ``copyright.draw``'s content
(a text label, optionally with a logo image next to it) is folded directly
into :func:`render_classified_map` rather than kept as a separate function,
since matplotlib doesn't need the two-step grid-graphics dance the R
version does to combine text and an image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # no GUI backend needed; the Qt layer displays the PNG we produce
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Polygon
from matplotlib.figure import Figure

from .config import logger


@dataclass
class ClassStyle:
    """One row of the GUI's class/color table: a raster class code, its
    display name, and the color to render it in (any matplotlib color
    spec — a "#RRGGBB" hex string is what the GUI's QColor swatches
    produce)."""

    class_id: int
    name: str
    color: str


def _nice_round(value: float) -> float:
    """Round a distance to the nearest "nice" 1/2/5 * 10^n number for a
    scale bar (R hard-coded a fixed 10000; this picks one proportional to
    the actual map extent instead, so it's sensible regardless of the
    raster's real-world size)."""
    if value <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(value))
    normalized = value / magnitude  # in [1, 10)
    if normalized < 1.5:
        nice = 1
    elif normalized < 3.5:
        nice = 2
    elif normalized < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def _draw_scale_bar(ax, left: float, bottom: float, right: float, top: float) -> None:
    """A simple two-segment black/white bar with 0/half/full distance
    labels, positioned in the bottom-left — same information R's
    ``scalebar(..., type='bar', divs=2, label=c('0','5','10'))`` shows."""
    width = right - left
    bar_length = _nice_round(width * 0.2)
    x0 = left + width * 0.05
    y0 = bottom + (top - bottom) * 0.05
    bar_height = (top - bottom) * 0.01

    half = bar_length / 2
    ax.add_patch(plt.Rectangle((x0, y0), half, bar_height, facecolor="black", edgecolor="black", zorder=5))
    ax.add_patch(
        plt.Rectangle((x0 + half, y0), half, bar_height, facecolor="white", edgecolor="black", zorder=5)
    )
    unit, scale = ("km", 1000.0) if bar_length >= 1000 else ("m", 1.0)
    for frac, label in ((0.0, "0"), (0.5, f"{bar_length * 0.5 / scale:g}"), (1.0, f"{bar_length / scale:g}")):
        ax.text(x0 + frac * bar_length, y0 + bar_height * 1.8, label, ha="center", va="bottom", fontsize=7, zorder=5)
    ax.text(x0 + bar_length / 2, y0 - bar_height * 3, unit, ha="center", va="top", fontsize=7, zorder=5)


def _draw_north_arrow(ax, left: float, right: float, top: float, bottom: float) -> None:
    """A filled triangle pointing up plus an "N" label, top-right corner —
    same idea as R's ``layout.north.arrow()``."""
    width = right - left
    height = top - bottom
    size = min(width, height) * 0.05
    cx = right - width * 0.08
    cy = top - height * 0.12
    triangle = Polygon(
        [(cx, cy + size), (cx - size * 0.4, cy - size * 0.5), (cx + size * 0.4, cy - size * 0.5)],
        closed=True, facecolor="black", edgecolor="black", zorder=5,
    )
    ax.add_patch(triangle)
    ax.text(cx, cy + size * 1.3, "N", ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=5)


def render_classified_map(
    raster_path: str,
    classes: Sequence[ClassStyle],
    title: str = "",
    legend_title: str = "",
    na_value: Optional[float] = None,
    credit: str = "OpenLDM",
) -> Figure:
    """Port of R's ``genrateMap``: render a single-band categorical raster
    with a discrete per-class legend, a scale bar, a north arrow, and a
    title.

    ``classes`` maps each raster class code to a display name and color —
    typically read straight off the GUI's class/color table
    (``twColorTable_ViewMaps``). Sorted internally by ``class_id`` so the
    legend and the color mapping always agree regardless of the order
    they're passed in (R's own version reorders ``className`` by sorted
    ``classNumber`` but plots with the *original*-order ``classColour`` —
    an inconsistency only harmless if the caller already passes
    pre-sorted colors; sorting both together here avoids relying on that).
    """
    ordered = sorted(classes, key=lambda c: c.class_id)
    if not ordered:
        raise ValueError("No classes given to render")

    with rasterio.open(raster_path) as src:
        band = src.read(1)
        transform = src.transform
        nodata = na_value if na_value is not None else src.nodata
        left, bottom, right, top = rasterio.transform.array_bounds(src.height, src.width, transform)

    data = band.astype(float)
    if nodata is not None:
        data = np.ma.masked_equal(data, nodata)

    class_ids = [c.class_id for c in ordered]
    colors = [c.color for c in ordered]
    names = [c.name for c in ordered]

    # One color bin per class id: boundaries at the midpoints between
    # consecutive ids, extended half a step past the first/last.
    if len(class_ids) > 1:
        mids = [(a + b) / 2 for a, b in zip(class_ids[:-1], class_ids[1:])]
        boundaries = [class_ids[0] - (mids[0] - class_ids[0])] + mids + [class_ids[-1] + (class_ids[-1] - mids[-1])]
    else:
        boundaries = [class_ids[0] - 0.5, class_ids[0] + 0.5]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(data, cmap=cmap, norm=norm, extent=(left, right, bottom, top), origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.ticklabel_format(style="plain", useOffset=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)

    handles = [Patch(facecolor=c, edgecolor="black", label=n) for c, n in zip(colors, names)]
    ax.legend(
        handles=handles, title=legend_title, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, title_fontsize=9, frameon=True,
    )

    _draw_scale_bar(ax, left, bottom, right, top)
    _draw_north_arrow(ax, left, right, top, bottom)
    fig.text(0.01, 0.01, credit, fontsize=8, color="gray")

    fig.tight_layout()
    return fig


def save_map(fig: Figure, output_path: str, dpi: int = 150) -> None:
    """Write a rendered figure to a PNG (or whatever format ``output_path``'s
    extension implies — matplotlib dispatches on it), then close it —
    callers are done with the ``Figure`` once it's on disk, and matplotlib
    figures aren't garbage-collected on their own; leaving this out would
    leak one per "Show"/"Export" click."""
    logger.info(f"Writing map image to: {output_path}")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
