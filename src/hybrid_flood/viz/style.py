"""Shared publication style, dimensions, colormaps, and dual-format saving."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Passing 'N' to ListedColormap is deprecated.*",
        category=mpl.MatplotlibDeprecationWarning,
    )
    import cmocean

SINGLE_COLUMN_WIDTH = 3.35
DOUBLE_COLUMN_WIDTH = 7.0
PRINT_DPI = 300
DEPTH_CMAP = cmocean.cm.deep
RESIDUAL_CMAP = cmocean.cm.balance
ROUGHNESS_CMAP = cmocean.cm.matter
RAIN_CMAP = cmocean.cm.rain
SLOPE_CMAP = cmocean.cm.speed
SOLVER_COLORS = {
    "anuga": "#0072B2",
    "jax": "#D55E00",
    "hybrid": "#009E73",
}


def apply_publication_style() -> None:
    """Apply a compact serif style suitable for LaTeX report figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "savefig.dpi": PRINT_DPI,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure_pair(
    figure: plt.Figure,
    output_base: str | Path,
    *,
    dpi: int = PRINT_DPI,
    close: bool = True,
) -> tuple[Path, Path]:
    """Save one figure as vector PDF and print-resolution PNG."""
    base = Path(output_base)
    if base.suffix.lower() in {".pdf", ".png"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(figure)
    return pdf_path, png_path


apply_publication_style()
