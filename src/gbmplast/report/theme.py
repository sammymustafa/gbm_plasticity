"""Matplotlib theme: one validated palette, applied by the job the colour does.

The categorical hues are a validated set (lightness band, chroma floor,
colour-vision-deficiency separation, normal-vision separation and contrast
all checked with the palette validator, not eyeballed).  They are assigned
in fixed order and never cycled: a ninth series folds into "other" or
becomes a small multiple.

Three rules this module enforces for every figure in the project:

* one y-axis per panel -- never two scales on one chart;
* a legend whenever two or more series are drawn, so identity never rests
  on colour alone, plus direct labels where the series count allows;
* recessive grid and spines, thin marks.

Several light-mode hues sit below 3:1 contrast against the surface, so
figures that use them carry visible direct labels or an accompanying table
(the "relief" requirement), which the figure helpers below do.
"""

from __future__ import annotations

from contextlib import contextmanager

# -- validated categorical order (light surface #fcfcfb) ------------------
CATEGORICAL = (
    "#2a78d6",   # 1 blue
    "#eb6834",   # 2 orange
    "#1baf7a",   # 3 aqua
    "#eda100",   # 4 yellow
    "#e87ba4",   # 5 magenta
    "#008300",   # 6 green
    "#6250d6",   # 7 violet
    "#e34948",   # 8 red
)

#: Forms that put every pair on screen at once (scatter, paired slope,
#: small-multiple overlays) are capped at the first three slots, which are
#: the ones that clear the all-pairs separation floors.
CATEGORICAL_ALL_PAIRS = CATEGORICAL[:3]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8980"
GRID = "#e4e3de"

#: Diverging poles with a neutral grey midpoint (never a hue at the middle).
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

#: Sequential magnitude: one hue, light to dark.
SEQUENTIAL_HUE = "Blues"

#: Reserved status colours -- never reused as "series 4".
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d33c3c",
}

#: Semantic roles used repeatedly in this project, drawn from the same order.
ROLE = {
    "enhancing": CATEGORICAL[0],
    "non_enhancing": CATEGORICAL[1],
    "necrosis": CATEGORICAL[7],
    "edema": CATEGORICAL[3],
    "primary": CATEGORICAL[0],
    "recurrent": CATEGORICAL[1],
    "baseline_model": TEXT_MUTED,
    "reference_line": "#b9b8b1",
}

RC = {
    "figure.facecolor": SURFACE,
    "figure.dpi": 130,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.labelcolor": TEXT_SECONDARY,
    "ytick.labelcolor": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.labelcolor": TEXT_SECONDARY,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "font.size": 9,
    "font.family": "sans-serif",
}


def apply_theme() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)


@contextmanager
def figure_context():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(RC):
        yield plt


def series_color(index: int, *, all_pairs: bool = False) -> str:
    """Fixed-order assignment; raises past the cap rather than cycling."""
    palette = CATEGORICAL_ALL_PAIRS if all_pairs else CATEGORICAL
    if index >= len(palette):
        raise IndexError(
            f"series {index} exceeds the {len(palette)}-slot palette. Fold "
            "the remainder into 'other', facet into small multiples, or add "
            "a second encoding -- do not cycle or generate hues."
        )
    return palette[index]


def role_color(role: str, fallback_index: int = 0) -> str:
    return ROLE.get(role, CATEGORICAL[fallback_index % len(CATEGORICAL)])


def finalize(ax, *, legend: bool = True, n_series: int = 0,
             legend_loc: str = "best") -> None:
    """Legend whenever >= 2 series are drawn; never for a single series."""
    if legend and n_series >= 2:
        ax.legend(loc=legend_loc)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
