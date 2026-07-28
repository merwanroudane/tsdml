"""
Figure styling for journal submission.

The defaults target the look of *The Econometrics Journal* / *Journal of
Applied Econometrics* figures: serif type, no top or right spine, hairline zero
line, light dotted grid, and a colourblind-safe palette (Wong 2011) matching
the paper's own figures -- ``#0173B2`` blue for the reference series,
``#D55E00`` vermillion for the first comparison, and so on.

Use it globally::

    from tsdml.style import use_journal_style
    use_journal_style()

or locally::

    with use_journal_style(context=True):
        fig = plot_irf(frame)

Every plotting function in :mod:`tsdml.plots` applies the style itself unless
you pass ``style=False``, so you rarely need to call this by hand.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["PALETTE", "COLORS", "use_journal_style", "journal_rc", "lighten"]


#: Colourblind-safe palette (Wong 2011), in the order used by the paper.
PALETTE = [
    "#0173B2",  # blue      -- reference series (RCF, Goldilocks)
    "#D55E00",  # vermillion-- first comparison (NLO, mis-specified)
    "#029E73",  # green
    "#CC78BC",  # purple
    "#DE8F05",  # orange
    "#555555",  # charcoal  -- neutral RMSE comparison
    "#9E4500",  # brown
    "#117733",  # dark teal
]

#: Semantic aliases.
COLORS: Dict[str, str] = {
    "rcf": "#0173B2",
    "nlo": "#D55E00",
    "lp": "#0173B2",
    "goldilocks": "#0173B2",
    "rmse": "#555555",
    "zero": "#000000",
    "highlight": "#0173B2",
}


def lighten(hex_color: str, factor: float = 0.45) -> str:
    """
    Mix a hex colour with white.

    Parameters
    ----------
    hex_color : str
        ``'#RRGGBB'``.
    factor : float
        ``0`` returns the input, ``1`` returns white.

    Examples
    --------
    >>> lighten('#0173B2', 0.5)
    '#80B9D8'
    """
    c = hex_color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda v: int(round(v + (255 - v) * factor))  # noqa: E731
    return f"#{mix(r):02X}{mix(g):02X}{mix(b):02X}"


def journal_rc(base_fontsize: float = 11.0, serif: bool = True) -> Dict[str, object]:
    """
    Return the rcParams dict used by :func:`use_journal_style`.

    Parameters
    ----------
    base_fontsize : float, default 11
        Body font size; ticks, labels and titles scale from it.
    serif : bool, default True
        Serif type matches most econometrics journals.  Set ``False`` for
        slides.
    """
    family = "serif" if serif else "sans-serif"
    return {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "font.family": family,
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize * 1.18,
        "axes.labelsize": base_fontsize * 1.02,
        "xtick.labelsize": base_fontsize * 0.92,
        "ytick.labelsize": base_fontsize * 0.92,
        "legend.fontsize": base_fontsize * 0.95,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.titlepad": 8.0,
        "axes.labelpad": 4.0,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "legend.frameon": False,
        "legend.handlelength": 2.4,

        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "mathtext.fontset": "cm" if serif else "dejavusans",
        "pdf.fonttype": 42,   # editable text in the PDF
        "ps.fonttype": 42,
    }


@contextmanager
def _style_context(rc: Dict[str, object]):
    with plt.rc_context(rc):
        yield


def use_journal_style(base_fontsize: float = 11.0, serif: bool = True,
                      context: bool = False):
    """
    Apply the journal style.

    Parameters
    ----------
    base_fontsize : float, default 11
    serif : bool, default True
    context : bool, default False
        If ``True`` return a context manager that restores the previous
        rcParams on exit; if ``False`` mutate the global rcParams and return
        the dict that was applied.

    Examples
    --------
    >>> rc = use_journal_style()
    >>> rc['axes.spines.top']
    False
    """
    rc = journal_rc(base_fontsize=base_fontsize, serif=serif)
    if context:
        return _style_context(rc)
    mpl.rcParams.update(rc)
    return rc
