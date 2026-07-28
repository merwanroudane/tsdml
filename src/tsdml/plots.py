"""
Publication-quality figures.

Every function returns the :class:`matplotlib.figure.Figure` so you can keep
editing it, and every one accepts ``save_path`` (``.pdf`` keeps text editable
and vector-clean, which is what journals ask for).

Figure map against the paper:

===================================  ==========================================
:func:`plot_block_structure`         Figure 1 -- the five-fold RCF scheme
:func:`plot_irf_panel`               Figure 2 -- cumulative IRFs to a capital shock
:func:`plot_irf`                     a single panel of the above
:func:`plot_goldilocks_profile`      Section 3 -- the selected stability window
:func:`plot_sample_use`              Section 2.1 -- RCF vs NLO sample use
:func:`plot_irf_comparison`          Goldilocks vs RMSE tuning, or RCF vs NLO
:func:`plot_residuals`               stage-one residual inspection
===================================  ==========================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .folds import reverse_cf_folds, sample_use_nlo, sample_use_rcf
from .style import COLORS, PALETTE, use_journal_style

__all__ = [
    "plot_irf",
    "plot_irf_panel",
    "plot_irf_comparison",
    "plot_block_structure",
    "plot_goldilocks_profile",
    "plot_sample_use",
    "plot_residuals",
    "save_figure",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def save_figure(fig, save_path: Optional[str], **kwargs):
    """Save ``fig`` to ``save_path`` (creating parent folders) and return it."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, **kwargs)
    return fig


def _as_frame(obj) -> pd.DataFrame:
    """Accept a DataFrame or anything exposing ``to_frame()``."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_frame"):
        return obj.to_frame()
    raise TypeError(
        f"expected a DataFrame or a fitted model with .to_frame(), got {type(obj)!r}"
    )


def _require(frame: pd.DataFrame, cols: Sequence[str], who: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(
            f"{who}: the IRF frame is missing column(s) {missing}. "
            f"Use DMLLocalProjections.to_frame(), which emits "
            f"Horizon / Coefficient / CI_Lower_95 / CI_Upper_95 / "
            f"CI_Lower_90 / CI_Upper_90."
        )


def _draw_irf(ax, frame: pd.DataFrame, color: str, ci_alpha_95: float,
              ci_alpha_90: float, linewidth: float, label: Optional[str] = None,
              linestyle: str = "-", show_bands: bool = True) -> None:
    h = frame["Horizon"].values
    irf = frame["Coefficient"].values
    if show_bands and {"CI_Lower_95", "CI_Upper_95"} <= set(frame.columns):
        ax.fill_between(h, frame["CI_Lower_95"], frame["CI_Upper_95"],
                        color=color, alpha=ci_alpha_95, linewidth=0)
    if show_bands and {"CI_Lower_90", "CI_Upper_90"} <= set(frame.columns):
        ax.fill_between(h, frame["CI_Lower_90"], frame["CI_Upper_90"],
                        color=color, alpha=ci_alpha_90, linewidth=0)
    ax.plot(h, irf, color=color, linewidth=linewidth, linestyle=linestyle,
            label=label, zorder=10, solid_capstyle="round")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.55,
               zorder=1)
    ax.set_xlim(h.min(), h.max())


# --------------------------------------------------------------------------- #
# single IRF
# --------------------------------------------------------------------------- #

def plot_irf(
    irf,
    title: Optional[str] = None,
    ylabel: str = "Percentage change",
    xlabel: str = "Horizon",
    color: str = COLORS["rcf"],
    ci_alpha_95: float = 0.12,
    ci_alpha_90: float = 0.25,
    linewidth: float = 2.5,
    figsize: Tuple[float, float] = (6.0, 4.0),
    ci_labels: Tuple[str, str] = ("90% CI", "95% CI"),
    legend: bool = True,
    save_path: Optional[str] = None,
    style: bool = True,
    ax=None,
):
    """
    Plot one impulse response with nested 90% and 95% bands.

    Parameters
    ----------
    irf : DataFrame or fitted DMLLocalProjections
        Needs ``Horizon`` and ``Coefficient``; confidence columns are drawn if
        present.
    title, ylabel, xlabel : str
    color : str
        Line and band colour.
    ci_alpha_95, ci_alpha_90 : float
        Band opacities.  The 90% band sits on top and is darker, so the two
        read as nested.
    linewidth : float
    figsize : tuple
    ci_labels : tuple of str
        Legend labels for the inner and outer band.
    legend : bool
    save_path : str, optional
    style : bool, default True
        Apply the journal style for this figure.
    ax : matplotlib Axes, optional
        Draw into an existing axes instead of creating a figure.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import pandas as pd, matplotlib
    >>> matplotlib.use('Agg')
    >>> frame = pd.DataFrame({'Horizon': [0, 1, 2],
    ...                       'Coefficient': [0.1, 0.2, 0.15]})
    >>> fig = plot_irf(frame, title='demo')
    >>> type(fig).__name__
    'Figure'
    """
    frame = _as_frame(irf)
    _require(frame, ["Horizon", "Coefficient"], "plot_irf")

    ctx = use_journal_style(context=True) if style else _null_context()
    with ctx:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure

        _draw_irf(ax, frame, color, ci_alpha_95, ci_alpha_90, linewidth)

        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if legend:
            inner, outer = ci_labels
            handles = [Line2D([0], [0], color=color, linewidth=linewidth)]
            labels = ["IRF"]
            if {"CI_Lower_90", "CI_Upper_90"} <= set(frame.columns):
                handles.append(Patch(facecolor=color, alpha=ci_alpha_90, linewidth=0))
                labels.append(inner)
            if {"CI_Lower_95", "CI_Upper_95"} <= set(frame.columns):
                handles.append(Patch(facecolor=color, alpha=ci_alpha_95, linewidth=0))
                labels.append(outer)
            ax.legend(handles, labels, loc="best", ncol=1)

    return save_figure(fig, save_path)


# --------------------------------------------------------------------------- #
# multi-panel IRF -- the paper's Figure 2
# --------------------------------------------------------------------------- #

def plot_irf_panel(
    irfs: Dict[str, pd.DataFrame],
    order: Optional[Sequence[str]] = None,
    titles: Optional[Dict[str, str]] = None,
    ylabels: Optional[Dict[str, str]] = None,
    layout: Tuple[int, int] = (2, 3),
    color: str = COLORS["rcf"],
    highlight: Optional[str] = None,
    highlight_color: str = COLORS["highlight"],
    ci_alpha_95: float = 0.12,
    ci_alpha_90: float = 0.25,
    linewidth: float = 2.5,
    figsize: Optional[Tuple[float, float]] = None,
    xlabel: str = "Horizon",
    ci_labels: Tuple[str, str] = ("90% CI", "95% CI"),
    suptitle: Optional[str] = None,
    equal_widths: bool = True,
    save_path: Optional[str] = None,
    style: bool = True,
):
    """
    Multi-panel impulse responses -- the layout of the paper's Figure 2.

    Rows are laid out as independent sub-figures.  A final row holding fewer
    panels than the others is centred: with ``equal_widths=True`` (the default)
    its panels keep the same width as the rest of the grid and the slack goes
    into symmetric margins, which is what a typeset figure looks like.  Set it
    to ``False`` to let the short row stretch across the full width.  One shared
    legend sits below the grid.

    Parameters
    ----------
    irfs : dict of {str: DataFrame}
        Variable name to IRF frame (see :meth:`~tsdml.lp.DMLLocalProjections.to_frame`).
    order : sequence of str, optional
        Panel order.  Defaults to the dict order.
    titles, ylabels : dict, optional
        Per-variable overrides; keys not present fall back to the variable name
        and to ``'Percentage change'``.
    layout : tuple, default (2, 3)
        ``(rows, cols)``.  Must be able to hold every panel.
    color : str
    highlight : str, optional
        Variable drawn in ``highlight_color`` -- useful to foreground the
        headline result (GDP in the paper).
    ci_alpha_95, ci_alpha_90, linewidth : float
    figsize : tuple, optional
        Defaults to ``(5.2 * cols, 3.3 * rows + 0.6)``.
    xlabel : str
    ci_labels : tuple of str
    suptitle : str, optional
    save_path : str, optional
    style : bool, default True

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import pandas as pd, matplotlib
    >>> matplotlib.use('Agg')
    >>> f = pd.DataFrame({'Horizon': range(5), 'Coefficient': [0, .1, .2, .1, 0]})
    >>> fig = plot_irf_panel({'A': f, 'B': f, 'C': f}, layout=(1, 3))
    >>> len(fig.axes) >= 3
    True
    """
    names = list(order) if order is not None else list(irfs)
    missing = [n for n in names if n not in irfs]
    if missing:
        raise KeyError(f"plot_irf_panel: no IRF supplied for {missing}")

    n_rows, n_cols = layout
    if len(names) > n_rows * n_cols:
        raise ValueError(
            f"layout {layout} holds {n_rows * n_cols} panels but "
            f"{len(names)} variables were supplied"
        )

    titles = titles or {}
    ylabels = ylabels or {}
    if figsize is None:
        figsize = (5.2 * n_cols, 3.3 * n_rows + 0.6)

    row_counts: List[int] = []
    remaining = len(names)
    for _ in range(n_rows):
        take = min(n_cols, remaining)
        if take > 0:
            row_counts.append(take)
            remaining -= take

    ctx = use_journal_style(context=True) if style else _null_context()
    with ctx:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        subfigs = fig.subfigures(len(row_counts), 1) if len(row_counts) > 1 else [fig]

        axes: List[plt.Axes] = []
        for r, count in enumerate(row_counts):
            sf = subfigs[r]
            if equal_widths and count < n_cols:
                # lay the row on a 2*n_cols grid and give each panel two slots,
                # offset by the leftover so the row stays centred at full width
                gs = sf.add_gridspec(1, 2 * n_cols)
                offset = n_cols - count
                axes.extend(sf.add_subplot(gs[0, offset + 2 * i: offset + 2 * i + 2])
                            for i in range(count))
            else:
                row_axes = sf.subplots(1, count)
                axes.extend([row_axes] if count == 1 else list(row_axes))

        used_highlight = False
        for ax, name in zip(axes, names):
            frame = _as_frame(irfs[name])
            _require(frame, ["Horizon", "Coefficient"], "plot_irf_panel")
            this = highlight_color if (highlight is not None and name == highlight) else color
            used_highlight |= (this == highlight_color and highlight is not None)

            _draw_irf(ax, frame, this, ci_alpha_95, ci_alpha_90, linewidth)
            ax.set_title(titles.get(name, name))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabels.get(name, "Percentage change"))
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        inner, outer = ci_labels
        if used_highlight and highlight_color != color:
            irf_handle = (Line2D([0], [0], color=color, linewidth=linewidth),
                          Line2D([0], [0], color=highlight_color, linewidth=linewidth))
            inner_handle = (Patch(facecolor=color, alpha=ci_alpha_90, linewidth=0),
                            Patch(facecolor=highlight_color, alpha=ci_alpha_90, linewidth=0))
            outer_handle = (Patch(facecolor=color, alpha=ci_alpha_95, linewidth=0),
                            Patch(facecolor=highlight_color, alpha=ci_alpha_95, linewidth=0))
            handler_map = {tuple: HandlerTuple(ndivide=None, pad=0.0)}
        else:
            irf_handle = Line2D([0], [0], color=color, linewidth=linewidth)
            inner_handle = Patch(facecolor=color, alpha=ci_alpha_90, linewidth=0)
            outer_handle = Patch(facecolor=color, alpha=ci_alpha_95, linewidth=0)
            handler_map = None

        fig.legend([irf_handle, inner_handle, outer_handle],
                   ["IRF", inner, outer],
                   loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.03),
                   handler_map=handler_map, handlelength=3.0, handleheight=1.2)

        if suptitle:
            fig.suptitle(suptitle, fontsize=plt.rcParams["axes.titlesize"] * 1.1)

    return save_figure(fig, save_path)


# --------------------------------------------------------------------------- #
# comparison of tuning rules / designs
# --------------------------------------------------------------------------- #

def plot_irf_comparison(
    series: Dict[str, pd.DataFrame],
    reference: Optional[str] = None,
    title: Optional[str] = None,
    ylabel: str = "Percentage change",
    xlabel: str = "Horizon",
    colors: Optional[Dict[str, str]] = None,
    bands_for: Optional[str] = None,
    ci_alpha: float = 0.15,
    figsize: Tuple[float, float] = (6.6, 4.2),
    save_path: Optional[str] = None,
    style: bool = True,
):
    """
    Overlay several impulse responses -- Goldilocks vs RMSE, or RCF vs NLO.

    The reference series is drawn solid; every other series is dashed, which is
    the visual contract the paper's figures use (same hue family, linestyle
    carries the distinction).

    Parameters
    ----------
    series : dict of {label: DataFrame}
    reference : str, optional
        Label drawn solid.  Defaults to the first key.
    colors : dict, optional
        Per-label colour override; otherwise the palette is cycled.
    bands_for : str, optional
        Label whose confidence bands are shaded.  Defaults to ``reference``.
        Only one set of bands is drawn -- overlaying several makes the figure
        unreadable.
    ci_alpha : float
    figsize : tuple
    save_path : str, optional
    style : bool, default True

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(series)
    if not labels:
        raise ValueError("series is empty")
    reference = reference or labels[0]
    bands_for = bands_for or reference
    colors = colors or {}

    ctx = use_journal_style(context=True) if style else _null_context()
    with ctx:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        for i, label in enumerate(labels):
            frame = _as_frame(series[label])
            _require(frame, ["Horizon", "Coefficient"], "plot_irf_comparison")
            color = colors.get(label, PALETTE[i % len(PALETTE)])
            solid = label == reference
            if label == bands_for and {"CI_Lower_95", "CI_Upper_95"} <= set(frame.columns):
                ax.fill_between(frame["Horizon"], frame["CI_Lower_95"],
                                frame["CI_Upper_95"], color=color,
                                alpha=ci_alpha, linewidth=0)
            ax.plot(frame["Horizon"], frame["Coefficient"], color=color,
                    linestyle="-" if solid else "--",
                    marker="o" if solid else "s", markersize=4.5,
                    linewidth=2.4 if solid else 1.9, label=label, zorder=10)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(loc="best")

    return save_figure(fig, save_path)


# --------------------------------------------------------------------------- #
# fold diagram -- the paper's Figure 1
# --------------------------------------------------------------------------- #

def plot_block_structure(
    n: Optional[int] = None,
    K: int = 5,
    figsize: Tuple[float, float] = (11.0, 5.5),
    main_color: str = "#C0504D",
    aux_color: str = "#9BBB59",
    sample_color: str = "#8497B0",
    out_color: str = "white",
    show_arrows: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    style: bool = True,
):
    """
    Draw the Reverse Cross-Fitting scheme -- the paper's Figure 1.

    The top band is the whole sample.  Each row below is one fold: the **red**
    cell is the main block where residuals are formed, **green** cells are the
    auxiliary sample used to train the nuisances, and **white** cells are unused
    by that fold.  Arrows give the direction of estimation -- right-to-left for
    the early, time-reversed folds and left-to-right for the late ones.

    Parameters
    ----------
    n : int, optional
        Sample length.  Only the block count matters visually; defaults to
        ``100 * K``.
    K : int, default 5
    figsize : tuple
    main_color, aux_color, sample_color, out_color : str
    show_arrows : bool, default True
    title : str, optional
    save_path : str, optional
    style : bool, default True

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import matplotlib; matplotlib.use('Agg')
    >>> fig = plot_block_structure(K=5)
    >>> type(fig).__name__
    'Figure'
    """
    n = n if n is not None else 100 * K
    blocks = reverse_cf_folds(n, K)
    block_sets = [set(b) for b in blocks.main_blocks]

    ctx = use_journal_style(context=True) if style else _null_context()
    with ctx:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        # whole-sample band on top
        for j in range(K):
            ax.add_patch(plt.Rectangle((j, K), 1, 1, facecolor=sample_color,
                                       edgecolor="black", linewidth=1.0))
        ax.text(-0.15, K + 0.5, "Whole sample", ha="right", va="center")

        for k in range(K):
            row = K - 1 - k
            direction = "reverse" if k <= K // 2 - 1 else (
                "both" if (k == K // 2 and K % 2 == 1) else "forward")

            if direction == "reverse":
                aux = set(blocks.aux_right_blocks[k])
            elif direction == "forward":
                aux = set(blocks.aux_left_blocks[k])
            else:
                aux = set(blocks.aux_right_blocks[k]) | set(blocks.aux_left_blocks[k])

            for j in range(K):
                if j == k:
                    face = main_color
                elif block_sets[j] & aux:
                    face = aux_color
                else:
                    face = out_color
                ax.add_patch(plt.Rectangle((j, row), 1, 1, facecolor=face,
                                           edgecolor="black", linewidth=1.0))

            ax.text(-0.15, row + 0.5, f"fold {k}", ha="right", va="center")

            if show_arrows:
                _fold_arrows(ax, k, K, row, direction, block_sets, aux)

        ax.set_xlim(-1.6, K + 0.1)
        ax.set_ylim(-0.1, K + 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_aspect("equal")

        legend = [
            mpatches.Patch(facecolor=sample_color, edgecolor="black", label="Whole sample"),
            mpatches.Patch(facecolor=main_color, edgecolor="black", label="Main block $B_k$"),
            mpatches.Patch(facecolor=aux_color, edgecolor="black", label="Auxiliary sample"),
            mpatches.Patch(facecolor=out_color, edgecolor="black", label="Not used by the fold"),
        ]
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=4, frameon=False)
        ax.set_title(title or f"Reverse Cross-Fitting, K = {K}")

    return save_figure(fig, save_path)


def _fold_arrows(ax, k, K, row, direction, block_sets, aux) -> None:
    """Arrow(s) showing the direction nuisances are estimated in for one fold."""
    kw = dict(head_width=0.16, head_length=0.22, fc="black", ec="black",
              length_includes_head=True, linewidth=1.1)
    aux_cols = [j for j in range(K) if block_sets[j] & aux]
    if not aux_cols:
        return
    y = row + 0.5
    if direction in ("reverse", "both"):
        right = [j for j in aux_cols if j > k]
        if right:
            start = max(right) + 0.85
            ax.arrow(start, y, -(start - (k + 0.9)), 0, **kw)
    if direction in ("forward", "both"):
        left = [j for j in aux_cols if j < k]
        if left:
            start = min(left) + 0.15
            ax.arrow(start, y, (k + 0.1) - start, 0, **kw)


# --------------------------------------------------------------------------- #
# Goldilocks zone
# --------------------------------------------------------------------------- #

def plot_goldilocks_profile(
    rmse_profile: Sequence[float],
    grid: Optional[Sequence[float]] = None,
    window_size: int = 3,
    title: Optional[str] = None,
    xlabel: str = r"penalty $\lambda$",
    ylabel: str = "validation RMSE",
    figsize: Tuple[float, float] = (6.6, 4.2),
    logx: bool = False,
    save_path: Optional[str] = None,
    style: bool = True,
):
    """
    Show the validation RMSE profile, the selected window and the chosen value.

    This is the picture behind Section 3: the shaded band is the Goldilocks
    zone :math:`\\mathcal W_{j^\\star}` -- the window with the lowest
    :math:`\\tilde V_j + \\bar{\\mathcal R}^{\\sim}_j` -- and the marker is
    :math:`\\lambda^\\star`, the RMSE minimiser *inside* that window.  When the
    global RMSE minimum sits at a sharp, unstable spike outside the zone, it is
    marked separately so the divergence is visible.

    Parameters
    ----------
    rmse_profile : sequence of float
        Validation RMSE per grid point, in grid order -- e.g.
        ``calibrator.rmse_profiles_[0]['outcome']``.
    grid : sequence of float, optional
        Hyperparameter values for the x-axis.  Defaults to grid indices.
    window_size : int, default 3
    title, xlabel, ylabel : str
    figsize : tuple
    logx : bool, default False
        Log-scale the x-axis, natural for a penalty grid spanning decades.
    save_path : str, optional
    style : bool, default True

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import matplotlib; matplotlib.use('Agg')
    >>> fig = plot_goldilocks_profile([0.5, 0.2, 0.6, 0.31, 0.30, 0.31])
    >>> type(fig).__name__
    'Figure'
    """
    from .calibration import goldilocks_select

    scores = np.asarray(list(rmse_profile), dtype=float)
    x = np.asarray(grid, dtype=float) if grid is not None else np.arange(len(scores))
    if len(x) != len(scores):
        raise ValueError(
            f"grid has {len(x)} values but the RMSE profile has {len(scores)}"
        )

    chosen = goldilocks_select(scores, window_size)
    global_min = int(np.argmin(scores))

    S = min(int(window_size), len(scores))
    starts = range(len(scores) - S + 1)
    variances = np.array([np.var(scores[s:s + S]) for s in starts])
    means = np.array([np.mean(scores[s:s + S]) for s in starts])
    norm = lambda v: ((v - v.min()) / (v.max() - v.min())  # noqa: E731
                      if v.max() - v.min() > 1e-10 else np.zeros_like(v))
    j_star = int(np.argmin(norm(variances) + norm(means)))
    win = slice(j_star, j_star + S)

    ctx = use_journal_style(context=True) if style else _null_context()
    with ctx:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        ax.axvspan(x[win][0], x[win][-1], color=COLORS["rcf"], alpha=0.13,
                   linewidth=0, label="Goldilocks zone")
        ax.plot(x, scores, color=COLORS["rcf"], linewidth=1.9, marker="o",
                markersize=3.4, alpha=0.9, label="validation RMSE")
        ax.plot(x[chosen], scores[chosen], marker="*", markersize=16,
                color=COLORS["rcf"], markeredgecolor="black", markeredgewidth=0.7,
                linestyle="none", zorder=12, label=r"selected $\lambda^\star$")
        if global_min != chosen:
            ax.plot(x[global_min], scores[global_min], marker="v", markersize=9,
                    color=COLORS["rmse"], markeredgecolor="black",
                    markeredgewidth=0.6, linestyle="none", zorder=11,
                    label="RMSE minimiser")

        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"Stability-based tuning (window S = {window_size})")
        ax.legend(loc="best")

    return save_figure(fig, save_path)


# --------------------------------------------------------------------------- #
# sample use
# --------------------------------------------------------------------------- #

def plot_sample_use(
    k_values: Iterable[int] = range(3, 16),
    figsize: Tuple[float, float] = (6.6, 4.2),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    style: bool = True,
):
    """
    Nuisance-sample usage of RCF against NLO, as a function of ``K``.

    Reproduces the comparison in Section 2.1: RCF uses more of the sample for
    ``K = 3, ..., 9``, the two designs coincide at ``K = 11``, and NLO overtakes
    from ``K = 12`` (and at ``K = 10``).  The moderate-``K`` region is where
    short macro samples live, which is the paper's point.

    Examples
    --------
    >>> import matplotlib; matplotlib.use('Agg')
    >>> fig = plot_sample_use()
    >>> type(fig).__name__
    'Figure'
    """
    ks = np.array(list(k_values), dtype=int)
    u_rcf = np.array([sample_use_rcf(k) for k in ks])
    u_nlo = np.array([sample_use_nlo(k) for k in ks])

    ctx = use_journal_style(context=True) if style else _null_context()
    with ctx:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.plot(ks, u_rcf, marker="o", color=COLORS["rcf"], label="RCF")
        ax.plot(ks, u_nlo, marker="s", linestyle="--", color=COLORS["nlo"],
                label="NLO")
        better = ks[u_rcf > u_nlo]
        if better.size:
            ax.axvspan(better.min() - 0.4, better.max() + 0.4,
                       color=COLORS["rcf"], alpha=0.08, linewidth=0)
        ax.set_xlabel("number of folds $K$")
        ax.set_ylabel("share of the sample used for nuisance estimation")
        ax.set_title(title or "Sample use: Reverse Cross-Fitting vs Neighbors-Left-Out")
        ax.set_xticks(ks)
        ax.legend(loc="best")

    return save_figure(fig, save_path)


# --------------------------------------------------------------------------- #
# residuals
# --------------------------------------------------------------------------- #

def plot_residuals(
    residuals: Dict[str, np.ndarray],
    index=None,
    n_blocks: Optional[int] = None,
    outcome_label: str = "outcome residual",
    treatment_label: str = "policy residual",
    figsize: Tuple[float, float] = (10.0, 5.4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    style: bool = True,
):
    """
    Plot the stage-one residuals with fold boundaries marked.

    Two stacked panels -- the residualised outcome and the residualised policy
    variable.  Vertical rules mark the main-block boundaries, which is where
    conditional stability is doing its work: a visible level or variance shift
    at a boundary is a reason to run :func:`tsdml.diagnostics.diagnose`.

    Parameters
    ----------
    residuals : dict
        ``{'outcome': ndarray, 'treatment': ndarray}``, e.g. from
        :meth:`~tsdml.rcf.ReverseCrossFitting.get_residuals`.
    index : array-like, optional
        Dates for the x-axis.
    n_blocks : int, optional
        Draw fold boundaries for this ``K``.
    outcome_label, treatment_label : str
    figsize : tuple
    title : str, optional
    save_path : str, optional
    style : bool, default True

    Returns
    -------
    matplotlib.figure.Figure
    """
    chi = np.asarray(residuals["outcome"], dtype=float)
    xi = np.asarray(residuals["treatment"], dtype=float)
    T = len(chi)

    # Plot against observation position, not calendar date. The estimation
    # sample can have holes -- dropped rows, an excluded episode -- and joining
    # across one with a straight line invents data that was never estimated on.
    x = np.arange(T)
    dates = pd.to_datetime(index) if index is not None else None

    ctx = use_journal_style(context=True) if style else _null_context()
    with ctx:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                 constrained_layout=True)
        for ax, series, label, color in (
            (axes[0], chi, outcome_label, COLORS["rcf"]),
            (axes[1], xi, treatment_label, COLORS["nlo"]),
        ):
            ax.plot(x, series, color=color, linewidth=1.6)
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
            ax.set_ylabel(label)

            if n_blocks:
                for b in reverse_cf_folds(T, n_blocks).main_blocks[1:]:
                    ax.axvline(b[0], color="grey", linewidth=0.8, linestyle=":",
                               alpha=0.8)

            if dates is not None:
                # shade any break in the calendar so the reader sees it
                gaps = _calendar_gaps(dates)
                for start, end in gaps:
                    ax.axvspan(start - 0.5, start + 0.5, color=COLORS["rmse"],
                               alpha=0.12, linewidth=0)

        if dates is not None:
            step = max(1, T // 8)
            ticks = list(range(0, T, step))
            axes[-1].set_xticks(ticks)
            axes[-1].set_xticklabels([dates[i].strftime("%Y-%m") for i in ticks],
                                     rotation=0)
            axes[-1].set_xlabel("observation (date label)")
        else:
            axes[-1].set_xlabel("observation")

        if title:
            fig.suptitle(title)

    return save_figure(fig, save_path)


def _calendar_gaps(dates: pd.DatetimeIndex):
    """Positions where consecutive observations skip more than one period."""
    if len(dates) < 3:
        return []
    deltas = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    typical = float(np.median(deltas))
    if typical <= 0:
        return []
    return [(i, i + 1) for i, dt in enumerate(deltas) if dt > 1.75 * typical]


# --------------------------------------------------------------------------- #

class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
