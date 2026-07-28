"""
Publication-quality tables.

Every builder returns a :class:`pandas.DataFrame` you can inspect or reshape,
and :func:`to_latex` turns any of them into a ``booktabs`` table with a caption,
a label and a notes block -- the format journals expect.

Conventions
-----------
* Significance stars follow the usual econometrics convention:
  ``***`` p < 0.01, ``**`` p < 0.05, ``*`` p < 0.10.
* Standard errors go in parentheses beneath the coefficient when
  ``se_below=True``, which is the standard layout for estimation tables.
* Numbers are formatted once, as strings, so LaTeX and the console agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .folds import sample_use_nlo, sample_use_rcf

__all__ = [
    "stars",
    "irf_table",
    "estimation_table",
    "calibration_table",
    "sample_use_table",
    "to_latex",
    "to_markdown",
]


def stars(p: float, thresholds: Sequence[float] = (0.01, 0.05, 0.10)) -> str:
    """
    Significance stars for a p-value.

    Examples
    --------
    >>> stars(0.004), stars(0.03), stars(0.08), stars(0.5)
    ('***', '**', '*', '')
    """
    if p is None or not np.isfinite(p):
        return ""
    a, b, c = thresholds
    if p < a:
        return "***"
    if p < b:
        return "**"
    if p < c:
        return "*"
    return ""


def _fmt(x: float, digits: int) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------------- #
# IRF table
# --------------------------------------------------------------------------- #

def irf_table(
    irf,
    digits: int = 3,
    se_below: bool = True,
    show_ci: bool = True,
    ci_level: int = 95,
    horizon_label: str = "h",
) -> pd.DataFrame:
    """
    Format one impulse response as a journal-ready table.

    Parameters
    ----------
    irf : DataFrame or fitted DMLLocalProjections
        Needs ``Horizon``, ``Coefficient``; uses ``Std_Error``, ``p_value`` and
        the CI columns when present.
    digits : int, default 3
    se_below : bool, default True
        Put the standard error in parentheses under the coefficient.  When
        ``False`` it gets its own column.
    show_ci : bool, default True
        Append a confidence-interval column.
    ci_level : {95, 90}, default 95
    horizon_label : str, default 'h'

    Returns
    -------
    DataFrame
        String-formatted and ready for :func:`to_latex`.

    Examples
    --------
    >>> import pandas as pd
    >>> f = pd.DataFrame({'Horizon': [0, 1], 'Coefficient': [0.12, -0.03],
    ...                   'Std_Error': [0.04, 0.05], 'p_value': [0.003, 0.55],
    ...                   'CI_Lower_95': [0.04, -0.13], 'CI_Upper_95': [0.20, 0.07]})
    >>> list(irf_table(f).columns)
    ['h', 'Estimate', '95% CI']
    >>> irf_table(f)['Estimate'].iloc[0]
    '0.120***'
    """
    frame = irf if isinstance(irf, pd.DataFrame) else irf.to_frame()
    if "Horizon" not in frame or "Coefficient" not in frame:
        raise KeyError("irf_table needs 'Horizon' and 'Coefficient' columns")

    lo, hi = f"CI_Lower_{ci_level}", f"CI_Upper_{ci_level}"
    rows = []
    for _, r in frame.iterrows():
        p = r.get("p_value", np.nan)
        se = r.get("Std_Error", np.nan)
        est = _fmt(r["Coefficient"], digits) + stars(p)
        row = {horizon_label: int(r["Horizon"])}
        if se_below:
            row["Estimate"] = est
        else:
            row["Estimate"] = est
            row["Std. error"] = _fmt(se, digits)
        if show_ci and lo in frame.columns and hi in frame.columns:
            row[f"{ci_level}% CI"] = f"[{_fmt(r[lo], digits)}, {_fmt(r[hi], digits)}]"
        rows.append(row)

        if se_below and np.isfinite(se):
            filler = {horizon_label: "", "Estimate": f"({_fmt(se, digits)})"}
            for key in row:
                filler.setdefault(key, "")
            rows.append(filler)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# estimation comparison table
# --------------------------------------------------------------------------- #

def estimation_table(
    models: Dict[str, object],
    digits: int = 4,
    include_ci: bool = True,
    include_blocks: bool = True,
) -> pd.DataFrame:
    """
    Side-by-side table of scalar DML estimates.

    Parameters
    ----------
    models : dict of {label: fitted estimator or dict}
        Accepts :class:`~tsdml.rcf.ReverseCrossFitting`,
        :class:`~tsdml.nlo.NLOCrossFitting`, or a plain dict with keys
        ``coef``, ``std_error``, ``t_stat``, ``p_value``, ``ci_lower``,
        ``ci_upper``.
    digits : int, default 4
    include_ci : bool, default True
    include_blocks : bool, default True
        Report ``K`` and the implied sample-use share.

    Returns
    -------
    DataFrame
        Rows are statistics, columns are specifications -- the layout of a
        typical results table.

    Examples
    --------
    >>> res = {'coef': 1.02, 'std_error': 0.11, 't_stat': 9.3,
    ...        'p_value': 0.0, 'ci_lower': 0.80, 'ci_upper': 1.24}
    >>> estimation_table({'RCF': res}).loc['theta', 'RCF']
    '1.0200***'
    """
    out: Dict[str, Dict[str, str]] = {}
    for label, model in models.items():
        if isinstance(model, dict):
            r = model
            K = model.get("n_blocks")
            design = model.get("design")
        else:
            r = getattr(model, "results_", None)
            if r is None:
                raise TypeError(f"'{label}' is not a fitted estimator with results_")
            K = getattr(model, "n_blocks", None)
            design = type(model).__name__

        col = {"theta": _fmt(r["coef"], digits) + stars(r.get("p_value", np.nan)),
               "Std. error": f"({_fmt(r['std_error'], digits)})",
               "t-statistic": _fmt(r.get("t_stat", np.nan), 3),
               "p-value": _fmt(r.get("p_value", np.nan), 4)}
        if include_ci:
            col["95% CI"] = (f"[{_fmt(r.get('ci_lower', np.nan), digits)}, "
                             f"{_fmt(r.get('ci_upper', np.nan), digits)}]")
        if include_blocks and K:
            col["Folds K"] = str(int(K))
            share = sample_use_nlo(int(K)) if (design or "").startswith("NLO") \
                else sample_use_rcf(int(K))
            col["Sample use"] = _fmt(share, 3)
        out[label] = col

    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# calibration table
# --------------------------------------------------------------------------- #

def calibration_table(calibrator, digits: int = 6) -> pd.DataFrame:
    """
    Per-fold tuning outcome: selected hyperparameters and validation RMSE.

    Parameters
    ----------
    calibrator : tsdml.calibration.Calibrator
        Already calibrated.
    digits : int, default 6

    Returns
    -------
    DataFrame
    """
    if calibrator.block_specific_learners_ is None:
        raise ValueError("calibrator has not been calibrated yet")

    rows = []
    for k in range(calibrator.n_blocks):
        row = calibrator.calibration_scores_.get(k, {})
        entry = {
            "Fold": k,
            "n train": row.get("n_train"),
            "n validate": row.get("n_val"),
            "Outcome RMSE": _fmt(row.get("outcome_score", np.nan), digits),
            "Policy RMSE": _fmt(row.get("treatment_score", np.nan), digits),
        }
        for eq, key in (("Outcome", "outcome_params"), ("Policy", "treatment_params")):
            params = row.get(key) or {}
            for name in ("alpha", "l1_ratio", "n_estimators", "max_depth"):
                if name in params:
                    value = params[name]
                    entry[f"{eq} {name}"] = (_fmt(float(value), 6)
                                             if isinstance(value, (int, float))
                                             else str(value))
        rows.append(entry)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# sample use
# --------------------------------------------------------------------------- #

def sample_use_table(k_values: Sequence[int] = tuple(range(3, 16)),
                     digits: int = 4) -> pd.DataFrame:
    """
    RCF versus NLO nuisance-sample usage across ``K`` (paper, Section 2.1).

    Examples
    --------
    >>> t = sample_use_table([6, 11, 12])
    >>> t.loc[t['K'] == 11, 'Winner'].iloc[0]
    'tie'
    """
    rows = []
    for K in k_values:
        u_r, u_n = sample_use_rcf(K), sample_use_nlo(K)
        if abs(u_r - u_n) < 1e-12:
            winner = "tie"
        else:
            winner = "RCF" if u_r > u_n else "NLO"
        rows.append({"K": int(K), "u_RCF": round(u_r, digits),
                     "u_NLO": round(u_n, digits), "Winner": winner})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# exporters
# --------------------------------------------------------------------------- #

def to_latex(
    frame: pd.DataFrame,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    notes: Optional[str] = None,
    index: bool = False,
    column_format: Optional[str] = None,
    position: str = "htbp",
    save_path: Optional[str] = None,
    escape: bool = True,
) -> str:
    """
    Render a frame as a ``booktabs`` LaTeX table.

    Requires ``\\usepackage{booktabs}`` in the preamble; add ``threeparttable``
    if you want the notes typeset as a proper table note environment (the
    output here uses a plain ``\\footnotesize`` paragraph, which needs nothing
    extra).

    Parameters
    ----------
    frame : DataFrame
    caption, label, notes : str, optional
    index : bool, default False
        Print the row index as the first column -- switch on for
        :func:`estimation_table`, whose index carries the statistic names.
    column_format : str, optional
        LaTeX column spec.  Defaults to left-aligned first column and centred
        remainder.
    position : str, default 'htbp'
    save_path : str, optional
        Write the LaTeX to this path as well as returning it.
    escape : bool, default True
        Escape LaTeX specials in the cells.  Turn off if your cells already
        contain math.

    Returns
    -------
    str

    Examples
    --------
    >>> import pandas as pd
    >>> tex = to_latex(pd.DataFrame({'h': [0], 'Estimate': ['0.12***']}),
    ...                caption='Impulse responses', label='tab:irf')
    >>> '\\\\toprule' in tex and 'tab:irf' in tex
    True
    """
    ncol = len(frame.columns) + (1 if index else 0)
    if column_format is None:
        column_format = "l" + "c" * (ncol - 1)

    body = frame.to_latex(
        index=index, escape=escape, column_format=column_format,
        na_rep="--", bold_rows=False,
    )
    # pandas already emits booktabs rules when jinja is absent? make sure:
    body = (body.replace("\\hline\\hline", "\\toprule")
                .replace("\\hline", "\\midrule"))
    if "\\toprule" not in body:
        body = body.replace("\\begin{tabular}{" + column_format + "}\n",
                            "\\begin{tabular}{" + column_format + "}\n\\toprule\n", 1)
        body = body.replace("\\end{tabular}", "\\bottomrule\n\\end{tabular}")

    parts = [f"\\begin{{table}}[{position}]", "\\centering"]
    if caption:
        parts.append(f"\\caption{{{caption}}}")
    if label:
        parts.append(f"\\label{{{label}}}")
    parts.append(body.rstrip())
    if notes:
        parts.append("\\begin{minipage}{\\linewidth}\\footnotesize")
        parts.append(f"\\textit{{Notes:}} {notes}")
        parts.append("\\end{minipage}")
    parts.append("\\end{table}")
    tex = "\n".join(parts) + "\n"

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tex, encoding="utf-8")
    return tex


def to_markdown(frame: pd.DataFrame, index: bool = False,
                save_path: Optional[str] = None) -> str:
    """
    Render a frame as a Markdown table (handy for READMEs and issues).

    Falls back to a fixed-width text table when ``tabulate`` is not installed.
    """
    try:
        md = frame.to_markdown(index=index)
    except ImportError:
        md = frame.to_string(index=index)
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
    return md
