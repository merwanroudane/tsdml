"""
Data preparation for time-series DML.

The input is a "wide" table with two metadata rows on top of the observations:

===============  ==========  ============  ===
``sasdate``      ``GDP``     ``Spread``    ...
===============  ==========  ============  ===
``speed``        ``slow``    ``fast``      ...
``Transform:``   ``5``       ``2``         ...
``2005-06-30``   ``107.51``  ``1.25``      ...
===============  ==========  ============  ===

``speed`` implements the paper's block-recursive timing assumption
(Section 5.2): **fast**-moving financial variables (prices, yields, spreads,
exchange rates) are plausibly observed within the quarter and enter the control
set contemporaneously; **slow**-moving macro and banking variables enter only
with lags, which avoids conditioning on post-treatment mediators sitting on the
transmission path.

``Transform:`` gives the stationarity-inducing transformation applied to each
column, in the FRED-MD convention extended with quarterly codes -- see
:data:`TRANSFORM_CODES`.

:meth:`DataProcessor.data_prep` returns the four arrays every estimator in this
package consumes: ``X`` (contemporaneous fast variables, all lags, optional
constant), ``y``, ``policy`` and ``leads``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["DataProcessor", "TRANSFORM_CODES", "transform_series"]


#: Human-readable description of every supported transformation code.
TRANSFORM_CODES: Dict[int, str] = {
    1: "no transformation (level)",
    2: "first difference",
    3: "second difference",
    4: "log",
    5: "log first difference, x100 (percent)",
    6: "log second difference",
    7: "point-over-point growth rate, x100 (percent)",
    8: "log twelfth difference (year-on-year, monthly)",
    81: "log fourth difference (year-on-year, quarterly)",
    9: "series divided by its HP trend",
    10: "first difference year-on-year (monthly)",
    101: "first difference year-on-year (quarterly)",
    11: "growth year-on-year, x100 (monthly)",
    111: "growth year-on-year, x100 (quarterly)",
}


# --------------------------------------------------------------------------- #
# Transformations
# --------------------------------------------------------------------------- #

def _diff(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    out[k:] = x[k:] - x[:-k]
    return out


def _log_diff(x: np.ndarray, k: int = 1) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.log(x.astype(float))
    return _diff(lx, k)


def _growth(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[k:] = (x[k:] - x[:-k]) / x[:-k]
    return out


def _hp_detrend(x: np.ndarray, lamb: float = 14400.0) -> np.ndarray:
    """
    Ratio of the series to its Hodrick-Prescott trend.

    This is the paper's code 9: ``x_t / tau_t``, not the additive cycle.
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter

    s = pd.Series(x, dtype=float)
    mask = s.notna()
    out = np.full(len(s), np.nan)
    if mask.sum() < 4:
        return out
    _, trend = hpfilter(s[mask], lamb=lamb)
    out[mask.values] = np.asarray(s[mask]) / np.asarray(trend)
    return out


def transform_series(x: np.ndarray, tcode: int) -> np.ndarray:
    """
    Apply transformation ``tcode`` to a 1-D series.

    NaNs pad the beginning so the output keeps the input length.

    .. note::

       Percentage-type transformations are returned **in percent**, matching
       the paper's replication code: codes 5, 7, 11 and 111 are multiplied by
       100, while the log year-on-year codes 6, 8 and 81 are not.  This is why
       the estimated responses of log-differenced variables read directly as
       percentage changes.

    Parameters
    ----------
    x : ndarray
    tcode : int
        A key of :data:`TRANSFORM_CODES`.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> transform_series(np.array([1.0, 3.0, 6.0]), 2)
    array([nan,  2.,  3.])
    >>> np.round(transform_series(np.array([100.0, 110.0]), 7), 3)
    array([nan, 10.])
    >>> np.round(transform_series(np.array([100.0, 110.0]), 5), 3)
    array([  nan, 9.531])
    """
    x = np.asarray(x, dtype=float)
    tcode = int(tcode)
    table = {
        1: lambda v: v.copy(),
        2: lambda v: _diff(v, 1),
        3: lambda v: _diff(_diff(v, 1), 1),
        4: lambda v: np.log(v, out=np.full_like(v, np.nan), where=v > 0),
        5: lambda v: 100.0 * _log_diff(v, 1),
        6: lambda v: _diff(_log_diff(v, 1), 1),
        7: lambda v: 100.0 * _growth(v, 1),
        8: lambda v: _log_diff(v, 12),
        81: lambda v: _log_diff(v, 4),
        9: _hp_detrend,
        10: lambda v: _diff(v, 12),
        101: lambda v: _diff(v, 4),
        11: lambda v: 100.0 * _growth(v, 12),
        111: lambda v: 100.0 * _growth(v, 4),
    }
    if tcode not in table:
        raise ValueError(
            f"invalid transformation code {tcode}; supported codes are "
            f"{sorted(TRANSFORM_CODES)}"
        )
    return table[tcode](x)


# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #

class DataProcessor:
    """
    Turn a metadata-headed wide table into ``(X, y, policy, leads)``.

    Attributes set after :meth:`data_prep`
    --------------------------------------
    original_index : pandas.DatetimeIndex
        Dates of the retained rows -- pass to
        :meth:`tsdml.lp.DMLLocalProjections.fit` so residuals carry dates.
    original_data : pandas.DataFrame
        Untransformed numeric data.
    transformed_data : pandas.DataFrame
        Post-transformation columns, named ``"{variable}_{tcode}"``.
    feature_names_ : list of str
        Column names of ``X``, in order.
    slow_columns_, fast_columns_ : list of str
    names_ : dict
        ``{'treatment': ..., 'outcome': ...}`` transformed column names.

    Examples
    --------
    >>> from tsdml.datasets import load_macroprudential
    >>> df = load_macroprudential()
    >>> proc = DataProcessor()
    >>> X, y, d, leads = proc.data_prep(
    ...     df, num_lags=3, H=8,
    ...     treatment_var='Tier 1 capital to risk-weighted assets_Percent',
    ...     treatment_code=2, outcome_var='GDP_K2020', outcome_code=5,
    ...     start_date='2005-12-31', K=6)
    >>> X.shape[0] % 6
    0
    >>> leads.shape[1]
    9
    """

    def __init__(self):
        self.original_index: Optional[pd.Index] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.transformed_data: Optional[pd.DataFrame] = None
        self.feature_names_: Optional[List[str]] = None
        self.slow_columns_: Optional[List[str]] = None
        self.fast_columns_: Optional[List[str]] = None
        self.names_: Dict[str, str] = {}
        self.data_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------ transform -- #

    def process_dataframe(
        self,
        df: pd.DataFrame,
        variables_to_remove: Optional[List[str]] = None,
        variables_to_fast: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Strip the metadata rows, transform every column and classify by speed.

        Parameters
        ----------
        df : DataFrame
            Rows ``'speed'`` and ``'Transform:'`` on top of a dated body.
        variables_to_remove : list of str, optional
            Original variable names to drop from the contemporaneous set.
        variables_to_fast : list of str, optional
            Slow variables to promote to contemporaneous.

        Returns
        -------
        transformed, slow_columns, fast_columns
        """
        for row in ("speed", "Transform:"):
            if row not in df.index:
                raise ValueError(
                    f"row '{row}' not found in the index. The first two rows of "
                    f"the table must be 'speed' and 'Transform:'."
                )

        speed = df.loc["speed"].astype(str).str.strip().str.lower()
        transforms = df.loc["Transform:"]
        data = df.drop(index=["speed", "Transform:"])

        try:
            data.index = pd.to_datetime(data.index, format="mixed", dayfirst=True)
        except (ValueError, TypeError):
            try:
                data.index = pd.to_datetime(data.index, dayfirst=True)
            except (ValueError, TypeError):
                import warnings
                warnings.warn("could not parse the date index; keeping it as-is")

        data = data.apply(pd.to_numeric, errors="coerce")
        self.original_data = data.copy()

        slow_raw = [c for c in df.columns if speed[c] == "slow"]
        fast_raw = [c for c in df.columns if speed[c] == "fast"]
        if variables_to_fast:
            for var in variables_to_fast:
                if var in slow_raw:
                    slow_raw.remove(var)
                if var not in fast_raw:
                    fast_raw.append(var)

        transformed: Dict[str, np.ndarray] = {}
        renamed: Dict[str, str] = {}
        for col in data.columns:
            tcode = int(float(transforms[col]))
            new_name = f"{col}_{tcode}"
            transformed[new_name] = transform_series(data[col].values, tcode)
            renamed[col] = new_name

        out = pd.DataFrame(transformed, index=data.index).dropna()

        slow_cols = [renamed[c] for c in slow_raw if c in renamed]
        fast_cols = [renamed[c] for c in fast_raw if c in renamed]
        if variables_to_remove:
            drop = {renamed[c] for c in variables_to_remove if c in renamed}
            fast_cols = [c for c in fast_cols if c not in drop]
            slow_cols = [c for c in slow_cols if c not in drop]
            out = out.drop(columns=[c for c in drop if c in out.columns])

        self.transformed_data = out
        self.slow_columns_, self.fast_columns_ = slow_cols, fast_cols
        return out, slow_cols, fast_cols

    # ----------------------------------------------------------------- prep -- #

    def data_prep(
        self,
        df: pd.DataFrame,
        num_lags: int,
        H: int = 1,
        treatment_var: str = "FEDFUNDS",
        outcome_var: str = "GDP",
        treatment_code: Optional[int] = None,
        outcome_code: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        scaling_method: str = "none",
        scale_outcome_treatment: bool = False,
        include_constant: bool = True,
        minmax_range: Tuple[float, float] = (-1.0, 1.0),
        K: int = 5,
        variables_to_remove: Optional[List[str]] = None,
        variables_to_fast: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the estimation arrays.

        The control matrix is
        :math:`X_t = [\\,1,\\; F_t,\\; Z_{t-1}, \\dots, Z_{t-p}\\,]` where
        :math:`F_t` collects the contemporaneous fast variables and :math:`Z`
        collects *every* transformed series (including the outcome and the
        policy variable) at lags 1 to ``p``.

        Parameters
        ----------
        df : DataFrame
            Metadata-headed wide table.
        num_lags : int
            Number of lags ``p``.  The paper uses 3.
        H : int, default 1
            Maximum horizon.  ``leads`` gets ``H + 1`` columns, ``h = 0..H``.
        treatment_var, outcome_var : str
            Original column names.
        treatment_code, outcome_code : int
            Transformation codes; **must match the sheet's ``Transform:`` row**
            for those columns, since they identify the transformed column.
        start_date, end_date : str, optional
            Inclusive date filter applied after lags and leads are built.
        scaling_method : {'none', 'l2', 'standard', 'robust', 'minmax'}
            Column scaling of the controls.  The paper's application uses
            ``'none'``.
        scale_outcome_treatment : bool, default False
            Also scale ``y`` and ``policy``.  Leave ``False`` to keep the
            estimated coefficient in the units of the data.
        include_constant : bool, default True
            Prepend a column of ones to ``X``.
        minmax_range : tuple, default (-1, 1)
        K : int, default 5
            Number of folds.  The sample is truncated at the end so that
            ``len(X) % K == 0`` and every observation lands in a main block.
        variables_to_remove, variables_to_fast : list of str, optional

        Returns
        -------
        X : ndarray, shape (T, p_features)
        y : ndarray, shape (T,)
        policy : ndarray, shape (T,)
        leads : ndarray, shape (T, H + 1)
        """
        if treatment_code is None or outcome_code is None:
            raise ValueError("treatment_code and outcome_code are required")

        transformed, _, fast_cols = self.process_dataframe(
            df, variables_to_remove, variables_to_fast)

        treat_col = f"{treatment_var}_{treatment_code}"
        out_col = f"{outcome_var}_{outcome_code}"
        for name, col in (("treatment", treat_col), ("outcome", out_col)):
            if col not in transformed.columns:
                raise KeyError(
                    f"{name} column '{col}' not found. Check that "
                    f"'{treatment_var if name == 'treatment' else outcome_var}' "
                    f"exists and that its 'Transform:' code matches the "
                    f"{name}_code you passed."
                )
        self.names_ = {"treatment": treat_col, "outcome": out_col}

        same = (treatment_var == outcome_var and treatment_code == outcome_code)

        # ---- lags of everything ------------------------------------------ #
        if num_lags > 0:
            lagged = pd.concat(
                [transformed.shift(lag).add_prefix(f"lag{lag}_")
                 for lag in range(1, num_lags + 1)],
                axis=1,
            )
        else:
            lagged = pd.DataFrame(index=transformed.index)

        # ---- contemporaneous fast block ---------------------------------- #
        contemp = transformed[fast_cols].copy()
        for col in (treat_col, out_col):
            if col in contemp.columns:
                contemp = contemp.drop(columns=[col])

        X_df = pd.concat([contemp, lagged], axis=1) if not lagged.empty else contemp
        if include_constant:
            X_df.insert(0, "cons", 1.0)

        y_series = transformed[out_col]
        policy_series = y_series.copy() if same else transformed[treat_col]

        leads_df = pd.DataFrame(index=transformed.index)
        for h in range(H + 1):
            leads_df[f"lead{h}_{out_col}"] = (
                y_series if h == 0 else y_series.shift(-h))

        y_key, d_key = "__outcome__", "__treatment__"
        parts = [y_series.to_frame(y_key)]
        if not same:
            parts.append(policy_series.to_frame(d_key))
        parts.extend([X_df, leads_df])
        combined = pd.concat(parts, axis=1)

        combined = self._scale(combined, scaling_method, scale_outcome_treatment,
                               y_key, y_key if same else d_key, minmax_range)

        # drop rows without full lags / leads, then filter dates
        combined = combined.iloc[num_lags:len(combined) - H]
        if start_date is not None:
            combined = combined[combined.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            combined = combined[combined.index <= pd.to_datetime(end_date)]

        excess = len(combined) % K
        if excess:
            combined = combined.iloc[:-excess]
        if len(combined) < 2 * K:
            raise ValueError(
                f"only {len(combined)} observations survive after "
                f"transformation, lags, leads and date filtering -- too few for "
                f"K={K} folds. Shorten num_lags/H, widen the date range, or "
                f"reduce K."
            )

        self.original_index = combined.index.copy()
        self._warn_on_calendar_gaps(self.original_index)
        combined = combined.reset_index(drop=True)

        y = combined[y_key].values.reshape(-1)
        policy = y.copy() if same else combined[d_key].values.reshape(-1)

        feature_cols = [c for c in combined.columns
                        if c not in (y_key, d_key) and not c.startswith("lead")]
        X = combined[feature_cols].values
        self.feature_names_ = feature_cols

        lead_cols = [c for c in combined.columns if c.startswith("lead")]
        leads = combined[lead_cols].values if lead_cols else np.empty((len(combined), 0))

        self.data_df = combined.copy()
        self.data_df.index = self.original_index
        return X, y, policy, leads

    # ----------------------------------------------------------- gap check -- #

    @staticmethod
    def _warn_on_calendar_gaps(index: pd.Index) -> None:
        """
        Flag holes in the retained sample.

        Blocks, lags and leads are all built *positionally*, so a missing
        period is stitched over silently: the observation after a gap is
        treated as the immediate successor of the one before it, and any lag or
        lead spanning the gap is wrong. This can come from the source data (an
        excluded episode), from a series with interior missing values that
        ``dropna`` removed, or from an irregular date index.
        """
        import warnings

        if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
            return
        deltas = np.diff(index.values).astype("timedelta64[D]").astype(float)
        typical = float(np.median(deltas))
        if typical <= 0:
            return
        breaks = [i for i, dt in enumerate(deltas) if dt > 1.75 * typical]
        if not breaks:
            return
        first = index[breaks[0]].date()
        after = index[breaks[0] + 1].date()
        warnings.warn(
            f"the retained sample is not contiguous: {len(breaks)} calendar "
            f"break(s), the first between {first} and {after}. Lags and leads "
            f"are built positionally, so those spanning a break do not "
            f"correspond to adjacent periods. Check for series with interior "
            f"missing values (they are dropped row-wise) or an excluded "
            f"episode in the source data.",
            stacklevel=3,
        )

    # ---------------------------------------------------------------- scale -- #

    @staticmethod
    def _scale(data: pd.DataFrame, method: str, scale_yd: bool,
               y_key: str, d_key: str, minmax_range) -> pd.DataFrame:
        method = (method or "none").lower()
        if method == "none":
            return data

        out = data.copy()
        protected = set() if scale_yd else {y_key, d_key}
        cols = [c for c in out.columns
                if c not in protected and not c.startswith("lead") and c != "cons"]

        block = out[cols].astype(float)
        if method == "l2":
            norms = np.sqrt((block ** 2).sum(axis=0))
            norms = norms.replace(0.0, 1.0)
            out[cols] = block / norms
        elif method == "standard":
            sd = block.std(ddof=0).replace(0.0, 1.0)
            out[cols] = (block - block.mean()) / sd
        elif method == "robust":
            iqr = (block.quantile(0.75) - block.quantile(0.25)).replace(0.0, 1.0)
            out[cols] = (block - block.median()) / iqr
        elif method == "minmax":
            lo, hi = minmax_range
            rng = (block.max() - block.min()).replace(0.0, 1.0)
            out[cols] = lo + (block - block.min()) * (hi - lo) / rng
        else:
            raise ValueError(
                f"invalid scaling_method '{method}'; choose from "
                f"'none', 'l2', 'standard', 'robust', 'minmax'"
            )
        return out
