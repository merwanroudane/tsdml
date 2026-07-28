"""
Bundled data and the paper's benchmark simulation design.

Real data
---------
:func:`load_macroprudential` returns the quarterly Italian dataset behind
Section 5 of the paper: 37 macro-financial and banking series, 2005Q2-2024Q4,
assembled from official sources (IMF Financial Soundness Indicators, ISTAT,
Bank of Italy, ECB/Refinitiv market data).  The policy variable is the Tier 1
capital-to-risk-weighted-assets ratio; outcomes are Tier 1 capital,
risk-weighted assets, the PNFC lending spread, PNFC lending and GDP.

See ``SOURCES.md`` next to the CSV for the per-series provenance.

Simulated data
--------------
:func:`simulate_svar` reproduces the approximately sparse recursive SVAR of
eq. (4.13),

.. math::

    Y_t = \\mu + \\Phi_1 Y_{t-1} + \\varepsilon_t, \\qquad
    \\varepsilon_t = P u_t, \\quad u_t \\sim \\mathcal N(0, I_n),

with :math:`P` lower triangular so that the contemporaneous impact of
structural shock ``i`` on variable ``j`` is exactly :math:`P_{ji}` -- which is
what makes the true causal parameter known and lets you check coverage.
"""

from __future__ import annotations

from importlib import resources
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "load_macroprudential",
    "macroprudential_spec",
    "simulate_svar",
    "simulate_plr",
]


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #

def load_macroprudential(as_frame: bool = True) -> pd.DataFrame:
    """
    Load the Italian macroprudential dataset used in Section 5.

    Parameters
    ----------
    as_frame : bool, default True
        Kept for API symmetry; the table is always returned as a
        :class:`pandas.DataFrame` because the metadata rows are part of the
        contract with :class:`tsdml.prep.DataProcessor`.

    Returns
    -------
    DataFrame
        Indexed by ``'speed'``, ``'Transform:'`` and then quarterly dates, with
        one column per variable -- exactly the layout
        :meth:`tsdml.prep.DataProcessor.data_prep` expects.

    Examples
    --------
    >>> df = load_macroprudential()
    >>> df.loc['speed', 'GrossYield_BTP10']
    'fast'
    >>> df.shape[1]
    37
    """
    with resources.files("tsdml.data").joinpath("data_macroprudential.csv").open("r") as fh:
        raw = pd.read_csv(fh)
    raw.index = raw.iloc[:, 0]
    raw = raw.iloc[:, 1:]
    raw.index.name = None
    return raw


def macroprudential_spec() -> Dict[str, object]:
    """
    The paper's Section 5 specification, ready to unpack.

    Returns
    -------
    dict
        ``treatment_var``, ``treatment_code``, ``outcomes`` (list of
        ``(name, code)``), ``drop`` (variables excluded to avoid mechanical
        collinearity with the policy ratio), ``num_lags``, ``H``, ``n_blocks``,
        ``start_date``, ``labels`` and ``ylabels`` for plotting.

    Examples
    --------
    >>> spec = macroprudential_spec()
    >>> spec['n_blocks'], spec['num_lags'], spec['H']
    (6, 3, 8)
    """
    return {
        "treatment_var": "Tier 1 capital to risk-weighted assets_Percent",
        "treatment_code": 2,
        "outcomes": [
            ("Tier 1 capital", 5),
            ("Risk-weighted assets", 5),
            ("PNFC_Spread", 2),
            ("PNFC_Lending_K2020", 5),
            ("GDP_K2020", 5),
        ],
        "drop": [
            "Tier 1 capital_to_total_Assets_Percent",
            "Specific provisions",
            "NPLs",
            "Total assets",
            "Regulatory capital to risk-weighted assets_Percent",
            "Total regulatory capital",
        ],
        "num_lags": 3,
        "H": 8,
        "n_blocks": 6,
        "start_date": "2005-12-31",
        "labels": {
            "Tier 1 capital": "Tier 1 Capital",
            "Risk-weighted assets": "Risk-weighted Assets",
            "PNFC_Spread": "PNFC Spread",
            "PNFC_Lending_K2020": "PNFC Lending",
            "GDP_K2020": "GDP",
        },
        "ylabels": {
            "Tier 1 capital": "Percentage change",
            "Risk-weighted assets": "Percentage change",
            "PNFC_Spread": "Basis points",
            "PNFC_Lending_K2020": "Percentage change",
            "GDP_K2020": "Percentage change",
        },
    }


# --------------------------------------------------------------------------- #
# Simulation: recursive SVAR (paper eq. 4.13)
# --------------------------------------------------------------------------- #

def _banded_phi(n: int, bandwidth: int = 6, strength: float = 0.12,
                thresh: float = 1e-4, seed: int = 0) -> np.ndarray:
    """Approximately sparse ``Phi_1`` with asymmetric band-limited decay."""
    rng = np.random.default_rng(seed)
    Phi = np.zeros((n, n))
    for j in range(n):
        for i in range(max(0, j - bandwidth), min(n, j + bandwidth + 1)):
            d = i - j
            if d > 0:
                val = strength / (1.0 + abs(d))
            elif d < 0:
                val = 0.6 * strength / (1.0 + abs(d) ** 1.8)
            else:
                val = strength
            val *= 0.8 + 0.4 * rng.random()
            if abs(val) >= thresh:
                Phi[i, j] = val
    return Phi


def _stabilize(Phi: np.ndarray, target_radius: float = 0.95) -> np.ndarray:
    radius = float(np.max(np.abs(np.linalg.eigvals(Phi))))
    if radius == 0:
        return Phi
    return Phi * min(1.0, target_radius / radius)


def _lower_tri_P(n: int, diag: float = 1.0, off_strength: float = 0.7,
                 decay: float = 2.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = np.eye(n) * diag
    for i in range(n):
        for j in range(i):
            P[i, j] = (off_strength / (1.0 + (i - j) ** decay)) * (rng.random() * 1.2 + 0.4)
    return P


def simulate_svar(
    n: int = 100,
    T: int = 200,
    burn: int = 300,
    theta: float = 0.5,
    specification: str = "misspecified",
    outcome_pos: Optional[int] = None,
    policy_pos: Optional[int] = None,
    seed: int = 123,
    seed_err: int = 456,
    phi_bandwidth: int = 6,
    phi_strength: float = 0.12,
    phi_radius: float = 0.95,
    P_off_strength: float = 0.7,
    P_decay: float = 2.0,
) -> Dict[str, object]:
    """
    Simulate the benchmark recursive SVAR of eq. (4.13).

    The contemporaneous impact of the policy shock on the outcome is pinned to
    ``theta`` by fixing one entry of the lower-triangular impact matrix.

    .. warning::

       ``theta`` is the **structural impact coefficient**, and the PLR estimand
       is guaranteed to coincide with it only under
       ``specification='specified'``.  In the paper's mis-specified benchmark
       the policy variable sits in the middle of the recursive ordering while
       the estimated PLR still conditions on the full contemporaneous control
       set -- which therefore includes variables ordered *downstream* of the
       policy.  Section 4 is explicit that this design "should be interpreted as
       a misspecification stress test rather than exact recovery of the
       structural target".  Any gap between :math:`\\hat\\theta` and ``theta``
       there mixes estimator bias with an estimand wedge whose size depends on
       the ordering and on :math:`\\Phi_1`; measure bias against
       ``specification='specified'``.

    Parameters
    ----------
    n : int, default 100
        Number of variables in the system.
    T : int, default 200
        Sample length after burn-in.
    burn : int, default 300
        Burn-in draws discarded.
    theta : float, default 0.5
        Structural impact effect :math:`P_{ji}` of the policy shock on the
        outcome.
    specification : {'misspecified', 'specified'}, default 'misspecified'
        ``'misspecified'``
            policy variable two thirds along the ordering -- the paper's
            benchmark stress test;
        ``'specified'``
            policy variable immediately before the outcome, so the
            contemporaneous control set is exactly the relevant one and the PLR
            coefficient equals ``theta``.  Use this when you want to measure
            bias or coverage against a known target.
    outcome_pos, policy_pos : int, optional
        Explicit positions, overriding ``specification``.  The outcome defaults
        to last, so it never enters the policy equation contemporaneously.
    seed, seed_err : int
        Seeds for the coefficient matrices and for the innovations.
    phi_bandwidth, phi_strength, phi_radius : float
        Shape and stability of :math:`\\Phi_1`.
    P_off_strength, P_decay : float
        Off-diagonal strength and distance decay of :math:`P`.

    Returns
    -------
    dict
        ``X`` (controls: contemporaneous system plus one lag, excluding the
        outcome and the policy variable), ``y``, ``d``, ``theta``, ``Y`` (the
        raw panel), ``Phi1``, ``P``, ``spectral_radius``, ``specification``.

    Examples
    --------
    >>> sim = simulate_svar(n=20, T=120, theta=0.5, seed=0)
    >>> sim['X'].shape[0], sim['theta']
    (119, 0.5)
    >>> spec = simulate_svar(n=20, T=120, specification='specified', seed=0)
    >>> spec['policy_pos'] == spec['outcome_pos'] - 1
    True
    """
    if specification not in ("misspecified", "specified"):
        raise ValueError("specification must be 'misspecified' or 'specified'")
    if outcome_pos is None:
        outcome_pos = n - 1
    if policy_pos is None:
        policy_pos = (outcome_pos - 1 if specification == "specified"
                      else max(0, (2 * n) // 3))
    if policy_pos > outcome_pos:
        raise ValueError(
            "policy_pos must be at or before outcome_pos: the recursive "
            "ordering makes P lower triangular, so a shock can only hit "
            "variables ordered after it"
        )

    rng_err = np.random.default_rng(seed_err)

    Phi1 = _stabilize(
        _banded_phi(n, phi_bandwidth, phi_strength, seed=seed), phi_radius)
    P = _lower_tri_P(n, off_strength=P_off_strength, decay=P_decay, seed=seed + 1)
    P[outcome_pos, policy_pos] = theta

    U = rng_err.standard_normal((T + burn, n))
    E = U @ P.T
    Y = np.zeros((T + burn, n))
    for t in range(1, T + burn):
        Y[t] = Phi1 @ Y[t - 1] + E[t]
    Y = Y[burn:]

    y = Y[1:, outcome_pos]
    d = Y[1:, policy_pos]
    contemp = np.delete(Y[1:], [outcome_pos, policy_pos], axis=1)
    lagged = Y[:-1]
    X = np.column_stack([contemp, lagged])

    return {
        "X": X,
        "y": y,
        "d": d,
        "theta": float(theta),
        "Y": Y,
        "Phi1": Phi1,
        "P": P,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(Phi1)))),
        "outcome_pos": outcome_pos,
        "policy_pos": policy_pos,
        "specification": specification,
    }


def simulate_plr(
    T: int = 200,
    p: int = 100,
    theta: float = 1.0,
    rho: float = 0.7,
    sparsity: int = 5,
    signal: float = 0.8,
    resid_persistence: float = 0.0,
    seed: int = 0,
) -> Dict[str, object]:
    """
    A small, transparent partially linear time-series design.

    Controls follow a stationary AR(1) vector process, and both nuisance
    functions are sparse linear indices:

    .. math::

        d_t = X_t' \\beta_d + \\xi_t, \\qquad
        y_t = \\theta d_t + X_t' \\beta_y + \\epsilon_t .

    Persistence lives in :math:`X_t`, which is what makes the problem a time
    series; by default the disturbances are white noise, so **conditional
    stability holds**: once :math:`X_t` is conditioned on, adjacent auxiliary
    blocks carry no information about :math:`\\xi_t` or :math:`\\epsilon_t`.

    Set ``resid_persistence > 0`` to give the disturbances their own AR(1)
    dynamics.  That deliberately breaks Assumption 2.4 -- lagged residuals then
    predict current ones after conditioning -- and is the design to use when you
    want :func:`tsdml.diagnostics.boundary_leakage_test` to fire.

    Parameters
    ----------
    T : int, default 200
        Sample length.
    p : int, default 100
        Number of controls.
    theta : float, default 1.0
        True causal parameter.
    rho : float, default 0.7
        Persistence of each control.
    sparsity : int, default 5
        Number of non-zero loadings in each nuisance index.
    signal : float, default 0.8
        Size of the non-zero loadings.
    resid_persistence : float, default 0.0
        AR(1) coefficient of :math:`\\xi_t` and :math:`\\epsilon_t`.  Zero keeps
        conditional stability; positive values violate it.
    seed : int, optional

    Returns
    -------
    dict
        ``X``, ``y``, ``d``, ``theta``, ``beta_y``, ``beta_d``.

    Examples
    --------
    >>> sim = simulate_plr(T=150, p=30, theta=1.0, seed=1)
    >>> sim['X'].shape
    (150, 30)
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((T, p))
    innov = rng.standard_normal((T, p)) * np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        X[t] = rho * X[t - 1] + innov[t]

    beta_d = np.zeros(p)
    beta_y = np.zeros(p)
    beta_d[:sparsity] = signal
    beta_y[:sparsity] = signal * np.linspace(1.0, 0.5, sparsity)

    r = float(resid_persistence)
    if r == 0.0:
        xi = rng.standard_normal(T)
        eps = rng.standard_normal(T)
    else:
        scale = np.sqrt(1 - r ** 2)
        e1 = rng.standard_normal(T) * scale
        e2 = rng.standard_normal(T) * scale
        xi, eps = np.zeros(T), np.zeros(T)
        for t in range(1, T):
            xi[t] = r * xi[t - 1] + e1[t]
            eps[t] = r * eps[t - 1] + e2[t]

    d = X @ beta_d + xi
    y = theta * d + X @ beta_y + eps

    return {"X": X, "y": y, "d": d, "theta": float(theta),
            "beta_y": beta_y, "beta_d": beta_d}
