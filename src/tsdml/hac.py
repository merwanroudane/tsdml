"""
Long-run variance estimation and critical values for RCF-DML inference.

Theorem 2.1 of Ciganovic, D'Amario and Tancioni (2026) gives

.. math::

    \\sqrt{T}(\\hat\\theta - \\theta_0) \\Rightarrow
    \\mathcal N\\!\\left(0,\\; A^{-1}\\Sigma (A^{-1})^{\\top}\\right),
    \\qquad
    \\Sigma = \\sum_{h=-\\infty}^{\\infty} \\Gamma(h),

so valid inference needs a heteroskedasticity- and autocorrelation-consistent
(HAC) estimate of the long-run variance :math:`\\Sigma` of the *stacked,
time-ordered* cross-fitted score sequence -- not of fold-wise variances.  This
module supplies:

* :func:`hac_lrv` -- Bartlett/Newey-West, Quadratic Spectral, Parzen kernels
  and the equal-weighted cosine (EWC) orthogonal-series estimator;
* :func:`compute_hac_bandwidth` -- the paper's default rule
  :math:`m = \\min(h+1, 24)` plus Andrews (1991), Newey-West (1994),
  power rules and the Lazarus, Lewis, Stock and Watson (2018) rules;
* :func:`fixed_b_critical_value` -- Kiefer and Vogelsang (2005) fixed-``b``
  critical values for the Bartlett kernel and the exact Student-``t``
  fixed-``b`` distribution of the EWC ``t``-ratio.

The defaults reproduce the paper's benchmark: Bartlett kernel with
``rule='small'``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.stats as stats

__all__ = [
    "hac_lrv",
    "compute_hac_bandwidth",
    "calculate_optimal_lag",
    "get_critical_value",
    "fixed_b_critical_value",
    "normalize_kernel",
]


# --------------------------------------------------------------------------- #
# Kernel names
# --------------------------------------------------------------------------- #

_BARTLETT_ALIASES = ("bartlett", "nw", "newey-west", "newey_west")
_QS_ALIASES = ("qs", "quadratic", "quadratic-spectral", "quadratic_spectral")
_EWC_ALIASES = ("ewc", "lls_ewc", "lls-ewc")


def normalize_kernel(kernel: str) -> str:
    """
    Map a kernel alias to one of ``'bartlett'``, ``'qs'``, ``'parzen'``, ``'ewc'``.

    Examples
    --------
    >>> normalize_kernel('Newey-West')
    'bartlett'
    >>> normalize_kernel('quadratic spectral'.replace(' ', '-'))
    'qs'
    """
    k = (kernel or "bartlett").lower()
    if k in _BARTLETT_ALIASES:
        return "bartlett"
    if k in _QS_ALIASES:
        return "qs"
    if k == "parzen":
        return "parzen"
    if k in _EWC_ALIASES:
        return "ewc"
    raise ValueError(f"Unknown HAC kernel '{kernel}'")


# --------------------------------------------------------------------------- #
# Bandwidth rules
# --------------------------------------------------------------------------- #

def calculate_optimal_lag(horizon: int = 0) -> int:
    """
    The paper's benchmark Newey-West bandwidth ``m = min(h + 1, 24)``.

    Examples
    --------
    >>> [calculate_optimal_lag(h) for h in (0, 3, 40)]
    [1, 4, 24]
    """
    return min(int(horizon) + 1, 24)


def compute_hac_bandwidth(
    rule: str = "small",
    horizon: int = 0,
    T: Optional[int] = None,
    scores: Optional[np.ndarray] = None,
    value: Optional[int] = None,
) -> int:
    """
    Truncation lag (or EWC cosine-term count) for a given selection rule.

    Parameters
    ----------
    rule : {'small', 'andrews', 'newey-west', 'pow14', 'pow15', 'lls_nw', \
'lls_ewc', 'fixed'}
        ``'small'``
            the paper's default ``min(horizon + 1, 24)``;
        ``'andrews'``
            Andrews (1991) plug-in for the Bartlett kernel using an AR(1)
            approximation of the scalar score;
        ``'newey-west'``
            Newey-West (1994) automatic ``floor(4 (T/100)^{2/9})``;
        ``'pow14'`` / ``'pow15'``
            ``floor(T^{1/4})`` / ``floor(1.3221 T^{1/5})``;
        ``'lls_nw'``
            Lazarus, Lewis, Stock and Watson (2018) ``ceil(1.3 T^{1/2})``;
        ``'lls_ewc'``
            LLSW (2018) ``floor(0.4 T^{2/3})`` cosine terms;
        ``'fixed'``
            use ``value`` directly.
    horizon : int
        Local-projection horizon, used by ``'small'``.
    T : int, optional
        Sample size; required by every rule except ``'small'`` and ``'fixed'``.
    scores : array-like, optional
        Score series used by the Andrews plug-in.
    value : int, optional
        Bandwidth when ``rule='fixed'``.

    Returns
    -------
    int
        Bandwidth, at least 1.

    Examples
    --------
    >>> compute_hac_bandwidth('small', horizon=4)
    5
    >>> compute_hac_bandwidth('newey-west', T=200)
    4
    """
    r = (rule or "small").lower()
    if r == "small":
        return calculate_optimal_lag(horizon)
    if r == "fixed":
        if value is None:
            raise ValueError("rule='fixed' requires 'value'")
        return max(1, int(value))
    if T is None:
        raise ValueError(f"rule='{r}' requires T")
    T = int(T)

    if r in ("pow14", "p14"):
        return max(1, int(np.floor(T ** 0.25)))
    if r in ("pow15", "p15"):
        return max(1, int(np.floor(1.3221 * T ** 0.2)))
    if r in ("newey-west", "newey_west", "nw1994"):
        return max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))
    if r in ("lls_nw", "lls-nw", "llsw_nw", "llsw-nw", "lazarus_nw"):
        return max(1, int(np.ceil(1.3 * T ** 0.5)))
    if r in ("lls_ewc", "lls-ewc", "llsw_ewc", "llsw-ewc", "ewc"):
        return max(1, int(np.floor(0.4 * T ** (2.0 / 3.0))))
    if r in ("andrews", "andrews1991"):
        fallback = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))
        if scores is None:
            return fallback
        s = np.asarray(scores, dtype=float).ravel()
        if s.size < 3 or np.var(s) < 1e-12:
            return fallback
        s = s - s.mean()
        denom = float(np.dot(s[:-1], s[:-1]))
        if denom <= 0:
            return fallback
        rho = float(np.clip(np.dot(s[1:], s[:-1]) / denom, -0.97, 0.97))
        alpha1 = 4.0 * rho ** 2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
        if alpha1 <= 0:
            return 1
        return max(1, int(np.floor(1.1447 * (alpha1 * T) ** (1.0 / 3.0))))
    raise ValueError(f"unknown bandwidth rule '{rule}'")


# --------------------------------------------------------------------------- #
# Long-run variance
# --------------------------------------------------------------------------- #

def hac_lrv(s, K: int = 0, kernel: str = "bartlett", bandwidth: Optional[int] = None):
    r"""
    HAC long-run variance of a time-ordered score series.

    .. math::

        \hat\Sigma = \hat\Gamma(0)
        + \sum_{h=1}^{H} k\!\left(\tfrac{h}{H+1}\right)
          \left[\hat\Gamma(h) + \hat\Gamma(h)^{\top}\right]

    with :math:`\hat\Gamma(h) = (T-K)^{-1} \sum_t s_t s_{t-h}^{\top}` on
    centered scores.

    Parameters
    ----------
    s : array-like, shape (T,) or (T, d)
        Score series in *time order*.  Centered internally.
    K : int, default 0
        Degrees-of-freedom correction subtracted from ``T`` in the
        autocovariance denominator.  The paper's RCF-DML implementation passes
        the number of folds here; keep ``K=0`` for a textbook estimator.
    kernel : {'bartlett', 'qs', 'parzen', 'ewc'}
        ``'ewc'`` selects the equal-weighted cosine estimator of Mueller (2004)
        and LLSW (2018); it is not a kernel estimator and ``bandwidth`` is then
        the number of cosine terms :math:`\nu`.
    bandwidth : int, optional
        Truncation lag ``H`` (or :math:`\nu` for EWC).  Defaults per kernel:
        ``floor(T^{1/4})`` (Bartlett), ``floor(1.3221 T^{1/5})`` (QS),
        ``floor(T^{1/5})`` (Parzen), ``floor(0.4 T^{2/3})`` (EWC).

    Returns
    -------
    float or ndarray
        Scalar when ``d == 1``, otherwise a ``(d, d)`` matrix.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> float(hac_lrv(x, kernel='bartlett')) > 0
    True
    """
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    T, d = s.shape
    s_c = s - s.mean(axis=0, keepdims=True)

    kname = (kernel or "bartlett").lower()

    # ---- Equal-weighted cosine (orthogonal series) ------------------------ #
    if kname in _EWC_ALIASES:
        nu = max(1, int(bandwidth)) if bandwidth is not None else max(
            1, int(np.floor(0.4 * T ** (2.0 / 3.0)))
        )
        nu = min(nu, max(1, T - 1))
        t_idx = np.arange(1, T + 1, dtype=float)
        j_idx = np.arange(1, nu + 1, dtype=float)
        B = np.sqrt(2.0 / T) * np.cos(np.pi * np.outer((t_idx - 0.5) / T, j_idx))
        Lambda = B.T @ s_c
        Sigma = (Lambda.T @ Lambda) / nu
        return Sigma.item() if d == 1 else Sigma

    # ---- Kernel estimators ------------------------------------------------ #
    if bandwidth is None:
        if kname in _BARTLETT_ALIASES:
            H = int(np.floor(T ** 0.25))
        elif kname in _QS_ALIASES:
            H = int(np.floor(1.3221 * T ** 0.2))
        elif kname == "parzen":
            H = int(np.floor(T ** 0.2))
        else:
            raise ValueError("kernel must be 'bartlett', 'qs', 'parzen' or 'ewc'")
    else:
        H = int(bandwidth)
    H = max(0, min(H, T - 1))

    if kname in _BARTLETT_ALIASES:
        def w(h):
            return 1.0 - h / (H + 1.0)
    elif kname in _QS_ALIASES:
        c1 = 25.0 / (12.0 * np.pi ** 2)

        def w(h):
            if h == 0:
                return 1.0
            x = h / (H + 1.0)
            a = 6.0 * np.pi * x / 5.0
            return c1 / (x ** 2) * (np.sin(a) / a - np.cos(a))
    elif kname == "parzen":
        def w(h):
            if h == 0:
                return 1.0
            x = abs(h / (H + 1.0))
            if x <= 0.5:
                return 1.0 - 6.0 * x ** 2 + 6.0 * x ** 3
            if x <= 1.0:
                return 2.0 * (1.0 - x) ** 3
            return 0.0
    else:
        raise ValueError("kernel must be 'bartlett', 'qs', 'parzen' or 'ewc'")

    Sigma = (s_c.T @ s_c) / (T - K)
    for h in range(1, H + 1):
        Gamma_h = (s_c[h:].T @ s_c[:-h]) / (T - K)
        weight = w(h)
        if weight != 0.0:
            Sigma = Sigma + weight * (Gamma_h + Gamma_h.T)

    return Sigma.item() if d == 1 else Sigma


# --------------------------------------------------------------------------- #
# Critical values
# --------------------------------------------------------------------------- #

# Kiefer and Vogelsang (2005, Econometric Theory, Table I) polynomial
# approximations cv(b) = a0 + a1 b + a2 b^2 + a3 b^3, b = S/T, Bartlett kernel,
# two-sided t-test with m = 1 restriction.
_KV2005_BARTLETT_T_COEFS = {
    0.10: (1.6449, 2.1859, 0.3142, -0.3427),
    0.05: (1.9600, 2.9694, 0.4160, -0.5324),
    0.01: (2.3263, 4.1618, 0.5368, -0.9060),
}


def fixed_b_critical_value(
    kernel: str,
    b: Optional[float] = None,
    nu: Optional[int] = None,
    m: int = 1,
    alpha: float = 0.05,
) -> Optional[float]:
    """
    Two-sided fixed-``b`` critical value for a HAR/HAC ``t``-test (``m = 1``).

    Parameters
    ----------
    kernel : {'bartlett', 'ewc'}
        For ``'bartlett'`` the Kiefer-Vogelsang (2005) polynomial in
        ``b = S/T`` is used, for two-sided ``alpha`` in ``{0.10, 0.05, 0.01}``.
        For ``'ewc'`` the fixed-``b`` distribution of the ``t``-ratio is exactly
        Student-``t`` with ``nu`` degrees of freedom.
    b : float, optional
        Bandwidth ratio ``S/T`` (Bartlett).
    nu : int, optional
        Number of cosine terms (EWC).
    m : int, default 1
        Number of restrictions; only ``m = 1`` is supported.
    alpha : float, default 0.05
        Two-sided size.

    Returns
    -------
    float or None
        ``None`` when the combination is unsupported, so callers can fall back
        to the standard normal/``t`` critical value.

    Examples
    --------
    >>> round(fixed_b_critical_value('bartlett', b=0.1, alpha=0.05), 4)
    2.2606
    >>> round(fixed_b_critical_value('ewc', nu=10, alpha=0.05), 4)
    2.2281
    """
    if m != 1:
        return None
    k = (kernel or "").lower()
    if k in _EWC_ALIASES:
        if nu is None or nu < 1:
            return None
        return float(stats.t.ppf(1.0 - alpha / 2.0, df=int(nu)))
    if k in _BARTLETT_ALIASES:
        if b is None or b <= 0:
            return None
        nearest = min(_KV2005_BARTLETT_T_COEFS, key=lambda a: abs(a - alpha))
        if abs(nearest - alpha) > 1e-3:
            return None
        c0, c1, c2, c3 = _KV2005_BARTLETT_T_COEFS[nearest]
        bb = float(min(max(b, 0.0), 1.0))
        return float(c0 + c1 * bb + c2 * bb ** 2 + c3 * bb ** 3)
    return None


def get_critical_value(
    confidence_level: float,
    n: int,
    use_fixed_critical: bool = True,
    *,
    fixed_b_kernel: Optional[str] = None,
    fixed_b_b: Optional[float] = None,
    fixed_b_nu: Optional[int] = None,
    m: int = 1,
) -> float:
    """
    Critical value used to build confidence intervals.

    Defaults to the paper's convention: Student-``t`` with ``df = n - 2`` for
    ``n < 100`` and the standard normal otherwise.  Passing ``fixed_b_kernel``
    switches to a fixed-``b`` critical value (see
    :func:`fixed_b_critical_value`).

    Examples
    --------
    >>> round(get_critical_value(0.95, 500), 4)
    1.96
    >>> round(get_critical_value(0.95, 60), 4)
    2.0017
    """
    alpha = 1.0 - confidence_level

    if fixed_b_kernel is not None and m == 1:
        cv = fixed_b_critical_value(
            kernel=fixed_b_kernel, b=fixed_b_b, nu=fixed_b_nu, m=1, alpha=alpha
        )
        if cv is not None:
            return cv

    if use_fixed_critical and n >= 100:
        return float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(stats.t.ppf(1.0 - alpha / 2.0, df=max(1, n - 2)))
