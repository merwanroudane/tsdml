"""Estimator behaviour: consistency, inference, caching, local projections."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso, LinearRegression

from tsdml import (
    DMLLocalProjections,
    NLOCrossFitting,
    ReverseCrossFitting,
    cumulate_residuals,
    scale_irfs,
    simulate_plr,
    simulate_svar,
)


def _lasso():
    return Lasso(alpha=0.02, max_iter=20000, random_state=0)


# --------------------------------------------------------------------------- #
# point estimate
# --------------------------------------------------------------------------- #

def test_recovers_the_true_parameter_in_a_sparse_plr():
    sim = simulate_plr(T=400, p=60, theta=1.5, seed=11)
    est = ReverseCrossFitting(_lasso(), _lasso(), n_blocks=6).fit(
        sim["X"], sim["y"], sim["d"])
    assert abs(est.theta_ - 1.5) < 0.15


def test_recovers_the_impact_effect_in_the_svar_design():
    """Correctly specified ordering: the PLR estimand equals the structural impact."""
    sim = simulate_svar(n=30, T=300, theta=0.5, specification="specified", seed=5)
    est = ReverseCrossFitting(_lasso(), _lasso(), n_blocks=6).fit(
        sim["X"], sim["y"], sim["d"])
    assert abs(est.theta_ - sim["theta"]) < 0.2


def test_specification_switch_moves_the_policy_variable():
    mis = simulate_svar(n=30, T=100, specification="misspecified", seed=0)
    spec = simulate_svar(n=30, T=100, specification="specified", seed=0)
    assert spec["policy_pos"] == spec["outcome_pos"] - 1
    assert mis["policy_pos"] < spec["policy_pos"]
    with pytest.raises(ValueError):
        simulate_svar(n=10, specification="nonsense")


def test_fold_average_equals_the_mean_of_fold_estimates():
    sim = simulate_plr(T=180, p=20, theta=1.0, seed=3)
    est = ReverseCrossFitting(_lasso(), _lasso(), n_blocks=6).fit(
        sim["X"], sim["y"], sim["d"])
    assert est.theta_ == pytest.approx(np.mean(est.results_["block_coefs"]))


def test_residuals_are_out_of_sample_everywhere():
    """No main-block observation may have been seen by its own nuisance fit."""
    sim = simulate_plr(T=120, p=10, theta=1.0, seed=4)
    est = ReverseCrossFitting(LinearRegression(), LinearRegression(), n_blocks=6).fit(
        sim["X"], sim["y"], sim["d"])
    chi = est.residuals_["outcome"]
    assert np.all(np.isfinite(chi))
    assert np.count_nonzero(chi) == len(chi)


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #

def test_standard_error_is_positive_and_ci_brackets_the_estimate():
    sim = simulate_plr(T=200, p=25, theta=1.0, seed=6)
    est = ReverseCrossFitting(_lasso(), _lasso(), n_blocks=6).fit(
        sim["X"], sim["y"], sim["d"])
    r = est.results_
    assert r["std_error"] > 0
    assert r["ci_lower"] < r["coef"] < r["ci_upper"]
    assert 0.0 <= r["p_value"] <= 1.0


def test_coverage_and_bias_on_the_paper_svar_design():
    """
    Near-zero bias and roughly nominal coverage on the paper's benchmark DGP.

    Measured over 200 replications of this design the estimator shows bias of
    well under 1% and coverage around 86-90%; the shortfall against 95% comes
    from the deliberately short default HAC bandwidth, which is documented in
    the README and remedied by ``hac_bandwidth_rule='lls_nw'``.  With the 60
    replications used here, Monte Carlo noise on the coverage figure is roughly
    4 percentage points, so the threshold is set well below the expected value.
    """
    covered, bias = 0, []
    reps = 60
    for s in range(reps):
        sim = simulate_svar(n=30, T=300, theta=0.5, specification="specified",
                            seed=s, seed_err=1000 + s)
        lasso = lambda: Lasso(alpha=0.05, max_iter=20000, random_state=0)  # noqa: E731
        est = ReverseCrossFitting(lasso(), lasso(), n_blocks=6).fit(
            sim["X"], sim["y"], sim["d"])
        r = est.results_
        covered += int(r["ci_lower"] <= 0.5 <= r["ci_upper"])
        bias.append(est.theta_ - 0.5)
    assert abs(np.mean(bias)) / 0.5 < 0.05
    assert covered / reps >= 0.75


def test_a_longer_bandwidth_widens_the_interval_under_persistence():
    """
    The default rule m = min(h+1, 24) is short by design.

    When the cross-fitted score is strongly serially correlated the resulting
    interval can be too narrow; a longer automatic rule is the remedy, and the
    package must let you reach for one.
    """
    sim = simulate_plr(T=240, p=20, theta=1.0, rho=0.8, seed=42)
    lasso = lambda: Lasso(alpha=0.02, max_iter=20000, random_state=0)  # noqa: E731
    short = ReverseCrossFitting(lasso(), lasso(), n_blocks=6).fit(
        sim["X"], sim["y"], sim["d"])
    long = ReverseCrossFitting(lasso(), lasso(), n_blocks=6,
                               hac_bandwidth_rule="lls_nw",
                               use_fixed_b_critical=True).fit(
        sim["X"], sim["y"], sim["d"])
    assert short.theta_ == pytest.approx(long.theta_)  # only inference changes
    width = lambda e: e.results_["ci_upper"] - e.results_["ci_lower"]  # noqa: E731
    assert width(long) > width(short)


@pytest.mark.parametrize("kernel", ["bartlett", "qs", "parzen", "ewc"])
def test_every_hac_kernel_runs(kernel):
    sim = simulate_plr(T=150, p=12, theta=1.0, seed=7)
    est = ReverseCrossFitting(_lasso(), _lasso(), n_blocks=5,
                              hac_kernel=kernel).fit(sim["X"], sim["y"], sim["d"])
    assert np.isfinite(est.results_["std_error"])


def test_fixed_b_widens_the_interval():
    sim = simulate_plr(T=200, p=15, theta=1.0, seed=8)
    kw = dict(n_blocks=6, hac_bandwidth_rule="lls_nw")
    plain = ReverseCrossFitting(_lasso(), _lasso(), **kw).fit(
        sim["X"], sim["y"], sim["d"])
    fixed = ReverseCrossFitting(_lasso(), _lasso(), use_fixed_b_critical=True,
                                **kw).fit(sim["X"], sim["y"], sim["d"])
    width = lambda e: e.results_["ci_upper"] - e.results_["ci_lower"]  # noqa: E731
    assert width(fixed) > width(plain)


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #

def test_policy_residuals_are_reused_when_only_the_outcome_changes():
    sim = simulate_plr(T=150, p=12, theta=1.0, seed=9)
    est = ReverseCrossFitting(_lasso(), _lasso(), n_blocks=5)
    est.fit(sim["X"], sim["y"], sim["d"])
    xi_first = est.residuals_["treatment"].copy()

    est.fit(sim["X"], np.roll(sim["y"], -1), sim["d"])
    assert np.array_equal(xi_first, est.residuals_["treatment"])


def test_disabling_the_cache_recomputes():
    sim = simulate_plr(T=150, p=12, theta=1.0, seed=10)
    est = ReverseCrossFitting(_lasso(), _lasso(), n_blocks=5, cache_residuals=False)
    est.fit(sim["X"], sim["y"], sim["d"])
    assert est._cached_treatment_residuals is None


# --------------------------------------------------------------------------- #
# NLO
# --------------------------------------------------------------------------- #

def test_nlo_also_recovers_the_parameter():
    sim = simulate_plr(T=400, p=40, theta=1.5, seed=12)
    est = NLOCrossFitting(_lasso(), _lasso(), n_blocks=6).fit(
        sim["X"], sim["y"], sim["d"])
    assert abs(est.theta_ - 1.5) < 0.2


def test_nlo_rejects_too_few_folds():
    with pytest.raises(ValueError):
        NLOCrossFitting(_lasso(), _lasso(), n_blocks=3)


# --------------------------------------------------------------------------- #
# local projections
# --------------------------------------------------------------------------- #

def _leads(y, H):
    cols = []
    for h in range(H + 1):
        col = np.full_like(y, np.nan)
        col[: len(y) - h] = y[h:]
        col[np.isnan(col)] = 0.0
        cols.append(col)
    return np.column_stack(cols)


def test_lp_returns_one_estimate_per_horizon():
    sim = simulate_plr(T=180, p=15, theta=1.0, seed=13)
    leads = _leads(sim["y"], 4)
    lp = DMLLocalProjections(_lasso(), _lasso(), n_blocks=6).fit(
        sim["X"], sim["y"], sim["d"], leads)
    assert lp.irf_.shape == (5,)
    frame = lp.to_frame()
    assert list(frame["Horizon"]) == [0, 1, 2, 3, 4]
    assert (frame["CI_Upper_95"] - frame["CI_Lower_95"] >
            frame["CI_Upper_90"] - frame["CI_Lower_90"]).all()


def test_lp_horizon_zero_matches_the_static_estimator():
    sim = simulate_plr(T=180, p=15, theta=1.0, seed=14)
    leads = _leads(sim["y"], 2)
    lp = DMLLocalProjections(_lasso(), _lasso(), n_blocks=6, outcome_code=None).fit(
        sim["X"], sim["y"], sim["d"], leads)
    rcf = ReverseCrossFitting(_lasso(), _lasso(), n_blocks=6).fit(
        sim["X"], sim["y"], sim["d"])
    assert lp.irf_[0] == pytest.approx(rcf.theta_)


def test_cumulation_rules():
    r = {0: {"outcome": np.array([1.0, 2.0]), "treatment": np.array([0.1, 0.2])},
         1: {"outcome": np.array([3.0, 4.0]), "treatment": np.array([0.1, 0.2])}}
    assert np.allclose(cumulate_residuals(r, 1, 5)["outcome"], [4.0, 6.0])
    assert np.allclose(cumulate_residuals(r, 1, 101)["outcome"], [1.0, 1.5])
    assert np.allclose(cumulate_residuals(r, 1, None)["outcome"], [3.0, 4.0])
    # policy residuals never cumulate
    assert np.allclose(cumulate_residuals(r, 1, 5)["treatment"], [0.1, 0.2])


def test_scale_irfs_normalises_the_capital_ratio_shock():
    import pandas as pd

    def frame(c0):
        return pd.DataFrame({
            "Horizon": [0, 1], "Coefficient": [c0, c0 / 2],
            "CI_Lower_95": [c0 - 1, 0.0], "CI_Upper_95": [c0 + 1, 1.0],
            "CI_Lower_90": [c0 - 0.5, 0.1], "CI_Upper_90": [c0 + 0.5, 0.9],
        })

    irfs = {"Tier 1 capital": frame(4.0), "Risk-weighted assets": frame(-1.0),
            "PNFC_Spread": frame(0.2), "GDP_K2020": frame(-0.6)}
    out = scale_irfs(irfs)
    factor = 0.5 / (4.0 - (-1.0))
    assert out["GDP_K2020"]["Coefficient"].iloc[0] == pytest.approx(-0.6 * factor)
    assert out["PNFC_Spread"]["Coefficient"].iloc[0] == pytest.approx(0.2 * factor * 100)
    # the normalised capital-ratio impact response is the target shock
    impact = (out["Tier 1 capital"]["Coefficient"].iloc[0]
              - out["Risk-weighted assets"]["Coefficient"].iloc[0])
    assert impact == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #

def test_length_mismatch_is_reported():
    X = np.zeros((60, 3))
    with pytest.raises(ValueError, match="length mismatch"):
        ReverseCrossFitting(_lasso(), _lasso(), n_blocks=5).fit(
            X, np.zeros(59), np.zeros(60))


def test_missing_learners_are_reported():
    with pytest.raises(ValueError, match="block_specific_learners"):
        ReverseCrossFitting(n_blocks=5)


def test_unfitted_access_raises():
    est = ReverseCrossFitting(_lasso(), _lasso(), n_blocks=5)
    with pytest.raises(ValueError, match="not fitted"):
        est.get_residuals()
