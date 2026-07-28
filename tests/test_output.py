"""Figures, tables, HAC utilities and diagnostics."""

import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tsdml import (  # noqa: E402
    boundary_leakage_test,
    compute_hac_bandwidth,
    estimation_table,
    fixed_b_critical_value,
    hac_lrv,
    irf_table,
    plot_block_structure,
    plot_goldilocks_profile,
    plot_irf,
    plot_irf_comparison,
    plot_irf_panel,
    plot_residuals,
    plot_sample_use,
    residual_autocorrelation_test,
    sample_use_table,
    to_latex,
)


@pytest.fixture
def irf_frame():
    h = np.arange(9)
    coef = np.array([0.1, 0.05, -0.02, -0.08, -0.10, -0.07, -0.03, 0.01, 0.04])
    se = np.full(9, 0.04)
    return pd.DataFrame({
        "Horizon": h, "Coefficient": coef, "Std_Error": se,
        "t_stat": np.abs(coef / se), "p_value": np.full(9, 0.03),
        "CI_Lower_95": coef - 1.96 * se, "CI_Upper_95": coef + 1.96 * se,
        "CI_Lower_90": coef - 1.645 * se, "CI_Upper_90": coef + 1.645 * se,
    })


# --------------------------------------------------------------------------- #
# HAC
# --------------------------------------------------------------------------- #

def test_lrv_of_white_noise_is_near_its_variance():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    assert hac_lrv(x, kernel="bartlett") == pytest.approx(1.0, abs=0.2)


def test_lrv_grows_with_positive_persistence():
    rng = np.random.default_rng(1)
    e = rng.standard_normal(4000)
    ar = np.zeros(4000)
    for t in range(1, 4000):
        ar[t] = 0.7 * ar[t - 1] + e[t]
    assert hac_lrv(ar, kernel="bartlett", bandwidth=20) > 3 * np.var(ar) / 2


def test_lrv_is_positive_for_every_kernel():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500)
    for kernel in ("bartlett", "qs", "parzen", "ewc"):
        assert hac_lrv(x, kernel=kernel) > 0


def test_multivariate_lrv_is_square_and_symmetric():
    rng = np.random.default_rng(3)
    S = hac_lrv(rng.standard_normal((300, 3)), kernel="bartlett", bandwidth=4)
    assert S.shape == (3, 3)
    assert np.allclose(S, S.T)


def test_paper_default_bandwidth_rule():
    assert [compute_hac_bandwidth("small", horizon=h) for h in (0, 5, 30)] == [1, 6, 24]


def test_bandwidth_rules_need_T():
    with pytest.raises(ValueError):
        compute_hac_bandwidth("andrews")
    with pytest.raises(ValueError):
        compute_hac_bandwidth("nonsense", T=100)


def test_fixed_b_exceeds_the_normal_critical_value():
    assert fixed_b_critical_value("bartlett", b=0.2, alpha=0.05) > 1.96
    assert fixed_b_critical_value("ewc", nu=8, alpha=0.05) > 1.96
    assert fixed_b_critical_value("qs", b=0.2) is None


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

def test_leakage_test_does_not_reject_on_white_noise():
    rng = np.random.default_rng(4)
    out = boundary_leakage_test(rng.standard_normal(300), n_blocks=6, max_lag=2)
    pooled = out.loc[out["fold"] == "pooled", "p_value"]
    assert len(pooled) == 1
    assert pooled.iloc[0] > 0.05


def test_leakage_test_rejects_on_a_strongly_dependent_series():
    rng = np.random.default_rng(5)
    e = rng.standard_normal(300)
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    out = boundary_leakage_test(ar, n_blocks=6, max_lag=2)
    assert out.loc[out["fold"] == "pooled", "p_value"].iloc[0] < 0.05


def test_leakage_test_tracks_conditional_stability():
    """
    Silent when Assumption 2.4 holds, loud when it is broken on purpose.

    ``simulate_plr`` puts the persistence in X by default, so the disturbances
    are innovations and conditional stability holds.  ``resid_persistence``
    gives them their own AR(1) dynamics, which is exactly the failure the
    diagnostic is meant to catch.
    """
    from sklearn.linear_model import Lasso

    from tsdml import ReverseCrossFitting, simulate_plr

    def pooled_p(resid_persistence, seed):
        sim = simulate_plr(T=240, p=20, theta=1.0, rho=0.7,
                           resid_persistence=resid_persistence, seed=seed)
        lasso = lambda: Lasso(alpha=0.02, max_iter=20000, random_state=0)  # noqa: E731
        est = ReverseCrossFitting(lasso(), lasso(), n_blocks=6).fit(
            sim["X"], sim["y"], sim["d"])
        out = boundary_leakage_test(est.residuals_["treatment"], 6, max_lag=2)
        return out.loc[out["fold"] == "pooled", "p_value"].iloc[0]

    clean = [pooled_p(0.0, s) for s in range(6)]
    broken = [pooled_p(0.7, s) for s in range(6)]
    assert sum(p < 0.05 for p in clean) <= 1     # at most one false positive
    assert sum(p < 0.05 for p in broken) >= 4    # fires on most draws


def test_ljung_box_columns():
    rng = np.random.default_rng(6)
    out = residual_autocorrelation_test(rng.standard_normal(200), lags=5)
    assert list(out.columns) == ["lag", "lb_stat", "lb_pvalue"]
    assert len(out) == 5


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def test_irf_table_has_stars_and_ci(irf_frame):
    tbl = irf_table(irf_frame)
    assert "95% CI" in tbl.columns
    assert tbl["Estimate"].iloc[0].endswith("**")
    # standard errors in parentheses beneath each estimate
    assert tbl["Estimate"].iloc[1].startswith("(")


def test_irf_table_without_se_below(irf_frame):
    tbl = irf_table(irf_frame, se_below=False)
    assert len(tbl) == len(irf_frame)
    assert "Std. error" in tbl.columns


def test_estimation_table_layout():
    res = {"coef": 1.02, "std_error": 0.11, "t_stat": 9.3, "p_value": 0.001,
           "ci_lower": 0.80, "ci_upper": 1.24, "n_blocks": 6}
    tbl = estimation_table({"RCF": res, "NLO": res})
    assert list(tbl.columns) == ["RCF", "NLO"]
    assert tbl.loc["theta", "RCF"].endswith("***")


def test_sample_use_table_flags_the_tie():
    tbl = sample_use_table([6, 11, 12])
    assert tbl.loc[tbl["K"] == 11, "Winner"].iloc[0] == "tie"
    assert tbl.loc[tbl["K"] == 6, "Winner"].iloc[0] == "RCF"


def test_latex_export_is_booktabs(tmp_path, irf_frame):
    path = tmp_path / "tab.tex"
    tex = to_latex(irf_table(irf_frame), caption="IRFs", label="tab:irf",
                   notes="HAC standard errors.", save_path=str(path))
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert "\\caption{IRFs}" in tex and "\\label{tab:irf}" in tex
    assert "Notes:" in tex
    assert path.read_text(encoding="utf-8") == tex


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def test_single_irf_figure(tmp_path, irf_frame):
    path = tmp_path / "irf.pdf"
    fig = plot_irf(irf_frame, title="GDP", save_path=str(path))
    assert path.exists()
    assert len(fig.axes) == 1


def test_panel_figure_handles_a_ragged_last_row(tmp_path, irf_frame):
    irfs = {name: irf_frame for name in ("A", "B", "C", "D", "E")}
    path = tmp_path / "panel.pdf"
    fig = plot_irf_panel(irfs, layout=(2, 3), highlight="E", save_path=str(path))
    assert path.exists()
    assert len(fig.axes) == 5


def test_panel_rejects_a_layout_that_is_too_small(irf_frame):
    with pytest.raises(ValueError, match="holds"):
        plot_irf_panel({str(i): irf_frame for i in range(7)}, layout=(2, 3))


def test_panel_reports_a_missing_variable(irf_frame):
    with pytest.raises(KeyError):
        plot_irf_panel({"A": irf_frame}, order=["A", "B"], layout=(1, 2))


def test_missing_columns_are_reported_clearly():
    with pytest.raises(KeyError, match="Horizon"):
        plot_irf(pd.DataFrame({"x": [1, 2]}))


def test_comparison_figure(irf_frame):
    other = irf_frame.copy()
    other["Coefficient"] *= 0.5
    fig = plot_irf_comparison({"Goldilocks": irf_frame, "RMSE": other})
    assert len(fig.axes[0].lines) >= 2


def test_block_structure_figure(tmp_path):
    path = tmp_path / "fig1.pdf"
    fig = plot_block_structure(K=5, save_path=str(path))
    assert path.exists()
    # K x K grid plus the whole-sample band
    assert len([p for p in fig.axes[0].patches]) >= 5 * 5


def test_goldilocks_figure_marks_both_optima():
    fig = plot_goldilocks_profile([0.5, 0.2, 0.6, 0.31, 0.30, 0.31], window_size=3)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("selected" in lab for lab in labels)
    assert any("minimiser" in lab for lab in labels)


def test_sample_use_figure():
    fig = plot_sample_use(range(3, 13))
    assert len(fig.axes[0].lines) == 2


def test_residual_figure_marks_fold_boundaries():
    rng = np.random.default_rng(7)
    resid = {"outcome": rng.standard_normal(60), "treatment": rng.standard_normal(60)}
    fig = plot_residuals(resid, n_blocks=6)
    assert len(fig.axes) == 2
