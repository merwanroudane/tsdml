"""Goldilocks-zone tuning (Section 3)."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso

from tsdml.calibration import Calibrator, GoldilocksZone, RMSE, goldilocks_select
from tsdml.datasets import simulate_plr


def test_declines_a_sharp_unstable_minimum():
    """A spike at index 1 loses to the flat region at 3-5."""
    profile = [0.50, 0.20, 0.60, 0.31, 0.30, 0.31]
    assert goldilocks_select(profile, window_size=3) == 4
    assert int(np.argmin(profile)) == 1


def test_agrees_with_rmse_on_a_smooth_profile():
    profile = [0.9, 0.7, 0.5, 0.35, 0.30, 0.31, 0.36, 0.5]
    assert goldilocks_select(profile, window_size=3) == int(np.argmin(profile))


def test_selection_is_inside_the_selected_window():
    rng = np.random.default_rng(0)
    for _ in range(30):
        profile = rng.random(20).tolist()
        idx = goldilocks_select(profile, window_size=3)
        assert 0 <= idx < len(profile)


def test_flat_profile_is_handled():
    assert goldilocks_select([0.5] * 7, window_size=3) == 0


def test_window_longer_than_grid_falls_back_to_argmin():
    assert goldilocks_select([0.4, 0.1, 0.3], window_size=9) == 1


def test_single_candidate():
    assert goldilocks_select([0.7]) == 0


def test_empty_profile_raises():
    with pytest.raises(ValueError):
        goldilocks_select([])


def test_metric_dispatch():
    assert isinstance(Calibrator(metric="rmse").metric, RMSE)
    assert isinstance(Calibrator(metric="goldilocks_zone").metric, GoldilocksZone)
    with pytest.raises(ValueError):
        Calibrator(metric="not-a-metric")


def test_validation_block_never_touches_the_main_block():
    """The tuning split must stay measurable w.r.t. the auxiliary sigma-field."""
    from tsdml.folds import reverse_cf_folds

    cal = Calibrator(n_blocks=6)
    blocks = reverse_cf_folds(120, 6)
    for k in range(6):
        train, val = cal._train_val_indices(k, blocks)
        main = set(blocks.main_blocks[k])
        assert not main & set(train.tolist())
        assert not main & set(val.tolist())
        assert not set(train.tolist()) & set(val.tolist())


def test_calibrate_returns_one_learner_per_fold():
    sim = simulate_plr(T=120, p=15, theta=1.0, seed=2)
    cal = Calibrator(metric="goldilocks_zone", n_blocks=5)
    out = cal.calibrate(
        sim["X"], sim["y"], sim["d"],
        outcome_learner_class=Lasso,
        outcome_param_grid={"alpha": np.linspace(0.001, 0.3, 12), "max_iter": [5000]},
        treatment_learner_class=Lasso,
        treatment_param_grid={"alpha": np.linspace(0.001, 0.3, 12), "max_iter": [5000]},
    )
    assert len(out["block_specific_learners"]["outcome_learners"]) == 5
    assert len(out["block_specific_learners"]["treatment_learners"]) == 5
    assert set(cal.rmse_profiles_) == set(range(5))
    assert all(len(v["outcome"]) == 12 for v in cal.rmse_profiles_.values())
    assert cal.selected_params_frame().shape[0] == 5
