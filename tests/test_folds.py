"""Fold construction: the RCF geometry of Section 2.1."""

import numpy as np
import pytest

from tsdml.folds import (
    fold_direction,
    nlo_folds,
    reverse_cf_folds,
    sample_use_nlo,
    sample_use_rcf,
)


def test_blocks_are_contiguous_and_disjoint():
    bs = reverse_cf_folds(60, 6)
    assert bs.block_size == 10
    seen = set()
    for blk in bs.main_blocks:
        assert blk == list(range(blk[0], blk[-1] + 1))
        assert not seen & set(blk)
        seen |= set(blk)
    assert seen == set(range(60))


def test_auxiliary_never_overlaps_the_main_block():
    bs = reverse_cf_folds(60, 6)
    for k in range(6):
        main = set(bs.main_blocks[k])
        assert not main & set(bs.aux_right_blocks[k])
        assert not main & set(bs.aux_left_blocks[k])


def test_rcf_has_no_buffer():
    """Every observation outside the main block is available as auxiliary."""
    bs = reverse_cf_folds(60, 6)
    for k in range(6):
        both = set(bs.aux_right_blocks[k]) | set(bs.aux_left_blocks[k])
        assert both == set(range(60)) - set(bs.main_blocks[k])
        assert bs.left_out_blocks[k] == []


def test_edge_folds_are_one_sided():
    bs = reverse_cf_folds(50, 5)
    assert bs.aux_left_blocks[0] == []
    assert bs.aux_right_blocks[-1] == []


@pytest.mark.parametrize("K,expected", [
    (5, ["reverse", "reverse", "both", "forward", "forward"]),
    (6, ["reverse"] * 3 + ["forward"] * 3),
    (4, ["reverse", "reverse", "forward", "forward"]),
])
def test_direction_rule(K, expected):
    assert [fold_direction(k, K) for k in range(K)] == expected


def test_only_odd_K_has_a_bidirectional_fold():
    for K in range(2, 13):
        both = [k for k in range(K) if fold_direction(k, K) == "both"]
        assert len(both) == (1 if K % 2 else 0)


def test_nlo_deletes_both_neighbours():
    main, aux = nlo_folds(50, 5)
    for k in range(5):
        forbidden = set()
        for j in (k - 1, k, k + 1):
            if 0 <= j < 5:
                forbidden |= set(main[j])
        assert not forbidden & set(aux[k])


def test_sample_use_matches_the_paper():
    """RCF wins for K=3..9, ties at 11, loses at 10 and from 12."""
    for K in range(3, 10):
        assert sample_use_rcf(K) > sample_use_nlo(K)
    assert sample_use_rcf(10) < sample_use_nlo(10)
    assert sample_use_rcf(11) == pytest.approx(sample_use_nlo(11))
    for K in range(12, 20):
        assert sample_use_rcf(K) < sample_use_nlo(K)


def test_sample_use_closed_forms():
    assert sample_use_rcf(6) == pytest.approx((3 * 6 - 2) / (4 * 6))
    assert sample_use_rcf(5) == pytest.approx(3 * (25 - 1) / (4 * 25))
    assert sample_use_nlo(6) == pytest.approx(5 * 4 / 36)


def test_rejects_degenerate_configurations():
    with pytest.raises(ValueError):
        reverse_cf_folds(20, 1)
    with pytest.raises(ValueError):
        reverse_cf_folds(3, 5)
