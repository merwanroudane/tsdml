"""
Block construction for time-series cross-fitting.

Two designs are provided:

* **Reverse Cross-Fitting (RCF)** -- Ciganovic, D'Amario and Tancioni (2026),
  Section 2.1.  The sample is cut into ``K`` adjacent, non-overlapping blocks
  :math:`\\{B_k\\}_{k=1}^{K}` of length :math:`T_{block} = \\lfloor T/K \\rfloor`.
  For block ``k`` the auxiliary sample is *one side only*:

  .. math::

      B_k^L = \\bigcup_{i<k} B_i, \\qquad B_k^R = \\bigcup_{i>k} B_i

  and **no buffer blocks are deleted** between the auxiliary and main samples.
  Early folds train on the right side in *time-reversed* order (licensed by
  time reversibility of stationary Gaussian processes), late folds train on
  the left side in forward order, and -- for odd ``K`` -- the central fold
  uses both sides and averages the two residual vectors.

* **Neighbors-Left-Out (NLO)** -- Semenova et al. (2023).  The auxiliary
  sample for block ``k`` drops block ``k`` *and its two adjacent neighbours*,
  which buys approximate independence at the cost of sample use.

The fold builders here reproduce the replication code of the paper
bit-for-bit, including the ``n // K`` truncation of the final partial block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

__all__ = [
    "BlockStructure",
    "reverse_cf_folds",
    "nlo_folds",
    "fold_direction",
    "sample_use_rcf",
    "sample_use_nlo",
]


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #

@dataclass
class BlockStructure:
    """
    Result of a time-series fold construction.

    Attributes
    ----------
    main_blocks : list of list of int
        Main (test) block indices :math:`B_k`, one list per fold.
    aux_right_blocks : list of list of int
        Right auxiliary indices :math:`B_k^R`.  Empty for the last fold.
    aux_left_blocks : list of list of int
        Left auxiliary indices :math:`B_k^L`.  Empty for the first fold.
    left_out_blocks : list of list of int
        Observations used by neither the main nor the auxiliary sample of the
        fold.  Under RCF this is empty for every fold by construction -- the
        design has no buffer.
    n_samples : int
        Total number of observations ``T``.
    block_size : int
        ``T // K``.
    n_blocks : int
        Number of folds ``K``.
    """

    main_blocks: List[List[int]]
    aux_right_blocks: List[List[int]]
    aux_left_blocks: List[List[int]]
    left_out_blocks: List[List[int]] = field(default_factory=list)
    n_samples: int = 0
    block_size: int = 0
    n_blocks: int = 0

    def auxiliary(self, k: int) -> List[int]:
        """Return the auxiliary sample actually used by fold ``k``."""
        direction = fold_direction(k, self.n_blocks)
        if direction == "reverse":
            return list(self.aux_right_blocks[k])
        if direction == "forward":
            return list(self.aux_left_blocks[k])
        return sorted(set(self.aux_right_blocks[k]) | set(self.aux_left_blocks[k]))

    def as_dict(self) -> dict:
        """Dict view matching the replication package's ``block_structure_``."""
        return {
            "main_blocks": self.main_blocks,
            "aux_right_blocks": self.aux_right_blocks,
            "aux_left_blocks": self.aux_left_blocks,
            "left_out_blocks": self.left_out_blocks,
            "n_samples": self.n_samples,
            "block_size": self.block_size,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BlockStructure(K={self.n_blocks}, T={self.n_samples}, "
            f"block_size={self.block_size})"
        )


# --------------------------------------------------------------------------- #
# Reverse cross-fitting
# --------------------------------------------------------------------------- #

def reverse_cf_folds(n: int, K: int) -> BlockStructure:
    """
    Build the Reverse Cross-Fitting block structure.

    Parameters
    ----------
    n : int
        Number of time-ordered observations.
    K : int
        Number of folds, ``K >= 2``.

    Returns
    -------
    BlockStructure

    Notes
    -----
    Blocks are contiguous chunks of length ``n // K``; any remainder at the end
    of the sample is not assigned to a main block, exactly as in the paper's
    replication code.  Feed ``n`` divisible by ``K`` (see
    :func:`tsdml.prep.DataProcessor.data_prep`, which truncates for you) if you
    want every observation used.

    Examples
    --------
    >>> bs = reverse_cf_folds(20, 5)
    >>> bs.main_blocks[0]
    [0, 1, 2, 3]
    >>> bs.aux_left_blocks[0]
    []
    >>> len(bs.aux_right_blocks[0])
    16
    """
    if K < 2:
        raise ValueError("K must be at least 2")
    if n < K:
        raise ValueError(f"n={n} is smaller than K={K}")

    t_block = n // K
    if t_block < 2:
        raise ValueError(
            f"block size n//K = {t_block} is too small; reduce K or extend the sample"
        )

    main = [list(range(k * t_block, (k + 1) * t_block)) for k in range(K)]

    aux_right: List[List[int]] = [[] for _ in range(K)]
    aux_left: List[List[int]] = [[] for _ in range(K)]

    for k in range(K):
        if k == 0:
            aux_right[k] = [i for blk in main[1:] for i in blk]
            aux_left[k] = []
        elif k < K - 1:
            aux_right[k] = [i for blk in main[k + 1:] for i in blk]
            aux_left[k] = [i for blk in main[:k] for i in blk]
        else:
            aux_left[k] = [i for blk in main[: K - 1] for i in blk]
            aux_right[k] = []

    all_idx = set(range(n))
    left_out = [
        sorted(all_idx - (set(main[k]) | set(aux_right[k]) | set(aux_left[k])))
        for k in range(K)
    ]

    return BlockStructure(
        main_blocks=main,
        aux_right_blocks=aux_right,
        aux_left_blocks=aux_left,
        left_out_blocks=left_out,
        n_samples=n,
        block_size=t_block,
        n_blocks=K,
    )


def fold_direction(k: int, K: int) -> str:
    """
    Estimation direction for fold ``k`` under RCF.

    Returns
    -------
    {'reverse', 'both', 'forward'}
        ``'reverse'`` for ``k <= K//2 - 1`` (train on the right side, reading
        time backwards), ``'both'`` for the central fold when ``K`` is odd, and
        ``'forward'`` otherwise (train on the left side, forward in time).

    Examples
    --------
    >>> [fold_direction(k, 5) for k in range(5)]
    ['reverse', 'reverse', 'both', 'forward', 'forward']
    >>> [fold_direction(k, 6) for k in range(6)]
    ['reverse', 'reverse', 'reverse', 'forward', 'forward', 'forward']
    """
    if k <= K // 2 - 1:
        return "reverse"
    if k == K // 2 and K % 2 == 1:
        return "both"
    return "forward"


# --------------------------------------------------------------------------- #
# Neighbors-left-out
# --------------------------------------------------------------------------- #

def nlo_folds(n: int, K: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build Neighbors-Left-Out folds (Semenova et al., 2023).

    For main block ``k`` the auxiliary sample drops blocks ``k-1``, ``k`` and
    ``k+1``.

    Returns
    -------
    main_blocks, aux_blocks : list of list of int

    Examples
    --------
    >>> M, A = nlo_folds(25, 5)
    >>> M[2]
    [10, 11, 12, 13, 14]
    >>> A[2]
    [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]
    """
    if K < 2:
        raise ValueError("K must be at least 2")
    t_block = n // K
    main = [list(range(k * t_block, (k + 1) * t_block)) for k in range(K)]

    aux: List[List[int]] = []
    for k in range(K):
        blocks = list(main)
        if k == 0:
            del blocks[0:2]
        elif k < K - 1:
            del blocks[k - 1: k + 2]
        else:
            blocks = blocks[0: K - 2]
        aux.append([i for blk in blocks for i in blk])
    return main, aux


# --------------------------------------------------------------------------- #
# Sample-use shares (paper, Section 2.1)
# --------------------------------------------------------------------------- #

def sample_use_rcf(K: int) -> float:
    r"""
    Nuisance-sample usage share of RCF.

    .. math::

        u_{\mathrm{RCF}}(K) =
        \begin{cases}
        \dfrac{3K-2}{4K}, & K \text{ even},\\[2ex]
        \dfrac{3(K^2-1)}{4K^2}, & K \text{ odd}.
        \end{cases}

    Examples
    --------
    >>> round(sample_use_rcf(6), 4)
    0.6667
    >>> round(sample_use_rcf(5), 4)
    0.72
    """
    K = int(K)
    if K % 2 == 0:
        return (3.0 * K - 2.0) / (4.0 * K)
    return 3.0 * (K ** 2 - 1.0) / (4.0 * K ** 2)


def sample_use_nlo(K: int) -> float:
    r"""
    Nuisance-sample usage share of NLO: :math:`(K-1)(K-2)/K^2`.

    Examples
    --------
    >>> round(sample_use_nlo(6), 4)
    0.5556
    >>> sample_use_rcf(11) == sample_use_nlo(11)
    True
    """
    K = int(K)
    return (K - 1.0) * (K - 2.0) / (K ** 2)


def sample_use_table(k_values=range(3, 15)) -> "np.ndarray":
    """
    Return an array ``[[K, u_RCF, u_NLO], ...]`` for the given ``K`` values.

    Reproduces the comparison in Section 2.1: RCF uses more data than NLO for
    ``K = 3, ..., 9``, the two coincide at ``K = 11`` and NLO uses more from
    ``K >= 12`` (and at ``K = 10``).
    """
    return np.array(
        [[K, sample_use_rcf(K), sample_use_nlo(K)] for K in k_values], dtype=float
    )
