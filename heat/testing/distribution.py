"""Helpers to create and check DNDarrays with a non-default (unbalanced) distribution in tests."""

from typing import Literal, Sequence

import torch

from heat.core.dndarray import DNDarray

__all__ = ["LAYOUTS", "layout_counts", "make_unbalanced", "assert_distribution_consistent"]

Layout = Literal["balanced", "uneven", "empty_first", "all_last"]

LAYOUTS: tuple[Layout, ...] = ("balanced", "uneven", "empty_first", "all_last")
"""Distribution layouts understood by :func:`layout_counts`."""


def layout_counts(n: int, nprocs: int, layout: Layout) -> list[int]:
    """
    Number of elements per process along the split axis for a named distribution layout.

    Parameters
    ----------
    n : int
        Global size of the split axis.
    nprocs : int
        Number of processes.
    layout : str
        - ``"balanced"``: Heat's default distribution (remainder on the first processes).
        - ``"uneven"``: process ``r`` gets a share proportional to ``r + 1``, so every process holds a
          different number of elements and the last process holds the most.
        - ``"empty_first"``: process 0 holds no elements, the rest are distributed evenly.
        - ``"all_last"``: the last process holds all elements.

    Returns
    -------
    counts : list[int]
        Counts per process, summing up to ``n``.
    """
    if layout == "balanced":
        counts = [n // nprocs + (r < n % nprocs) for r in range(nprocs)]
    elif layout == "uneven":
        weights = range(1, nprocs + 1)
        total = sum(weights)
        counts = [n * w // total for w in weights]
        counts[-1] += n - sum(counts)
    elif layout == "empty_first":
        if nprocs < 2:
            raise ValueError("Layout 'empty_first' requires at least 2 processes.")
        counts = [0] + layout_counts(n, nprocs - 1, "balanced")
    elif layout == "all_last":
        counts = [0] * (nprocs - 1) + [n]
    else:
        raise ValueError(f"Unknown layout '{layout}'. Use one of {LAYOUTS}.")
    return counts


def make_unbalanced(x: DNDarray, counts: Sequence[int] | Layout) -> DNDarray:
    """
    Redistribute ``x`` in-place along its split axis so that process ``r`` holds ``counts[r]`` elements.

    This is a collective operation and has to be called on every process.

    Parameters
    ----------
    x : DNDarray
        Distributed array (``x.split`` must not be ``None``).
    counts : Sequence[int] or str
        Counts per process along the split axis, or a layout name accepted by :func:`layout_counts`.

    Returns
    -------
    x : DNDarray
        The redistributed input array, returned for convenience.
    """
    if x.split is None:
        raise ValueError("Cannot redistribute a non-distributed DNDarray.")
    if isinstance(counts, str):
        counts = layout_counts(x.gshape[x.split], x.comm.size, counts)
    if len(counts) != x.comm.size or sum(counts) != x.gshape[x.split]:
        raise ValueError(
            f"counts must have one entry per process ({x.comm.size}) and sum up to "
            f"{x.gshape[x.split]}, but got {list(counts)}."
        )
    target = x.lshape_map.clone()
    target[:, x.split] = torch.tensor(counts, dtype=target.dtype, device=target.device)
    x.redistribute_(target_map=target)
    return x


def assert_distribution_consistent(x: DNDarray) -> None:
    """
    Assert that the local tensors of ``x`` are consistent with its global metadata.

    Checks the number of dimensions and the non-split dimensions of every local tensor, that the
    local sizes along the split axis add up to the global shape, and that ``x.balanced`` is not
    claiming a balanced distribution when the data is not balanced.
    This is a collective operation and has to be called on every process.

    Parameters
    ----------
    x : DNDarray
        Array to check.
    """
    shapes = x.comm.allgather(tuple(x.larray.shape))
    for rank, shape in enumerate(shapes):
        assert len(shape) == x.ndim, (
            f"rank {rank}: local tensor has {len(shape)} dimensions, gshape {x.gshape} has {x.ndim}"
        )
        for dim, (local, glob) in enumerate(zip(shape, x.gshape)):
            if dim != x.split:
                assert local == glob, f"rank {rank}: local shape {shape} vs gshape {x.gshape}"
    if x.split is None:
        assert all(shape == tuple(x.gshape) for shape in shapes), (
            f"non-distributed array with local shapes {shapes} and gshape {x.gshape}"
        )
    else:
        total = sum(shape[x.split] for shape in shapes)
        assert total == x.gshape[x.split], (
            f"local shapes {shapes} along split {x.split} sum to {total}, gshape is {x.gshape}"
        )
        if x.balanced:
            counts = [shape[x.split] for shape in shapes]
            assert counts == layout_counts(x.gshape[x.split], x.comm.size, "balanced"), (
                f"array flagged as balanced, but local sizes along split are {counts}"
            )
