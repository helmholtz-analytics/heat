# Handover: PR 1825 (Local Outlier Factor) and distance computation

Branch: `features/1758-Implementation_of_local_outlier_factor`. The review diffs it against `main` (merge-base `7f855c9e0`). `gh` was not authenticated, so it was never confirmed that this branch is PR 1825. Nothing has been posted to GitHub.

Environment: `./.venv` (heat 1.10.0-dev, torch 2.13). Tests and probes ran with `mpirun --oversubscribe -n {1,2,3,4}`.

## 1. Scope of the PR

| File | Change |
|---|---|
| `heat/classification/localoutlierfactor.py` | New `LocalOutlierFactor` class (`fit`, `threshold` / `top_n` decision, `chunks`, `fully_distributed`) |
| `heat/spatial/distance.py` | New `cdist_small` (k smallest distances per row, using shift-by-k communication), `_chunk_wise_topk`, `_stable_lexsort`, `METRIC_MAP`; euclidian→euclidean rename with aliases; `_quadratic_expand` cleanup; removed an unreachable `elif` in `_dist` |
| `heat/core/_indexing_utils.py` | Indexing with a distributed index array: the output now inherits `key.split` |
| `tests/classification/test_lof.py`, `tests/spatial/test_distances.py`, `tests/core/test_dndarray.py` | New tests |
| `benchmarks/cb/classification.py` | LOF benchmarks |

The PR's own tests pass at 1–4 processes.

## 2. Confirmed correctness bugs (each reproduced by running it)

1. **Unevenly split input gives wrong neighbor indices and wrong LOF scores** (2+ processes, both `fully_distributed` modes).
   - Cause: `distance.py:402` takes `ydispl` from `comm.counts_displs_shape`, which assumes an even split.
   - Fix: use `Y.counts_displs()`.
2. **`top_n` can flag fewer than `top_n` points.** `top_n=1` flagged 0 points at 1–4 processes.
   - Cause: `localoutlierfactor.py:210` compares a float64 array with a Python float, which Heat casts to float32 first (an existing core behaviour; `main` does the same).
   - Fix: build `anomaly` from the indices `ht.topk` returns.
3. **`metric="gaussian"` returns the farthest points as "nearest"** (`distance.py:146`). It is a similarity, but `cdist_small` keeps the smallest values.
4. **The point itself is dropped by position (column 0), not by index** (`localoutlierfactor.py:140`).
   - Above 25 rows, `torch.cdist` switches to the matmul shortcut. On float32 data with a large offset, a point's distance to itself came out as 0.707, so column 0 isn't self.
   - LOF error was 0.044 at 1–2 processes and 0 at 3–4: **results depend on the process count.**
   - With duplicate rows, `idx_n_neighbors` contains the point itself.
   - Fix: remove self by index (as sklearn does) and/or compute float32 distances in float64.
5. **Ties make results depend on the process count** (grid data: checksums 41.7745 / 41.7656 / 41.7776 at 1 / 2 / 3 processes). Each block's `torch.topk` drops tied candidates before `_stable_lexsort` runs, and with 1 process the lexsort is skipped.
6. **Indexing with a distributed key whose rows are all on one rank now silently returns wrong data** (on `main` it raised). Cause: the shape heuristic at `dndarray.py:1790`. This affects `fully_distributed=True` on uneven data.
7. **Missing validation.**
   - `top_n > n_samples` raises an unhelpful `RuntimeError: Padding length must be divisible by 2`.
   - Integer input crashes in `torch.cdist` when unsplit, because `cdist_small` (`distance.py:367`) only converts types in the distributed path.

**Existing issues this PR touches:**
- `_dist` with `X.split=0`, `Y.split=1` silently returns zeros. The removed `elif` was dead code; a real check is still missing.
- `tests/core/linalg/test_basics.py:75,88`: `assertTrue(est.dtype, dtype)` always passes.

**Rejected claims from the automated review:** int16 index keys fail on `main` too, so that isn't a regression; `top_n > n` raises rather than flagging everything.

## 3. Test gaps (the tests do not guarantee distributed correctness)

What's good: LOF scores are compared with sklearn at 1e-6 on split data for both modes, and `cdist_small` is compared with `cdist` + `topk`.

Gaps (each lets one of the bugs above pass):
- No unevenly split inputs (#1).
- `test_advanced_indexing` (`test_lof.py:95,97`) uses in-place `resplit_(None)`, so both operands are replicated before the call and nothing distributed is tested. It also passes a 2-D `A`, producing an inconsistent array (gshape `(5,)`, 2-D local tensor).
- The `top_n` test (`test_lof.py:151`) doesn't assert that exactly `top_n` points are flagged (#2).
- Only float64 Euclidean is tested; no float32, manhattan or gaussian (#3, #4).
- Not covered: LOF with `chunks>1`, LOF on unsplit data, `n_neighbors >= n`, duplicates or ties, fewer rows than processes, and local Y block smaller than `n_smallest`. The last two were checked manually and work.
- Index results are compared with `allclose`; use an exact comparison.

**Creating uneven arrays for new tests.** There is no shared helper in the suite. Use `redistribute_`:
```python
def make_unbalanced(x, counts):
    target = x.lshape_map.clone()
    target[:, x.split] = torch.tensor(counts)
    x.redistribute_(target_map=target)
    return x
```
- Skip when `comm.size < 2`.
- `lshape_map`, `is_balanced(force_check=True)` and `numpy()` are collective: call them on every rank.
- Cover a rank with zero rows and the case where all rows are on the last rank.

## 4. Performance

Measured on one node with 8 cores, n=20 000, f=8, k=20, float32:

| | 1 proc | 2 procs | 4 procs |
|---|---|---|---|
| `cdist_small` | 0.54 s | 0.51 s | 0.64 s |
| `cdist_small`, chunks=4 | 0.90 s | 0.74 s | 0.72 s |
| LOF `fully_distributed=False` | 0.65 s | 0.64 s | 0.66 s |
| LOF `fully_distributed=True` | 0.60 s | 0.69 s | 0.76 s |

- `cdist_small` dominates LOF runtime.
- `fully_distributed=True` is 0–15 % slower here, not "much slower" as the docstring says. Multi-node and GPU were not tested.

Improvements for `cdist_small`:
- Overlap communication with computation: it waits on each send/receive (`distance.py:468`) before computing.
- Stop routing GPU data through host memory every step (`distance.py:432,472`); Heat's communication layer already handles CUDA-aware MPI.
- Use the symmetry when X is Y (always the case for LOF), as `_dist` does.
- Avoid the `astype` copies when the dtype already matches (`distance.py:389`).
- The "chunks larger than local rows" warning fires on every rank at every step.

Also: the `lof_non_distributed` benchmark builds a 20 000×20 000 float64 matrix (3.2 GB) on every rank.

## 5. How `_dist` works (`distance.py:512-776`)

Each rank owns a horizontal strip of `D`: its X rows against all columns.

| `X.split` | `Y` / `Y.split` | Path | Result split |
|---|---|---|---|
| None | None (no Y) | local `metric(X, X)` on every rank | None |
| 0 | no Y | **symmetric half-exchange** | 0 |
| None | None / 0 | local computation | None / 1 |
| 0 | None | local computation | 0 |
| 0 | 0 | **shift-by-k exchange** | 0 |
| 0 | 1 | no branch matches → zeros (bug) | 0 |

**Shift-by-k exchange (X and Y split).** In iteration k (1…p−1), rank r sends its own Y block to r+k and receives block r−k, then fills column block (r−k) mod p. Each rank holds one received Y block at a time. `cdist_small` uses the same pattern.

**Symmetric half-exchange (Y omitted).** In iteration k (1…⌈p/2⌉−1):
- Phase 1: send X_q to q+k, which computes tile D[q+k, q].
- Phase 2: that tile is sent back to q, which stores its transpose as D[q, q+k].

Each tile is computed once, which halves the arithmetic but sends (n/p)² tiles back. For even p, an extra half-step handles the opposite tile, with half the ranks idle.

**Blocking communication.** Ranks `< k` send first and the others receive first, to avoid deadlock. For messages above MPI's eager limit this forms a chain: a shift costs up to p back-to-back transfer times instead of about one. This is analysis, not measured.

**Verified:** results are exact for evenly split input at 1/3/4/5 processes. Uneven input raises a shape-mismatch `RuntimeError` (`counts_displs_shape` assumes an even split).

## 6. Working tree state and files

- **`.gitignore`:** a `### Claude Code ###` section with `.claude/` was added. Uncommitted; commit it separately from the PR.
- **`tests/spatial/test_distances.py`:** has uncommitted work in progress (not from this session). Notes on its `_shape` helper and new `test_cdist`:
  - `_shape(balanced=False)` does not make an unevenly split array: Heat still splits evenly, so `is_balanced()` is True.
  - `if split:` skips `split=0`; use `if split is not None:`.
  - `@pytest.mark.parametrize` doesn't work on Heat's `TestCase`, a `unittest.TestCase` (verified: "missing positional argument").
  - `parametrize("dist_func", splits)` passes the `splits` list.
- **Empty read-only dotfiles in the repo root** (`.bashrc`, `.gitconfig`, `.idea`, …): created by the Claude Code sandbox. Don't `git add -A`; they're safe to delete after the session.
- **Probe scripts** in `.claude/tmp/`:
  - `lof-review/`:
    - `repro_unbalanced.py`: uneven input, small local blocks, gaussian, chunks
    - `repro_edge.py`, `repro_edge2.py`: duplicates, float32 precision, top_n, ties, int input. In `repro_edge.py` the top_n/int sections call collectives only on rank 0 and hang with more than 1 process; use `repro_edge2.py`.
    - `repro_indexing.py`, `repro_idx2.py`: index keys, PR branch vs `main`
    - `bench.py`: timings
  - `dist-explain/`: `_dist` with even and uneven input.
  - `unbalanced/`: ways to create uneven arrays.
  - Comparisons against `main` used `git archive 7f855c9e0 heat` extracted to `$TMPDIR/heat-main` and `PYTHONPATH`.

## 7. Suggested next steps

1. Fix bugs #1–#4 (wrong offsets, `top_n`, gaussian metric, self removed by position); they silently change results.
2. Add tests for uneven input and float32, count-check `top_n`, and rewrite `test_advanced_indexing` to use distributed operands.
3. Decide how ties should be handled (deterministic selection or documentation), and add the validation from #7.
4. Performance work on `cdist_small`: overlap communication with computation, GPU path, symmetry.

## 8. LOF test rewrite (status 2026-11-04)

### What exists now (uncommitted)

| File | Contents |
|---|---|
| `heat/testing/distribution.py` | New helpers. `layout_counts(n, nprocs, layout)` gives per-rank counts for the layouts `balanced`, `uneven`, `empty_first` and `all_last`. `make_unbalanced(x, counts_or_layout)` redistributes `x` in place with `redistribute_`. `assert_distribution_consistent(x)` checks local shapes against `gshape`, `split` and `balanced`. |
| `tests/conftest.py` | New fixtures. From `TestCase`: `device_config`, `device`, `other_device`, `is_mps`, `comm`, `rank`, `size` and `seed`. For the tests: `make_array` (builds a split/layout DNDarray and skips layouts that can't apply), and the LOF fixtures `lof_data`, `lof_n_outliers` and `sklearn_lof`. `sklearn_lof` is a cached sklearn reference that clamps `n_neighbors` to n−1 and fails if the data has ties at the k-th neighbour. |
| `tests/classification/test_lof.py` | Fully parametrised rewrite: input validation, sklearn agreement over split × dtype × layout × mode × metric, `n_neighbors`, `chunks`, threshold and `top_n` decisions, float32 with a large offset, duplicates, ties and distribution, gaussian neighbours, and `_advanced_indexing`. |

Ruff check and format pass on all three. The conftest is in `tests/` (the repo's test directory), not `test/`.

### Verified so far
- **1 process:** the run completes. 93 failed, 132 passed, 621 skipped (skips are unbalanced layouts, which need ≥2 processes). Every failure looked at is a real bug from section 2 or missing validation, not a test mistake:
  - split=1 is not supported.
  - Integer input crashes when unsplit (#7).
  - `top_n > n` raises `RuntimeError` (#7).
  - 1-D/3-D or non-DNDarray input gives `IndexError`/`AttributeError`.
  - No validation of `chunks`, or of `top_n` given as a float.
  - float32 with a large offset puts a point among its own neighbours (#4).
  - Duplicate rows put a point among its own neighbours (#4).
  - The gaussian metric returns the farthest points (#3).
  - `_advanced_indexing` returns `split=None` for split input with `fully_distributed=True`.
- **2 processes:** the run **deadlocks** in `test_lof_matches_sklearn[0-float32-empty_first-False-manhattan]`, even when that test runs alone. The cause isn't known yet. It looks like a Heat-side hang when a rank holds no rows, probably in `cdist_small`.
- **3 and 4 processes:** no usable results. Those runs were affected by the hang and then stopped. Only the 1-process results above are complete.
- Run every `mpirun` under `timeout -k 10 600`. Sandboxed commands each get their own PID namespace, so `ps`/`pgrep` don't show MPI processes started by earlier commands. Logs from these runs are in `.claude/tmp/lof-tests/`.

### What still needs to be done
1. **Find where the 2-process hang happens.** Run a standalone probe with `faulthandler.dump_traceback_later` under `timeout`. Then decide:
   - If it's a Heat bug, that is a valid finding. The test then needs a guard so it can't block the whole suite: for example, add `pytest-timeout` (with permission, since installing dependencies needs approval), or mark the case `xfail(run=False)` until it's fixed.
   - If the test causes it, fix the test.
2. **Get complete runs at 2, 3 and 4 processes**, each under `timeout`. Then sort every failure into "real bug" or "test bug", as was done for 1 process. The uneven-split (#1), `top_n` count (#2) and tie (#5) bugs should only show up there.
3. **Open decisions:**
   - The tests expect split=1 to give correct results instead of `NotImplementedError`. Should it be supported?
   - Should `_advanced_indexing` always return `idx.split`? Its docstring says so; it currently returns `None` on 1 process.
   - Several validation tests accept `TypeError` or `ValueError` (non-int `n_neighbors`, float `top_n`, `chunks`, wrong input types). Should one of them be required?
4. **Optional cleanups:**
   - `test_fit_invalid_shape[None]` covers the 1-D case twice.
   - `TestCase.setUpClass` and the `device_config` fixture duplicate the device selection; it could move into one shared function.
5. **The failure count is large by design.** These tests are written to fail until the bugs in section 2 are fixed. If they should go into the PR before the fixes, they need `xfail` markers that point at each bug.
