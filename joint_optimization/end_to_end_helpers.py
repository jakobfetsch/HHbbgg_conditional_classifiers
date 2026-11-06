#!/usr/bin/env python3
"""
Helpers for end_to_end_training.py: data loading/shuffling, the combined
classifier+GMM model, and small run-management/plotting utilities.

This module holds everything end_to_end_training.py needs that ISN'T the
top-level training orchestration (that stays in main()) and isn't a
training-loop mechanic (that lives in training.py - pretrain(), the chunked
Stage-B training step train_complete_model(), memory/profile helpers, etc).
Roughly:

    training.py             -> HOW one epoch/chunk is trained
    end_to_end_helpers.py    -> WHAT is trained and WHAT DATA it sees (this file)
    end_to_end_training.py   -> orchestration: parses args/config, wires the
                                 pieces above together, drives Stage A -> Stage B
"""

import csv
import json
import os

import hist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from scipy.optimize import curve_fit

from gatohep.models import gato_gmm_model
from gatohep.utils import asymptotic_significance
from gatohep.plotting_utils import (
    assign_bins_and_order,
    fill_histogram_from_assignments,
    plot_stacked_histograms,
)

from dataset_prep import (
    build_training_dataset,
    CLASSIFIER_PRESELECTION_DIJET_MASS_COLUMN,
    CLASSIFIER_PRESELECTION_LEAD_MVAID_COLUMN,
    CLASSIFIER_PRESELECTION_SUBLEAD_MVAID_COLUMN,
)

from training import (
    soft_mass_window_weight, _host_rss_bytes, _host_peak_rss_bytes,
    evaluate_complete_model_chunked, build_data_dict_chunked, make_ce_loss_fn,
    scaled_nbins_for_split, resolve_nonres_nbins,
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
class ConcatView:
    """Zero-copy axis-0 view over multiple numpy/memmap arrays."""

    def __init__(self, *parts):
        self._parts = tuple(parts)
        if not self._parts:
            raise ValueError("ConcatView requires at least one array")
        self.shape = (sum(p.shape[0] for p in self._parts),) + self._parts[0].shape[1:]
        self.ndim = len(self.shape)
        self.dtype = self._parts[0].dtype
        self._offsets = np.cumsum([0] + [p.shape[0] for p in self._parts])

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        """See PermutedView.__array__ - avoids a row-by-row Python fallback."""
        out = self[0:self.shape[0]]
        return np.asarray(out, dtype=dtype) if dtype is not None else np.asarray(out)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                return np.concatenate(
                    [self[i] for i in range(start, stop, step)], axis=0
                )
            return self._slice_range(start, stop)
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0:
                idx += self.shape[0]
            for part, off in zip(self._parts, self._offsets[:-1]):
                local = idx - off
                if 0 <= local < part.shape[0]:
                    return part[local]
            raise IndexError(key)
        raise TypeError(f"Unsupported index type: {type(key)}")

    def _slice_range(self, start, stop):
        chunks = []
        for part, off in zip(self._parts, self._offsets[:-1]):
            part_start = max(0, start - off)
            part_stop = min(part.shape[0], stop - off)
            if part_start < part_stop:
                chunks.append(part[part_start:part_stop])
        if not chunks:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)


class PermutedView:
    """
    Lazy, zero-copy row-permutation view over a numpy/memmap array.

    Only the rows actually requested (via a slice or an index array, the
    same access patterns used by ConcatView / make_memmap_dataset) are ever
    read from the underlying array - the full (possibly huge) dataset is
    never materialized as a shuffled copy in RAM. This is what lets us
    shuffle X_train/X_val (which can be large memmapped feature matrices)
    between Stage A and Stage B without doubling memory/disk usage.
    """

    def __init__(self, base, perm):
        perm = np.asarray(perm)
        # COMPOSE nested views instead of stacking them. main() wraps
        # X_train twice - once for the train/val shuffle, once for the
        # preselection row selection - and a two-level view turned every
        # single chunk read into two Python-level fancy-index hops before it
        # ever reached the memmap. Composing collapses that to one.
        if isinstance(base, PermutedView):
            perm = base._perm[perm]
            base = base._base
        self._base = base
        self._perm = perm
        self.shape = (self._perm.shape[0],) + tuple(base.shape[1:])
        self.ndim = base.ndim
        self.dtype = base.dtype

    @property
    def base_array(self):
        """The underlying (un-permuted) array this view reads from."""
        return self._base

    @property
    def index_array(self):
        """base_array row index for each logical row of this view."""
        return self._perm

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            idx = self._perm[key]
        elif isinstance(key, (int, np.integer)):
            return self._base[int(self._perm[int(key)])]
        else:
            idx = self._perm[np.asarray(key)]

        # Fast path: a run of consecutive base rows is served as a plain
        # slice, which for a memmap is a zero-copy view and a purely
        # sequential read instead of a gather. Only reachable once the
        # index array is ascending - see sequentialize_stage_b_split().
        if idx.size == 0:
            return np.empty((0,) + tuple(self.shape[1:]), dtype=self.dtype)
        if (int(idx[-1]) - int(idx[0]) == idx.size - 1) and bool(np.all(np.diff(idx) == 1)):
            return self._base[int(idx[0]): int(idx[-1]) + 1]

        # Fancy-indexing a memmap with an integer array reads only those
        # rows from disk into a small in-memory array.
        return self._base[idx]

    def __array__(self, dtype=None, copy=None):
        """
        Makes np.asarray(view) go through the vectorized slice path above.
        Without this, numpy falls back on __len__ + __getitem__(int) and
        materializes the array one ROW AT A TIME in a Python loop - which on
        a large feature matrix is both extremely slow and a silent way to
        pull the whole memmap into RAM.
        """
        out = self[0:self.shape[0]]
        return np.asarray(out, dtype=dtype) if dtype is not None else np.asarray(out)

    def sorted_by_base_row(self):
        """
        Returns (view, order) where `view` addresses exactly the same set of
        rows as `self` but in ascending base-array order, and `order` is the
        permutation of THIS view's logical positions that produces it
        (i.e. view[i] == self[order[i]]).

        `order` is what callers must apply to every other array that is
        row-aligned with this view, so the whole set stays consistent.
        """
        order = np.argsort(self._perm, kind="stable")
        return PermutedView(self._base, self._perm[order]), order


def shuffle_raw_data(raw_data, class_weights, seed):
    """
    Shuffles X/y/rel_w (and the matching class_weight arrays) once, using
    one fixed permutation per split (train, val) so all arrays belonging to
    the same split stay aligned. Wraps arrays in PermutedView instead of
    copying them, so this is safe to use on large memmapped datasets.

    Returns (raw_data, class_weights, train_perm, val_perm) - the two
    permutation arrays are returned so that OTHER arrays loaded later but
    still row-aligned with the original (pre-shuffle) X_train.npy/X_val.npy
    - e.g. mass_train.npy/mass_val.npy for the mass-window feature - can be
    permuted with the identical permutation instead of silently staying
    unshuffled (and therefore desynced from raw_data) if they are combined
    with raw_data downstream.
    """
    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(raw_data["X_train"].shape[0])
    val_perm = rng.permutation(raw_data["X_val"].shape[0])

    raw_data["X_train"] = PermutedView(raw_data["X_train"], train_perm)
    raw_data["y_train"] = PermutedView(raw_data["y_train"], train_perm)
    raw_data["rel_w_train"] = PermutedView(raw_data["rel_w_train"], train_perm)

    raw_data["X_val"] = PermutedView(raw_data["X_val"], val_perm)
    raw_data["y_val"] = PermutedView(raw_data["y_val"], val_perm)
    raw_data["rel_w_val"] = PermutedView(raw_data["rel_w_val"], val_perm)

    class_weights["cw_train"] = PermutedView(class_weights["cw_train"], train_perm)
    class_weights["cw_train_no_abs"] = PermutedView(class_weights["cw_train_no_abs"], train_perm)
    class_weights["cw_val"] = PermutedView(class_weights["cw_val"], val_perm)

    print("[INFO] Shuffled train/val datasets (X, y, rel_w, class_weights) once between Stage A and Stage B.")
    return raw_data, class_weights, train_perm, val_perm


def _reorder_aligned(arr, order):
    """
    Applies the logical row permutation `order` to one array that is
    row-aligned with a Stage-B feature matrix.

    PermutedViews stay lazy (the permutations are composed, nothing is read
    from disk); already-materialized numpy arrays - which here are only ever
    1-D (rel_w, cw, mass) - are reordered by an ordinary fancy index.
    """
    if arr is None:
        return None
    if isinstance(arr, PermutedView):
        return PermutedView(arr.base_array, arr.index_array[order])
    return np.asarray(arr)[order]


def sequentialize_stage_b_split(X, aligned, split_name="", enabled=True):
    """
    Re-orders one Stage-B split so that reading it chunk by chunk walks the
    underlying .npy memmap in ASCENDING row order.

    WHY THIS IS FREE. Stage B is full-batch: the quantities it forms are
        S_total = sum_over_events(...),  B_total = sum_over_events(...)
    and Z = f(S_total, B_total) is computed once from the totals. Nothing in
    finetune_train_step_chunked depends on which events land in which chunk
    or in what order (this is exactly the additivity argument spelled out in
    EndToEndClassifierGMM.chunk_yields' docstring). The per-event CE term is
    likewise a plain sum. So permuting the rows is mathematically a no-op.

    The one honest caveat: float32 addition is not associative, so summing the
    same values in a different order can move the last bit or two (measured at
    ~1e-7 relative). That is the identical, already-accepted effect you get
    from changing training.chunk_size, and it is orders of magnitude below
    anything that matters for Z. Set training.sequential_row_order: false to
    opt out and reproduce a previous run exactly.

    WHY IT MATTERS. shuffle_raw_data() wraps X_train in a PermutedView with a
    random permutation (needed for Stage A's minibatch SGD, meaningless for
    Stage B), and preselection wraps it again with a row-selection index. The
    net effect is that a "sequential" chunk read X_all[a:b] becomes a gather
    of ~chunk_size scattered rows from all over a multi-GB file. Stage B
    reads the full training set several times per epoch (two passes per inner
    step, plus validation and diagnostics), so that random-access pattern is
    paid over and over. Sorting the index restores sequential I/O and lets
    PermutedView.__getitem__ hit its zero-copy contiguous-slice path wherever
    the preselection kept a contiguous run.

    X:        the split's feature matrix (a PermutedView, or anything else -
              in which case this is a no-op).
    aligned:  iterable of arrays row-aligned with X (y, rel_w, cw, mass...).
              None entries are passed through untouched.

    Returns (X_seq, [aligned_seq...]).
    """
    aligned = list(aligned)
    if not enabled or not isinstance(X, PermutedView):
        return X, aligned

    X_seq, order = X.sorted_by_base_row()
    if np.array_equal(order, np.arange(order.shape[0])):
        print(
            f"[INFO] Stage-B '{split_name}' split is already in ascending on-disk "
            f"row order - nothing to reorder."
        )
        return X, aligned

    aligned_seq = [_reorder_aligned(a, order) for a in aligned]
    print(
        f"[INFO] Stage-B '{split_name}' split re-ordered to ascending on-disk row "
        f"order ({X_seq.shape[0]:,} rows): chunk reads are now sequential instead "
        f"of scattered. The Stage-B loss is order-independent, so S, B, Z and the "
        f"gradients are unchanged up to float32 summation order (~1e-7 relative, "
        f"the same effect as changing chunk_size)."
    )
    return X_seq, aligned_seq

def _save_nonres_fit_diagnostics_plot(edges, vals, variances, cat, n_raw, nbins, branch, chi2_ndof,
                                       factor, fit_A, fit_B,
                                       mass_sb_low, mass_sb_high, mass_sig_low, mass_sig_high,
                                       plot_dir, plot_dtype="pdf", call_label=""):
    """Per-category histogram + exponential-fit-curve + pull plot for the
    non-resonant sideband fit - same visual layout as
    nonres_fit_diagnostics.py's _save_nonres_fit_plot (see that script),
    but deliberately WITHOUT that script's adaptive per-category bin count
    or bounds-constrained/low-stats-fallback "robust" fit machinery: nbins
    here is always the single, fixed value the caller/config passed to
    compute_nonres_reweight_factors (see training_config.yaml's
    nonres_reweight.nbins - e.g. 25 instead of the 40 default), the same
    for every category, per the user's request to keep the simple
    fixed-bin-count fit rather than reintroducing the adaptive version.

    Takes plain numpy arrays (edges, vals, variances) rather than a
    hist.Hist object, so this can be called BOTH from the live fit below
    (which still has the hist.Hist around) AND from a standalone replot
    script working only from the saved diagnostics CSV (see
    plot_nonres_sideband_fits.py) - no hist.Hist/model/data access is
    needed in that second case, since edges/vals/variances (plus the
    window bounds and fit result) are exactly what the CSV now stores
    per category."""
    import matplotlib.pyplot as plt
    import mplhep as hep

    def expf(x, A, B):
        return A * np.exp(B * x)

    edges = np.asarray(edges, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    errs = np.sqrt(np.maximum(np.asarray(variances, dtype=np.float64), 1e-12))
    has_fit = fit_A is not None and fit_B is not None

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax_pull = fig.add_subplot(gs[1], sharex=ax)

    hep.histplot(
        vals, edges, ax=ax, histtype="fill", color="crimson", alpha=0.55,
        edgecolor="black", linewidth=1.0, label=f"non-res. MC (n_raw={n_raw}, nbins={nbins})",
    )
    hep.histplot(
        vals, edges, ax=ax, histtype="errorbar", yerr=errs, color="black",
        markersize=4, elinewidth=1.2,
    )

    if has_fit:
        x_fine = np.linspace(mass_sb_low, mass_sb_high, 400)
        ax.plot(x_fine, expf(x_fine, fit_A, fit_B), linestyle="--", color="navy", linewidth=2,
                label=rf"fit: $A\,e^{{Bm}}$, $\chi^2$/ndof={chi2_ndof:.2f}")

    ax.axvspan(mass_sig_low, mass_sig_high, color="tab:blue", alpha=0.15,
               label=f"signal window [{mass_sig_low:.0f}, {mass_sig_high:.0f}]")
    ax.set_yscale("log")
    ax.set_ylabel("Weighted yield", fontsize=18)
    ax.legend(fontsize=10, loc="upper right")
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(f"cat {cat}" + (f" ({call_label})" if call_label else "") + f" | factor={factor:.4f}", fontsize=20, loc="right")

    if has_fit:
        fit_vals = expf(centers, fit_A, fit_B)
        pulls = (vals - fit_vals) / errs
        ax_pull.axhspan(-1, 1, color="0.8", alpha=0.6)
        ax_pull.axhline(0, color="navy", linestyle="--", linewidth=1.5)
        ax_pull.errorbar(centers, pulls, fmt="o", ms=4, color="black")
        y_extent = max(3.0, float(np.max(np.abs(pulls))) * 1.2) if len(pulls) else 3.0
        ax_pull.set_ylim(-y_extent, y_extent)
        ax_pull.set_ylabel(r"$\frac{N_i-\mathrm{fit}_i}{\sigma_i}$", fontsize=18)
    else:
        ax_pull.axhline(0, color="navy", linestyle="--", linewidth=1.5)
        ax_pull.set_ylim(-3, 3)
        ax_pull.text(
            0.5, 0.5, f"no fit performed ({branch})", transform=ax_pull.transAxes,
            ha="center", va="center", fontsize=12, color="0.3",
        )
        ax_pull.set_ylabel("Pull", fontsize=14)

    ax_pull.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]", fontsize=18)
    ax_pull.set_xlim(mass_sb_low, mass_sb_high)

    os.makedirs(plot_dir, exist_ok=True)
    safe_label = "".join(c if (c.isalnum() or c in "._-") else "_" for c in (call_label or "run"))
    fname = f"nonres_sideband_fit_cat{cat}_{safe_label}.{plot_dtype}"
    fig.savefig(os.path.join(plot_dir, fname), bbox_inches="tight")
    plt.close(fig)


def compute_nonres_reweight_factors(
    model, X, y, weights, mass, nonres_class, n_cats,
    mass_sb_low=100.0, mass_sb_high=180.0,
    mass_sig_low=120.0, mass_sig_high=130.0,
    nbins=40, default_factor=0.062, chunk_size=None,
    diagnostics_csv_path=None, save_fit_plots=False, plot_dir=None,
    plot_dtype="pdf", call_label="",
):
    """
    Per-GMM-category continuum-background reweight factors, ported from
    gato_utils.py's compute_nonres_reweight_factors_3d.

    Unlike the reference (one dataframe per physical process, picked out by
    name), this works off the single class-indexed (X, y, weights, mass)
    arrays used everywhere else here: only events with y == nonres_class
    (the non-resonant continuum background class, e.g. "is_nonRes_bkg",
    class index 0 in training_config_edit.yaml's `classes` list) are used
    to build the per-category sideband mass histograms.

    factor[k] = (continuum yield predicted in [mass_sig_low, mass_sig_high)
                 by an exponential fit to the [mass_sb_low, mass_sb_high)
                 sideband of GMM category k) / (raw MC yield of category k
                 in that same window)

    A *hard* (arg-max) GMM assignment is used to bin events into
    categories - exactly like the reference's model.get_bin_indices - the
    soft/differentiable membership is only used later, when the resulting
    per-category factors are applied to the training loss (see
    EndToEndClassifierGMM.build_data_dict / GMMFromScores.call).

    Returns a length-n_cats float32 numpy array. A category whose fit fails,
    is empty, or has zero raw MC in the signal window falls back to
    `default_factor` or the naive window-width fraction, matching the
    reference exactly.

    Diagnostics (opt-in, off by default - existing callers of this function
    are unaffected): this is the SIMPLE, fixed-nbins fit only - it does NOT
    use nonres_fit_diagnostics.py's adaptive-per-category-binning/bounded-
    fit "robust" version (per-category `nbins` here is always the single
    fixed value passed in, e.g. 25 or 40 - see training_config.yaml's
    nonres_reweight.nbins). If `diagnostics_csv_path` is given, a one-row-
    per-category diagnostic table (n_raw, nbins, branch, chi2/ndof, fit A/B,
    factor) - the same numbers nonres_fit_diagnostics.py prints/logs, minus
    the adaptive-binning columns that don't apply here - is written there.
    If `save_fit_plots` is also True, a histogram+fit-curve+pull PDF plot
    per category is additionally written to `plot_dir` (see
    _save_nonres_fit_diagnostics_plot above).
    """
    y = np.asarray(y)
    mask = (y == nonres_class)
    if not np.any(mask):
        factors = np.full(n_cats, default_factor, dtype=np.float32)
        if diagnostics_csv_path is not None:
            _write_nonres_diagnostics_csv(
                diagnostics_csv_path,
                [(k, 0, nbins, "fallback(no-nonres-events)", float("nan"), None, None, float(factors[k]),
                  mass_sb_low, mass_sb_high, mass_sig_low, mass_sig_high,
                  json.dumps(None), json.dumps(None), json.dumps(None))
                 for k in range(n_cats)],
            )
        return factors

    X_sel = X[mask]
    w_sel = np.asarray(weights)[mask].astype(np.float64)
    m_sel = np.asarray(mass)[mask].astype(np.float64)

    # Classifier scores -> hard GMM bin per event, chunked to bound memory
    # (mirrors build_data_dict_chunked's chunking, but for a single class).
    n = X_sel.shape[0]
    step = chunk_size or n
    bin_chunks = []
    for a in range(0, n, step):
        b = min(a + step, n)
        scores, _ = model.scores(X_sel[a:b], training=False)
        bin_chunks.append(model.gato.get_bin_indices(scores).numpy())
    bins = np.concatenate(bin_chunks, axis=0) if bin_chunks else np.zeros(0, dtype=np.int64)

    bin_width = (mass_sb_high - mass_sb_low) / nbins
    factors = np.full(n_cats, default_factor, dtype=np.float64)
    naive_fraction = (mass_sig_high - mass_sig_low) / (mass_sb_high - mass_sb_low)
    diagnostics = []  # (cat, n_raw, nbins, branch, chi2_ndof, fit_A, fit_B, factor)

    def expf(x, A, B):
        return A * np.exp(B * x)

    def integral_exp(A, B, x1, x2):
        return (A / B) * (np.exp(B * x2) - np.exp(B * x1)) if abs(B) > 1e-10 else A * (x2 - x1)

    if save_fit_plots and plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

    # Common trailing fields every diagnostics row carries, regardless of
    # which branch produced it - the window bounds are always known (they
    # are call arguments, not derived from the fit), so they're always
    # written even when no histogram exists for this category yet.
    window_fields = (mass_sb_low, mass_sb_high, mass_sig_low, mass_sig_high)

    for k in range(n_cats):
        cat_mask = (bins == k)
        if not np.any(cat_mask):
            diagnostics.append((k, 0, nbins, "fallback(no-events-in-category)", float("nan"), None, None,
                                 float(factors[k]), *window_fields,
                                 json.dumps(None), json.dumps(None), json.dumps(None)))
            continue  # leave factor = default_factor

        m_k, w_k = m_sel[cat_mask], w_sel[cat_mask]
        n_raw = int(np.sum((m_k >= mass_sb_low) & (m_k < mass_sb_high)))
        h = hist.Hist.new.Reg(nbins, mass_sb_low, mass_sb_high, name="mass").Weight()
        h.fill(m_k, weight=w_k)
        raw_sigwin = w_k[(m_k >= mass_sig_low) & (m_k < mass_sig_high)].sum()

        vals = h.values()
        edges = h.axes[0].edges
        variances = h.variances()
        # Captured up front (rather than only inside the "has a usable fit"
        # branch below) so every category with ANY events - including the
        # "empty-hist" fallback right below - still has its raw histogram
        # saved to the diagnostics CSV; only the two branches with no
        # events at all (checked above/at the top of this function) have
        # nothing to save.
        edges_json = json.dumps(edges.tolist())
        vals_json = json.dumps(vals.tolist())
        variances_json = json.dumps(variances.tolist())

        fit_A, fit_B, chi2_ndof, branch = None, None, float("nan"), "fit"
        if vals.sum() < 1e-10:
            diagnostics.append((k, n_raw, nbins, "fallback(empty-hist)", float("nan"), None, None,
                                 float(factors[k]), *window_fields,
                                 edges_json, vals_json, variances_json))
            continue  # leave factor = default_factor

        centers = 0.5 * (edges[:-1] + edges[1:])
        errs = np.sqrt(np.maximum(variances, 1e-12))

        A0, B0 = max(vals[0], 1e-6), -0.03
        try:
            (A, B), _ = curve_fit(
                expf, centers, vals, p0=[A0, B0], sigma=errs,
                absolute_sigma=True, maxfev=2000,
            )
            fit_A, fit_B = float(A), float(B)
            resid = (vals - expf(centers, A, B)) / errs
            ndof = max(len(vals) - 2, 1)
            chi2_ndof = float(np.sum(resid ** 2) / ndof)
            pred_sig = integral_exp(A, B, mass_sig_low, mass_sig_high) / bin_width
            pred_tot = integral_exp(A, B, mass_sb_low, mass_sb_high) / bin_width
            if pred_tot > 0:
                factors[k] = pred_sig / pred_tot
            else:
                branch = "fallback(pred_tot<=0)"
        except RuntimeError:
            branch = "fallback(fit-failed)"  # keep default_factor

        # No raw MC at all in the signal window: fall back to the naive
        # window-width fraction (overrides the fit result too - same as
        # the reference).
        if raw_sigwin <= 0.0:
            factors[k] = naive_fraction
            branch = "fallback(zero-sigwin-mc)"

        diagnostics.append((k, n_raw, nbins, branch, chi2_ndof, fit_A, fit_B, float(factors[k]),
                             *window_fields, edges_json, vals_json, variances_json))

        if save_fit_plots and plot_dir is not None:
            _save_nonres_fit_diagnostics_plot(
                edges=edges, vals=vals, variances=variances,
                cat=k, n_raw=n_raw, nbins=nbins, branch=branch, chi2_ndof=chi2_ndof,
                factor=float(factors[k]), fit_A=fit_A, fit_B=fit_B,
                mass_sb_low=mass_sb_low, mass_sb_high=mass_sb_high,
                mass_sig_low=mass_sig_low, mass_sig_high=mass_sig_high,
                plot_dir=plot_dir, plot_dtype=plot_dtype, call_label=call_label,
            )

    if diagnostics_csv_path is not None:
        _write_nonres_diagnostics_csv(diagnostics_csv_path, diagnostics)

    return factors.astype(np.float32)


NONRES_DIAGNOSTICS_COLUMNS = [
    "category", "n_raw", "nbins", "branch", "chi2_ndof", "fit_A", "fit_B", "factor",
    "mass_sb_low", "mass_sb_high", "mass_sig_low", "mass_sig_high",
    # bin_edges/bin_values/bin_variances are JSON-encoded lists (or the
    # JSON literal "null" when no histogram was built for this category -
    # see the "no-events"/"no-nonres-events" fallback branches below).
    # Storing them is what makes this CSV alone sufficient to regenerate
    # _save_nonres_fit_diagnostics_plot's histogram+fit+pull figure later
    # (see plot_nonres_sideband_fits.py) with no access to the model,
    # training data, or the original hist.Hist object required - before
    # this, only the FIT SUMMARY (factor/chi2/fit_A/fit_B) was saved, not
    # the underlying histogram itself, so the actual histogram+errorbar
    # panel and the pull values could not be reconstructed after the fact.
    "bin_edges", "bin_values", "bin_variances",
]


def _write_nonres_diagnostics_csv(path, diagnostics):
    """One row per GMM category - see NONRES_DIAGNOSTICS_COLUMNS for the
    full schema and compute_nonres_reweight_factors' docstring for how
    each field is derived. `branch` records which code path produced
    that category's factor (a plain exponential "fit", or which fallback
    fired), so a fit that silently fell back to default_factor/
    naive_fraction is visible in the table instead of looking identical
    to a converged fit."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(NONRES_DIAGNOSTICS_COLUMNS)
        for row in diagnostics:
            writer.writerow(row)
    print(f"[INFO] Saved non-resonant sideband-fit diagnostics (incl. histogram data for later replotting): {path}")


def log_gpu_memory(tag):
    """Prints current/peak GPU memory and host RSS (MiB) at a named point in
    the run. Use this to verify the memory fixes and to set HTCondor's
    request_GPUMemory / request_memory realistically (peak + safety margin)."""
    parts = []
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        parts.append(f"gpu_current={info['current'] / 1024**2:.0f} MiB")
        parts.append(f"gpu_peak={info['peak'] / 1024**2:.0f} MiB")
    except (ValueError, KeyError, AttributeError):
        parts.append("gpu=inactive")
    rss = _host_rss_bytes()
    if rss is not None:
        parts.append(f"host_rss={rss / 1024**2:.0f} MiB")
    peak_rss = _host_peak_rss_bytes()
    if peak_rss is not None:
        parts.append(f"host_rss_peak={peak_rss / 1024**2:.0f} MiB")
    print(f"[MEM][{tag}] " + "  ".join(parts))


def load_raw_dataset(input_path):
    """
    Loads the raw feature / label / weight arrays used by training_utils_tf.py,
    plus the "rel_w" significance weights used by gato_categorization_optimized.py.

    Feature matrices stay as read-only memmaps; they are cast to float32
    batch-by-batch during training instead of being copied into RAM here.
    """
    X_train = np.load(f"{input_path}/X_train.npy", mmap_mode="r")
    X_val = np.load(f"{input_path}/X_val.npy", mmap_mode="r")
    y_train = np.load(f"{input_path}/y_train.npy", mmap_mode="r")
    y_val = np.load(f"{input_path}/y_val.npy", mmap_mode="r")

    # one-hot -> class index, same as training_utils_tf.py
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val = np.argmax(y_val, axis=1)

    rel_w_train = np.load(f"{input_path}/rel_w_train.npy", mmap_mode="r")
    rel_w_val = np.load(f"{input_path}/rel_w_val.npy", mmap_mode="r")
    if rel_w_train.ndim > 1:
        rel_w_train = np.squeeze(rel_w_train)
    if rel_w_val.ndim > 1:
        rel_w_val = np.squeeze(rel_w_val)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "rel_w_train": rel_w_train,
        "rel_w_val": rel_w_val,
    }


def load_mass_split(input_path):
    """
    Loads the mass column for the train/val split already produced by
    prepare_training_data.py (mass_train.npy / mass_val.npy), row-aligned
    with X_train.npy/X_val.npy (and therefore with raw_data["X_train"]/
    raw_data["X_val"]) - used for the Stage-B mass-window weighting
    (training.mass_window_weighting in the config).
    """
    mass_train_path = f"{input_path}/mass_train.npy"
    mass_val_path = f"{input_path}/mass_val.npy"
    for p in (mass_train_path, mass_val_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"preselection or training.mass_window_weighting is enabled, but '{p}' does not exist. "
                f"It must be produced by the same prepare_training_data.py run as X_train.npy/X_val.npy."
            )
    mass_train = np.asarray(np.load(mass_train_path, mmap_mode="r"), dtype=np.float32)
    mass_val = np.asarray(np.load(mass_val_path, mmap_mode="r"), dtype=np.float32)
    return mass_train, mass_val


def load_class_weights(input_path, weight_scheme):
    if weight_scheme == "weighted_abs":
        class_weights_for_training = np.load(f"{input_path}/class_weights_for_training_abs.npy", mmap_mode="r")
    elif weight_scheme == "weighted_only_positive":
        class_weights_for_training = np.load(f"{input_path}/class_weights_only_positive.npy", mmap_mode="r")
    else:
        raise NotImplementedError(f"weight_scheme '{weight_scheme}' is not implemented yet")

    class_weights_for_train_no_absolute = np.load(f"{input_path}/true_class_weights.npy", mmap_mode="r")
    class_weights_for_val = np.load(f"{input_path}/class_weights_for_val.npy", mmap_mode="r")
    return {
        "cw_train": class_weights_for_training,
        "cw_train_no_abs": class_weights_for_train_no_absolute,
        "cw_val": class_weights_for_val,
    }


def build_raw_dataset_from_samples(base_dir, data_config, fraction, seed, eras,
                                    train_fraction, val_fraction, weight_scheme,
                                    class_groups="auto", verbose=True,
                                    info_out_path=None, extra_columns=(),
                                    split_seed=None,
                                    apply_classifier_preselection=False,
                                    dijet_mass_column=CLASSIFIER_PRESELECTION_DIJET_MASS_COLUMN,
                                    lead_mvaid_column=CLASSIFIER_PRESELECTION_LEAD_MVAID_COLUMN,
                                    sublead_mvaid_column=CLASSIFIER_PRESELECTION_SUBLEAD_MVAID_COLUMN):
    """
    In-process replacement for load_raw_dataset() + load_class_weights() +
    load_mass_split() combined: builds the full train/val/test dataset
    directly from individual_samples/ (via dataset_prep.build_training_dataset
    - see that module's docstring for the mechanism-1/mechanism-2 rescaling
    and the sample-stratified split), with NO intermediate .npy files or
    separate prepare_inputs.py run required.

    This is what lets nonres_reweight "handle this by itself": it needs the
    per-event mass column, which previously only existed via a separate
    prepare_inputs.py run producing mass_train.npy/mass_val.npy; now it is
    built right here, every run, alongside everything else.

    base_dir: individual_samples/ root (era/sample subdirectories).
    data_config: the loaded YAML dict with sample_to_class, classes,
    weight_scheme_process, class_groups (e.g. training_config_edit.yaml).
    weight_scheme: training_config["class_weight_scheme"] ("weighted_abs" or
    "weighted_only_positive") - selects which of the two Stage-A classifier
    weight arrays becomes cw_train, same as load_class_weights().
    info_out_path: optional path - if given, the same sampling/split/weight
    info that's printed to the terminal (via build_training_dataset) is also
    written there as JSON, in the same call (no re-reading/re-sampling of
    individual_samples/).

    seed: the FRACTION-SAMPLING seed (dataset_prep.build_dataset's
    fraction_seed) - which random --fraction subset of each era/sample
    file's rows is drawn at all.
    split_seed: OPTIONAL, separate seed for the train/val/test split
    (dataset_prep.build_training_dataset's split_seed). Defaults to `seed`
    if not given (old behavior: one seed drives both). Pass a different
    split_seed while keeping `seed` fixed to hold the exact same
    --fraction-sampled event pool fixed across runs and vary ONLY the
    train/val/test partitioning of that pool - this is what pretrain.py
    exposes as seeds.data (-> seed here) / seeds.data_split (-> split_seed
    here), so a "sampling stability" test can isolate the split's effect
    from the effect of which random subset of the full sample got drawn.

    Returns (raw_data, class_weights, mass_train, mass_val, mass_test, classes,
    split_rescale_factors) - same 7-tuple as always (an 8th, split_rescale_
    factors, was added for the event_counts "combined" table - see
    build_event_count_tables - existing callers that only unpack the first
    6 values are unaffected):
    raw_data/class_weights have exactly the same keys/shapes
    load_raw_dataset()/load_class_weights() would have produced from a
    cached directory (y_* converted one-hot -> class index, rel_w_*
    squeezed to 1-D, X_* cast to float32), so everything downstream in
    end_to_end_training.py is unchanged. mass_train/mass_val are float32
    arrays row-aligned with X_train/X_val (pre-shuffle order - the caller
    still applies shuffle_raw_data()'s permutation to them, same as it
    always has for the cached-file path).

    The test split (mechanism 1 + mechanism 2 rescaled, exactly like train/
    val) IS also returned now: raw_data["X_test"/"y_test"/"rel_w_test"],
    class_weights["cw_test"], and mass_test. Stage A/B never train or
    select on it - it is there only for the test-set evaluation
    (evaluate_on_test_set / evaluate_model.py), which needs the SAME
    rescaled weights dataset_prep.py already computed for it, not a
    separately-cached copy.

    split_rescale_factors: {"train": array, "val": array, "test": array} -
    dataset_prep.build_training_dataset's "*_rescale_factors" outputs
    (mechanism-2's rescale factor actually applied to each row of that
    split, one entry per row - see rescale_split_weights_per_key), passed
    through unchanged. Only consumed by build_event_count_tables, to UNDO
    mechanism 2 when building its "combined" (train+val+test) event_counts
    table - see that function's docstring for why the undo is needed.

    apply_classifier_preselection/dijet_mass_column/lead_mvaid_column/
    sublead_mvaid_column: passed straight through to dataset_prep.
    build_training_dataset - see that function's docstring. Off by default
    (existing callers, e.g. combined_optimization.py, are unaffected);
    pretrain.py turns this on (via dataset_prep.classifier_preselection_
    kwargs) to apply the classifier (Stage A) preselection - a DIFFERENT,
    fixed cut from the "gato model" (Stage B) preselection applied
    downstream in combined_optimization.py itself.
    """
    result = build_training_dataset(
        base_dir, data_config, fraction, seed, eras,
        train_fraction=train_fraction, val_fraction=val_fraction,
        class_groups=class_groups, verbose=verbose,
        info_out_path=info_out_path, extra_columns=extra_columns,
        split_seed=split_seed,
        apply_classifier_preselection=apply_classifier_preselection,
        dijet_mass_column=dijet_mass_column,
        lead_mvaid_column=lead_mvaid_column,
        sublead_mvaid_column=sublead_mvaid_column,
    )

    y_train = np.asarray(result["y_train"])
    y_val = np.asarray(result["y_val"])
    y_test = np.asarray(result["y_test"])
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val = np.argmax(y_val, axis=1)
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    rel_w_train = np.asarray(result["rel_w_train"])
    rel_w_val = np.asarray(result["rel_w_val"])
    rel_w_test = np.asarray(result["rel_w_test"])
    if rel_w_train.ndim > 1:
        rel_w_train = np.squeeze(rel_w_train)
    if rel_w_val.ndim > 1:
        rel_w_val = np.squeeze(rel_w_val)
    if rel_w_test.ndim > 1:
        rel_w_test = np.squeeze(rel_w_test)

    raw_data = {
        "X_train": np.asarray(result["X_train"], dtype=np.float32),
        "X_val": np.asarray(result["X_val"], dtype=np.float32),
        "X_test": np.asarray(result["X_test"], dtype=np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "rel_w_train": rel_w_train.astype(np.float32),
        "rel_w_val": rel_w_val.astype(np.float32),
        "rel_w_test": rel_w_test.astype(np.float32),
        # Each split's rel_w exactly as it was right after mechanism 1,
        # BEFORE mechanism 2's split-rescale - carried through the split
        # unchanged (dataset_prep.build_training_dataset's rel_w_premech2_*)
        # rather than derived. Concatenating these three directly gives the
        # full, pre-split dataset's weights row-for-row - see
        # build_event_count_tables' combined table.
        "rel_w_premech2_train": np.asarray(result["rel_w_premech2_train"], dtype=np.float32),
        "rel_w_premech2_val": np.asarray(result["rel_w_premech2_val"], dtype=np.float32),
        "rel_w_premech2_test": np.asarray(result["rel_w_premech2_test"], dtype=np.float32),
    }
    # Extra preselection columns (dijet_mass, lead/sublead mvaID, is_boosted,
    # ...), if any were requested - row-aligned with X_train/X_val/X_test
    # exactly like mass_train/mass_val/mass_test.
    for c in extra_columns:
        raw_data[f"{c}_train"] = np.asarray(result[f"{c}_train"])
        raw_data[f"{c}_val"] = np.asarray(result[f"{c}_val"])
        raw_data[f"{c}_test"] = np.asarray(result[f"{c}_test"])

    if weight_scheme == "weighted_abs":
        cw_train = result["class_weights_abs"]
    elif weight_scheme == "weighted_only_positive":
        cw_train = result["class_weights_only_positive"]
    else:
        raise NotImplementedError(f"weight_scheme '{weight_scheme}' is not implemented yet")

    class_weights = {
        "cw_train": np.asarray(cw_train),
        "cw_train_no_abs": np.asarray(result["true_class_weights"]),
        "cw_val": np.asarray(result["class_weights_val"]),
        "cw_test": np.asarray(result["class_weights_test"]),
    }

    mass_train = np.asarray(result["mass_train"], dtype=np.float32)
    mass_val = np.asarray(result["mass_val"], dtype=np.float32)
    mass_test = np.asarray(result["mass_test"], dtype=np.float32)

    split_rescale_factors = {
        "train": np.asarray(result["train_rescale_factors"]),
        "val": np.asarray(result["val_rescale_factors"]),
        "test": np.asarray(result["test_rescale_factors"]),
    }

    return raw_data, class_weights, mass_train, mass_val, mass_test, result["classes"], split_rescale_factors


# --------------------------------------------------------------------------
# The combined, end-to-end model
# --------------------------------------------------------------------------
class GMMFromScores(gato_gmm_model):
    """
    Identical to the GMM wrapper in gato_categorization_optimized.py: turns a
    dict of {"signal": {...}, "background": {...}} tensors into an Asimov
    significance loss. Kept as its own class so it can be swapped out or
    reused independently of the classifier.
    """

    def __init__(self, n_cats, dim, temperature=0.3, name="GMMFromScores"):
        super().__init__(
            n_cats=n_cats,
            dim=dim,
            temperature=temperature,
            mean_norm="softmax",
            name=name,
        )

    @tf.function
    def call(self, data_dict, rewt=None):
        """
        `rewt`, if given (a length-n_cats tensor), rescales ONLY the
        "background_nonres" bucket's yield contribution to B - see
        compute_nonres_reweight_factors. Every other bucket ("signal",
        "background", "background_other") is unaffected, matching
        gato_utils.py's compute_nonres_reweight_factors_3d semantics
        (resonant processes always get fac == 1).

        Also accumulates B_w2 (sum of squared weights per category,
        soft-assignment-weighted: sum_i gamma_ik * w_i^2), the MC-
        statistical-uncertainty input needed by
        gatohep.losses.high_bkg_uncertainty_penalty. Same fac/rewt
        treatment as B itself (fac multiplies the yield; the correct
        scaling for a sum-of-squares quantity under a per-event weight
        rescale w_i -> fac*w_i is fac**2 * sum(w_i^2), not fac*sum(w_i^2) -
        so `rewt` is squared here, unlike for B's own `rewt` factor).
        Purely additive over `proc`, same as S/B, so this is safe to call
        chunk-by-chunk and sum, exactly like S/B - see chunk_yields'
        docstring for the underlying additivity argument, which applies
        identically to a soft-assignment-weighted sum of w_i^2.
        """
        S = tf.zeros(self.n_cats, dtype=tf.float32)
        B = tf.zeros(self.n_cats, dtype=tf.float32)
        B_w2 = tf.zeros(self.n_cats, dtype=tf.float32)

        for proc, t in data_dict.items():
            X = t["NN_output"]
            w = t["weight"]
            memberships = self.get_probs(X)  # (N, n_cats)

            # Optional soft mass window applied AFTER the GMM assignment
            # (mass_window_weighting.position = "after_gmm"). The per-event
            # category yields are formed first, then damped by the window
            # factor. For position="before_gmm" this key is absent because
            # the factor was already folded into `w` upstream - see
            # EndToEndClassifierGMM.build_data_dict and the note there on
            # why the two give identical numbers.
            yields = tf.reduce_sum(memberships * w[:, None], axis=0)
            sumw2 = tf.reduce_sum(memberships * tf.square(w)[:, None], axis=0)

            if proc == "signal":
                S += yields
                # B_w2 is a background-only quantity (matches
                # high_bkg_uncertainty_penalty's own bkg_yields/bkg_sumsq
                # convention) - signal never contributes to it.
            elif proc == "background_nonres" and rewt is not None:
                B += yields * rewt
                B_w2 += sumw2 * tf.square(rewt)
            else:
                B += yields
                B_w2 += sumw2

        z_per_cat = asymptotic_significance(S, B)
        Z = tf.sqrt(tf.reduce_sum(z_per_cat**2))
        loss = -Z
        return loss, S, B, Z, B_w2


class EndToEndClassifierGMM(tf.keras.Model):
    """
    Wraps an MLP classifier and a gato GMM categorizer into a single model so
    both sets of trainable variables live under one roof (useful for
    checkpointing / trainable_variables bookkeeping), while still exposing
    each sub-model for staged (pretrain-then-finetune) training.

    mass_window_cfg (optional): if given and mass_window_cfg["enabled"] is
    True, adds two scalar variables, self.mass_low/self.mass_high
    (initialized from mass_window_cfg["init_low"]/["init_high"]), and a
    fixed self.mass_softness. Whenever a `mass` tensor is passed in, the
    differentiable window factor

        f(m) = sigmoid((m - low)/softness) * sigmoid((high - m)/softness)

    is applied to the per-event significance weights. This is the m_gammagamma
    selection: unlike the rough preselection (a genuine hard cut applied once
    to the data before training), it sits INSIDE the graph so its edges stay
    differentiable and can be optimized.

    drop_last_class (optional, default False): softmax over n_classes
    outputs always sums to 1, so the last column is redundant - it equals
    1 - sum(the other n_classes-1 columns). If True, .scores() drops that
    last column before handing the result to the GMM, so `gato` only ever
    sees an (n_classes-1)-dimensional embedding instead of n_classes (see
    training_config.yaml's gato_model.drop_last_softmax_class - when set,
    gato_model.dim MUST be n_classes-1, checked at startup in
    end_to_end_training.py). This does NOT change the classifier itself:
    self.classifier still outputs n_classes raw logits (needed for the
    softmax to normalize correctly, and for the per-class cross-entropy
    loss / build_data_dict's masking-by-class below), and the softmax is
    still computed over all n_classes logits internally - only the
    (already-computed) LAST softmax probability is discarded afterward.
    Differentiability is unaffected: the discarded column still
    contributes to the softmax's shared normalization denominator, so
    gradients still reach every one of the classifier's n_classes output-
    layer weights, including the ones "feeding" the dropped column - this
    was verified numerically (autodiff vs. finite differences, matching to
    ~1e-10) before this option was added; see the PR/commit message for
    the toy reproduction. The actual compute saving comes from `gato`
    doing its per-category Gaussian-mixture math (covariance handling
    etc., typically O(dim^2)-O(dim^3)) in one fewer dimension - the
    classifier's own forward/backward cost is essentially unchanged, since
    it still has to produce and normalize all n_classes logits.

    Three further knobs:

    * mass_window_cfg["trainable"] (default True) - whether mass_low/mass_high
      are optimized jointly with the classifier/GMM during Stage B. If False,
      the window is applied every forward pass exactly as above but stays
      fixed at its init values (no gradients computed/applied, see training.py).

    * mass_window_cfg["position"] (default "before_gmm") - "before_gmm" folds
      f(m) into the per-event weight here, so the GMM sees already-damped
      weights; "after_gmm" leaves `weight` untouched and passes f(m) along
      under the "mass_weight" key, so GMMFromScores applies it to the
      per-event category yields AFTER the soft assignment. See the note in
      build_data_dict below: these two are algebraically identical.

    * mass_window_cfg["classes"] (default None = all classes) - list of class
      indices the window applies to. Mirrors the reference script
      gato_categorization_vbf_only_ggh.py, which mass-windows only the
      RESONANT processes and leaves the continuum background untouched.
      Events of any other class get f = 1.

    If mass_window_cfg is None/disabled, self.mass_window_enabled is False
    and passing `mass` anywhere is simply ignored (identical to the original
    behavior).
    """

    def __init__(self, classifier, gato, signal_class, mass_window_cfg=None, nonres_reweight_cfg=None,
                 drop_last_class=False):
        super().__init__(name="EndToEndClassifierGMM")
        self.classifier = classifier
        self.gato = gato
        self.signal_class = signal_class
        self.drop_last_class = bool(drop_last_class)

        # nonres_reweight_cfg (optional): if given and ["enabled"] is True,
        # build_data_dict splits the background bucket further into
        # "background_nonres" (events of class nonres_reweight_cfg
        # ["nonres_bkg_class"]) and "background_other" (everything else
        # non-signal), so a periodically-recomputed per-GMM-category factor
        # (see compute_nonres_reweight_factors, train_complete_model in
        # training.py) can rescale ONLY the non-resonant-continuum yield
        # contribution to the significance loss - mirroring
        # gato_utils.py's compute_nonres_reweight_factors_3d, which leaves
        # resonant processes' factor == 1. Also see mass_window_factor
        # below: whenever this is enabled, the soft mgg window is
        # automatically skipped for the non-resonant class specifically -
        # applying both would double-suppress it.
        nonres_reweight_cfg = nonres_reweight_cfg or {}
        self.nonres_reweight_enabled = bool(nonres_reweight_cfg.get("enabled", False))
        self.nonres_class = (
            int(nonres_reweight_cfg["nonres_bkg_class"]) if self.nonres_reweight_enabled else None
        )

        mass_window_cfg = mass_window_cfg or {}
        self.mass_window_enabled = bool(mass_window_cfg.get("enabled", False))
        self.mass_window_classes = None
        if self.mass_window_enabled:
            self.mass_low = tf.Variable(
                float(mass_window_cfg["init_low"]), trainable=False,
                dtype=tf.float32, name="mass_low",
            )
            self.mass_high = tf.Variable(
                float(mass_window_cfg["init_high"]), trainable=False,
                dtype=tf.float32, name="mass_high",
            )
            self.mass_softness = float(mass_window_cfg.get("softness", 5.0))
 
            classes = mass_window_cfg.get("classes", None)
            if classes is not None:
                classes = [int(c) for c in classes]
                if not classes:
                    raise ValueError(
                        "mass_window_weighting.classes was given but is empty - use "
                        "null/omit it to apply the window to all classes."
                    )
                self.mass_window_classes = tf.constant(classes, dtype=tf.int32)

    def scores(self, X, training=False):
        """Raw features -> softmax class scores. This is the differentiable
        hand-off point between the classifier and the GMM.

        If self.drop_last_class, the softmax is still computed over ALL
        n_classes logits (required for correct normalization), then the
        last column is discarded before returning - see this class's
        drop_last_class docstring for why that's still fully
        differentiable end-to-end."""
        logits = self.classifier(X, training=training)
        full_scores = tf.nn.softmax(logits, axis=1)
        if self.drop_last_class:
            return full_scores[:, :-1], logits
        return full_scores, logits

    def mass_window_factor(self, y, mass):
        """
        Per-event soft mass-window factor f(m) in (0, 1), or None if the
        window is disabled / no mass column was supplied.

        If self.mass_window_classes is set, f is forced to exactly 1 for
        every event whose class is not in that list - i.e. the window only
        selects the listed (typically resonant) processes, matching
        gato_categorization_vbf_only_ggh.py's `if proc in resonant_processes`.

        If self.nonres_reweight_enabled, f is ALSO forced to exactly 1 for
        every event of self.nonres_class (the non-resonant continuum
        background), regardless of mass_window_classes above. Rationale:
        the non-resonant class's contribution to B is computed from its
        FULL, un-windowed yield multiplied by the fitted per-category
        sideband factor `rewt` (see GMMFromScores.call and
        compute_nonres_reweight_factors) - `rewt` IS the estimate of what
        fraction of that full yield falls in the signal window, fit from a
        smooth exponential sideband model rather than trusted directly
        from (often low-statistics) MC counts in the narrow window itself.
        If the soft window ALSO suppressed those same events before `rewt`
        is applied, the non-resonant background would be shrunk TWICE -
        once by the window, once by the fit - artificially shrinking B and
        inflating the significance loss. So whenever nonres reweighting is
        actually being used for this class, the window must not touch it;
        the two are two different ways of restricting the same background
        to its signal-window-equivalent yield, and only one may apply.
        Whenever nonres_reweight_enabled is False, this exemption does NOT
        apply - the non-resonant class is windowed exactly like every
        other class (i.e. never silently left with NO cut at all: turning
        the reweight off always falls back to the ordinary soft window).
        """
        if mass is None or not self.mass_window_enabled:
            return None
        f = soft_mass_window_weight(
            mass, self.mass_low, self.mass_high, self.mass_softness
        )
        y_i = tf.cast(tf.reshape(y, [-1]), tf.int32)
        if self.mass_window_classes is not None:
            in_window_classes = tf.reduce_any(
                tf.equal(y_i[:, None], self.mass_window_classes[None, :]), axis=1
            )
            f = tf.where(in_window_classes, f, tf.ones_like(f))
        if self.nonres_reweight_enabled:
            is_nonres = tf.equal(y_i, self.nonres_class)
            f = tf.where(is_nonres, tf.ones_like(f), f)
        return f

    def build_data_dict(self, X, y, weights, mass=None, training=False):
        """
        Splits one batch into the buckets the GMM consumes: "signal", and
        either a single "background" bucket, or - when nonres reweighting
        is enabled - "background_nonres" (events of self.nonres_class)
        split out from "background_other" (everything else non-signal).
        Optionally attaches the soft mass-window factor.
        """
        scores, logits = self.scores(X, training=training)
        weights = tf.cast(weights, tf.float32)

        f = self.mass_window_factor(y, mass)
        if f is not None:
            weights = weights * f
            f = None  # already folded in; nothing left to hand downstream

        signal_mask = tf.equal(y, self.signal_class)
        bkg_mask = tf.logical_not(signal_mask)

        # NOTE: the "mass_weight" key is omitted entirely (rather than set to
        # None) when it isn't needed - GMMFromScores.call is a @tf.function and
        # a None inside its input structure would be part of the traced
        # signature.
        data_dict = {
            "signal": {
                "NN_output": tf.boolean_mask(scores, signal_mask),
                "weight": tf.boolean_mask(weights, signal_mask),
            },
        }

        if self.nonres_reweight_enabled:
            nonres_mask = tf.logical_and(bkg_mask, tf.equal(y, self.nonres_class))
            other_mask = tf.logical_and(bkg_mask, tf.logical_not(tf.equal(y, self.nonres_class)))
            data_dict["background_nonres"] = {
                "NN_output": tf.boolean_mask(scores, nonres_mask),
                "weight": tf.boolean_mask(weights, nonres_mask),
            }
            data_dict["background_other"] = {
                "NN_output": tf.boolean_mask(scores, other_mask),
                "weight": tf.boolean_mask(weights, other_mask),
            }
        else:
            data_dict["background"] = {
                "NN_output": tf.boolean_mask(scores, bkg_mask),
                "weight": tf.boolean_mask(weights, bkg_mask),
            }

        if f is not None:
            # Vestigial: f is always folded into `weights` above and set to
            # None, so this never actually runs today (see the comment on
            # position="before_gmm" above) - guarded per-bucket so it stays
            # correct if that ever changes.
            data_dict["signal"]["mass_weight"] = tf.boolean_mask(f, signal_mask)
            if "background" in data_dict:
                data_dict["background"]["mass_weight"] = tf.boolean_mask(f, bkg_mask)
            else:
                data_dict["background_nonres"]["mass_weight"] = tf.boolean_mask(f, nonres_mask)
                data_dict["background_other"]["mass_weight"] = tf.boolean_mask(f, other_mask)
        return data_dict, scores, logits

    def call(self, X, y, weights, mass=None, training=False, rewt=None):
        """Full forward pass: raw features -> significance loss.

        `rewt`, if given (length-n_cats tensor), is forwarded to
        GMMFromScores.call and rescales only the "background_nonres"
        bucket's yield contribution - see compute_nonres_reweight_factors.
        """
        data_dict, scores, logits = self.build_data_dict(X, y, weights, mass=mass, training=training)
        sig_loss, S, B, Z, B_w2 = self.gato.call(data_dict, rewt=rewt)
        return sig_loss, S, B, Z, B_w2, scores, logits

    def chunk_yields(self, X, y, weights, mass=None, training=False, rewt=None):
        """
        Like `call`, but stops *before* the Asimov significance Z is formed:
        it only returns the per-category signal/background yield vectors
        S_chunk, B_chunk (plus B_w2_chunk and the raw logits) for whatever
        subset of events (X, y, weights) is passed in.

        Why this is safe to call on an arbitrary chunk of the dataset:
        `self.gato.get_probs(x)` (the GMM's soft category membership) only
        ever looks at a *single* event's own classifier score - it never
        compares one event to another. So the yield contribution of an
        event to S/B does not depend on which other events happen to sit
        in the same chunk. This means:

            S_full = sum_over_all_chunks( S_chunk )
            B_full = sum_over_all_chunks( B_chunk )

        exactly, for any way of partitioning the dataset into chunks. This
        additivity is exactly what `train_complete_model`'s chunked training
        step in training.py relies on to process a large dataset in pieces
        while still computing gradients that are identical to processing
        the whole dataset in one go (see the comments there for details).
        Since a chunk's own S_chunk/B_chunk contribution never depends on
        the other chunks regardless of *which* parameters affect it, this
        additivity - and therefore the chunked-gradient trick - holds just
        as well when self.mass_low/self.mass_high are trainable.

        Note this is *not* true of Z itself (asymptotic_significance is a
        nonlinear function of the aggregated S, B), which is why Z must
        always be computed once from the fully-summed S_full/B_full, never
        chunk-by-chunk and then averaged/summed. high_bkg_uncertainty_penalty
        is likewise a nonlinear function of the aggregated B_w2_full/B_full,
        for the identical reason - see finetune_train_step_chunked's
        docstring for how B_w2_chunk is folded into the SAME gradient-
        caching trick used for Z, just extended to a 3-way (S, B, B_w2)
        aggregate instead of a 2-way (S, B) one.

        B_w2_chunk: sum_i gamma_ik * w_i^2 over this chunk's events, soft-
        assignment-weighted exactly like `yields` above - background-only
        (signal never contributes), same fac**2/rewt**2 scaling as
        GMMFromScores.call (see its docstring for why squared, not linear).

        `rewt`, if given (length-n_cats tensor), rescales this chunk's OWN
        "background_nonres" yield contribution before folding it into
        B_chunk. Since rewt is a fixed (non-trainable) constant for the
        whole epoch - recomputed only every nonres_reweight.recompute_every
        epochs in train_complete_model - this is exactly equivalent to
        applying it once to the fully-summed B_nonres_total
        (rewt * sum_c(B_c) == sum_c(rewt * B_c)), so the additivity
        argument above (and the chunked-gradient trick built on it in
        training.py) is unaffected.
        """
        data_dict, scores, logits = self.build_data_dict(X, y, weights, mass=mass, training=training)
        S = tf.zeros(self.gato.n_cats, dtype=tf.float32)
        B = tf.zeros(self.gato.n_cats, dtype=tf.float32)
        B_w2 = tf.zeros(self.gato.n_cats, dtype=tf.float32)
        for proc, t in data_dict.items():
            Xp = t["NN_output"]
            wp = t["weight"]
            memberships = self.gato.get_probs(Xp)
            # position="before_gmm" (or window disabled): the window
            # factor is already inside wp.
            yields = tf.reduce_sum(memberships * wp[:, None], axis=0)
            sumw2 = tf.reduce_sum(memberships * tf.square(wp)[:, None], axis=0)

            if proc == "signal":
                S = S + yields
            elif proc == "background_nonres" and rewt is not None:
                B = B + yields * rewt
                B_w2 = B_w2 + sumw2 * tf.square(rewt)
            else:
                B = B + yields
                B_w2 = B_w2 + sumw2
        return S, B, B_w2, logits

# Plot / dataframe helpers (adapted from gato_categorization_optimized.py to
# work from raw features + a trained end-to-end model instead of precomputed
# score .npy files)
# --------------------------------------------------------------------------
def make_plot_dataframe_from_model(model, X, y, weights, signal_class, plot_dim, chunk_size=None,
                                    nonres_class=None):
    """
    NOTE on GPU memory: this used to call model.scores() on the *entire*
    dataset X in one shot, which forces one classifier forward pass over
    all N rows at once (all hidden-layer activations for N rows alive in
    GPU memory simultaneously). If chunk_size is given, we instead score
    the data in pieces and immediately pull each piece down to numpy,
    keeping peak GPU memory at O(chunk_size) instead of O(len(X)).

    nonres_class (optional): if given, the non-signal events are further
    split into "background_nonres" (y == nonres_class) and
    "background_other" (everything else non-signal) instead of a single
    "background" bucket - mirrors EndToEndClassifierGMM.build_data_dict's
    bucket split when nonres reweighting is enabled, so the resulting
    per-bucket histograms can have the nonres reweight factors applied to
    ONLY the background_nonres bucket before being recombined (see
    evaluate_on_test_set).
    """
    import pandas as pd

    if chunk_size:
        n = X.shape[0]
        score_chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            s_chunk, _ = model.scores(tf.constant(X[start:end], dtype=tf.float32), training=False)
            score_chunks.append(s_chunk.numpy())
        scores = np.concatenate(score_chunks, axis=0)[:, :plot_dim]
    else:
        scores, _ = model.scores(tf.constant(X, dtype=tf.float32), training=False)
        scores = scores.numpy()[:, :plot_dim]

    signal_mask = y == signal_class
    if nonres_class is None:
        buckets = (("signal", signal_mask), ("background", ~signal_mask))
    else:
        bkg_mask = ~signal_mask
        nonres_mask = bkg_mask & (y == nonres_class)
        other_mask = bkg_mask & (y != nonres_class)
        buckets = (
            ("signal", signal_mask),
            ("background_nonres", nonres_mask),
            ("background_other", other_mask),
        )

    plot_data = {}
    for proc, mask in buckets:
        df = pd.DataFrame(
            {
                "NN_output": [tuple(v) for v in scores[mask]],
                "weight": weights[mask],
            }
        )
        plot_data[proc] = df
    return plot_data


def build_class_score_dataframes(model, X, y, weights, mass, classes, plot_dim, chunk_size=None):
    """
    Per-CLASS analogue of make_plot_dataframe_from_model() above, built for
    the event_counts tables (see make_event_count_table / build_event_count_
    tables below). make_plot_dataframe_from_model() pools every non-signal
    class into one "background" (or background_nonres/background_other)
    bucket, which is what the training-time S/B histograms need; this
    function instead keeps every entry of `classes` (e.g. is_nonRes_bkg /
    is_Res_bkg / is_ggHH_sig / is_VBFHH_sig, see training_config_edit.yaml)
    as its own bucket, mirroring gato_categorization_merged.py's per-
    PROCESS data_dict_MC - just at per-CLASS granularity, since this
    pipeline's raw_data/y only ever carries a class label per event, not
    the underlying physical-sample identity (proc_num/sample_name are not
    threaded through build_raw_dataset_from_samples).

    Scores X in chunks (same GPU-memory-saving pattern as
    make_plot_dataframe_from_model), then obtains each event's HARD
    (argmax) category assignment via the same assign_bins_and_order() used
    everywhere else in this file (evaluate_on_test_set, the final plot in
    end_to_end_training.py), so cat_index here always matches what those
    other diagnostics call "category".

    Returns {class_name: pd.DataFrame(columns=["cat_index", "weights", "massH"])}.
    """
    import pandas as pd

    y = np.asarray(y)
    weights = np.asarray(weights, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float32)

    if chunk_size:
        n = X.shape[0]
        score_chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            s_chunk, _ = model.scores(tf.constant(X[start:end], dtype=tf.float32), training=False)
            score_chunks.append(s_chunk.numpy())
        scores = np.concatenate(score_chunks, axis=0)[:, :plot_dim] if score_chunks else np.zeros((0, plot_dim))
    else:
        scores, _ = model.scores(tf.constant(X, dtype=tf.float32), training=False)
        scores = scores.numpy()[:, :plot_dim]

    plot_data = {}
    for idx, cls in enumerate(classes):
        mask = (y == idx)
        plot_data[cls] = pd.DataFrame(
            {
                "NN_output": [tuple(v) for v in scores[mask]],
                "weight": weights[mask],
                "massH": mass[mask],
            }
        )

    assign, order, _, inv = assign_bins_and_order(model.gato, plot_data, reduce=True)

    class_dfs = {}
    for cls, df in plot_data.items():
        class_dfs[cls] = pd.DataFrame(
            {
                "cat_index": np.asarray(assign[cls]),
                "weights": df["weight"].to_numpy(),
                "massH": df["massH"].to_numpy(),
            }
        )
    return class_dfs


def _asimov_z_scalar(s, b):
    """Plain-Python Asimov significance for a single (s, b) pair - same
    formula/edge-case convention (0 if s<=0 or b<=0) as the local
    asymptotic_significance() redefined inside gato_categorization_merged.py's
    main(), kept separate from the tensor-valued gatohep.utils.
    asymptotic_significance imported at the top of this file (which expects
    array/tensor inputs, not a bare scalar pair)."""
    if s <= 0 or b <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


def make_event_count_table(class_dict_of_df, output_pdf, title="Event Counts by Category",
                            signal_classes=("is_ggHH_sig", "is_VBFHH_sig"),
                            mass_windows=((120, 130), (123, 127))):
    """
    Per-CLASS analogue of gato_categorization_merged.py's make_event_count_pdf
    (see that script's "get event yields" section for the original per-
    PROCESS version this mirrors). Builds a per-category event-count /
    significance PDF table (via reportlab) from a
    {class_name: DataFrame(cat_index, weights, massH)} dict - see
    build_class_score_dataframes() above for how that dict is built.

    Differences from the reference version:
      - rows are per CLASS (is_nonRes_bkg/is_Res_bkg/is_ggHH_sig/
        is_VBFHH_sig), not per physical MC process - see
        build_class_score_dataframes' docstring for why.
      - there is no "Data" row: this pipeline has no real-data sample
        loaded alongside the MC classes (unlike gato_categorization_merged.
        py's df_data_obs), so the Data / Data-sideband rows are dropped.
      - significance rows are computed for every class in `signal_classes`
        against ALL OTHER classes pooled as background, for every window in
        `mass_windows` - by default this reproduces the reference's ggHH /
        VBFHH x [120,130]/[123,127] rows, using this config's own class
        names (see training_config_edit.yaml's `classes` list).

    Returns the table as a pandas DataFrame (same rows/columns written to
    the PDF), same convention as make_event_count_pdf.
    """
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    categories = sorted(
        set(
            int(cat) for df in class_dict_of_df.values()
            for cat in pd.unique(df["cat_index"])
        )
    )

    def weighted_count_and_unc(df):
        w = df["weights"].to_numpy()
        return float(np.sum(w)), float(np.sqrt(np.sum(w ** 2)))

    def fmt_count(count, unc):
        return f"{count:.2f} +/- {unc:.2f}"

    def class_yield(cat, mass_min=None, mass_max=None, exclude=()):
        """Weighted yield (+unc) summed over every class NOT in `exclude`,
        restricted to category `cat` and (if given) windowed to
        [mass_min, mass_max]."""
        total, total_var = 0.0, 0.0
        for cls, df in class_dict_of_df.items():
            if cls in exclude:
                continue
            sel = df[df["cat_index"] == cat]
            if mass_min is not None:
                sel = sel[(sel["massH"] >= mass_min) & (sel["massH"] <= mass_max)]
            w = sel["weights"].to_numpy()
            total += float(np.sum(w))
            total_var += float(np.sum(w ** 2))
        return total, float(np.sqrt(total_var))

    columns = ["Class"] + [f"cat_{cat}" for cat in categories] + ["Quadrature sum"]
    table_rows = [columns]

    # Per-class rows, full (preselected) mass range.
    for cls, df in class_dict_of_df.items():
        row = [cls]
        for cat in categories:
            count, unc = weighted_count_and_unc(df[df["cat_index"] == cat])
            row.append(fmt_count(count, unc))
        row.append("")
        table_rows.append(row)

    # Significance rows: each signal class vs. every other class pooled as background.
    for sig_cls in signal_classes:
        if sig_cls not in class_dict_of_df:
            continue
        for mass_min, mass_max in mass_windows:
            z_values = []
            for cat in categories:
                s, _ = class_yield(
                    cat, mass_min, mass_max,
                    exclude=tuple(c for c in class_dict_of_df if c != sig_cls),
                )
                b, _ = class_yield(cat, mass_min, mass_max, exclude=(sig_cls,))
                z_values.append(_asimov_z_scalar(s, b))
            row = [f"{sig_cls} Z [{mass_min},{mass_max}]"] + [f"{z:.3f}" for z in z_values]
            row.append(f"{np.sqrt(np.sum(np.array(z_values) ** 2)):.3f}")
            table_rows.append(row)

    # Per-class yields within each mass window (every class windowed).
    for mass_min, mass_max in mass_windows:
        table_rows.append(
            [f"--- Class yields in [{mass_min},{mass_max}] (windowed) ---"] + [""] * (len(categories) + 1)
        )
        for cls, df in class_dict_of_df.items():
            row = [f"{cls} [{mass_min},{mass_max}]"]
            for cat in categories:
                sel = df[(df["cat_index"] == cat) & (df["massH"] >= mass_min) & (df["massH"] <= mass_max)]
                count, unc = weighted_count_and_unc(sel)
                row.append(fmt_count(count, unc))
            row.append("")
            table_rows.append(row)

    # Total row (all classes, full mass range).
    total_row = ["Total"]
    for cat in categories:
        count, unc = class_yield(cat)
        total_row.append(fmt_count(count, unc))
    total_row.append("")
    table_rows.append(total_row)

    # Total sideband row (all classes, excl. [120,130]).
    sideband_row = ["Total sideband excl. [120,130]"]
    for cat in categories:
        total, total_var = 0.0, 0.0
        for cls, df in class_dict_of_df.items():
            sel = df[(df["cat_index"] == cat) & ~((df["massH"] >= 120) & (df["massH"] <= 130))]
            w = sel["weights"].to_numpy()
            total += float(np.sum(w))
            total_var += float(np.sum(w ** 2))
        sideband_row.append(fmt_count(total, np.sqrt(total_var)))
    sideband_row.append("")
    table_rows.append(sideband_row)

    # PDF
    doc = SimpleDocTemplate(
        output_pdf, pagesize=landscape(A3),
        rightMargin=10, leftMargin=10, topMargin=20, bottomMargin=20,
    )
    styles = getSampleStyleSheet()
    elements = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    table = Table(table_rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -2), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -2), (-1, -2), colors.whitesmoke),
        ("BACKGROUND", (0, -1), (-1, -1), colors.beige),
        ("FONTSIZE", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(table)
    doc.build(elements)

    return pd.DataFrame(table_rows[1:], columns=table_rows[0])


def build_event_count_tables(
    classifier, gato, signal_class, plot_dim, classes,
    raw_data, mass_train, mass_val, mass_test,
    out_dir, chunk_size=None, nonres_reweight_cfg=None, plot_dtype="pdf",
    drop_last_class=False, data_prep_cfg=None,
):
    """
    Builds and writes the full set of "event_counts" diagnostic tables,
    mirroring gato_categorization_merged.py's make_event_count_pdf calls
    (its "get event yields" and "TEST-SET EVALUATION" sections) but adapted
    to this pipeline's per-CLASS data (see build_class_score_dataframes)
    and its own in-process train/val/test split (dataset_prep.
    build_training_dataset, via build_raw_dataset_from_samples).

    Writes 5 PDFs under out_dir/event_counts/, and returns the matching
    DataFrames as {"combined", "train", "val", "test",
    "test_with_nonres_reweight"} (the last key is absent - so only 4 PDFs
    are written - if nonres_reweight_cfg has no usable "nonres_bkg_class"):

      event_counts_combined.pdf
          Full sample (train+val+test recombined) - see the mechanism-2
          "undo" note below. Per-CLASS analogue of the reference script's
          "combined" table (built there from the full, un-split
          data_dict_MC).
      event_counts_train_only.pdf / event_counts_val_only.pdf /
      event_counts_test_only.pdf
          One table per split, each using that split's own (already
          mechanism-2-rescaled) weights as-is - i.e. each already reports
          full-dataset-equivalent statistics for its own split, same
          convention as the reference script's val-only/test-only tables.
          None of these three apply the non-resonant reweight or narrow the
          mass range - they cover the full preselected range, with
          per-window class yields broken out inside the table itself (see
          make_event_count_table).
      event_counts_test_only_with_nonres_reweight.pdf
          Test split only, FULL preselected mass range (deliberately NO
          [120,130] cut - an earlier version of this function wrongly
          combined the mass cut with the reweight in one table; those are
          two independent things and this table applies ONLY the reweight),
          with the non-resonant background class's weights multiplied by
          the per-category factors fit from the test split's OWN sideband -
          mirrors gato_categorization_merged.py's
          apply_nonres_reweight_to_dict() applied to
          data_dict_MC_test_reweighted (its TEST-SET EVALUATION section).
          The fit itself uses compute_nonres_reweight_factors' plain,
          fixed-nbins mode (see nonres_reweight_cfg["nbins"] in
          training_config.yaml - e.g. 25 instead of 40) - NOT
          nonres_fit_diagnostics.py's adaptive-binning "robust" version.
          Only built if nonres_reweight_cfg has a usable
          "nonres_bkg_class" entry; skipped (with a message, and then only
          4 PDFs total) otherwise.
          Always writes a per-category sideband-fit diagnostic CSV
          (n_raw, nbins, branch, chi2/ndof, fit A/B, factor) AND a
          histogram+fit-curve+pull PDF plot per category, under
          out_dir/event_counts/nonres_fit_diagnostics/ - this is a fit on
          the TEST dataset, so both are unconditional, not config flags
          (unlike the periodic IN-TRAINING refit - see
          train_complete_model's out_dir param in training.py - which
          stays config-gated, since it can run many times over a long
          training run).

    nonres_reweight_cfg: training_config["nonres_reweight"] (or None/{} to
    skip the with-reweight table entirely) - its mass_sb_low/mass_sb_high/
    mass_sig_low/mass_sig_high/nbins/default_factor/nonres_bkg_class fields
    (NOT its "enabled" flag - this table is produced regardless of whether
    nonres reweighting was actually enabled during training, same
    convention as evaluate_on_test_set), plus the two new
    save_fit_diagnostics_csv/save_fit_plots flags above.

    data_prep_cfg: training_config["data_prep"] (its train_fraction/
    val_fraction) - used to scale nonres_reweight_cfg["nbins"] DOWN for
    this table's test-split sideband fit, since test has far fewer events
    than whatever nbins was tuned for on train (see
    scaled_nbins_for_split). None/{} falls back to that function's own
    0.7/0.15 defaults.

    Mechanism-2 undo (for the "combined" table only): raw_data["rel_w_train"]
    /"rel_w_val"/"rel_w_test" already have dataset_prep.py's mechanism-2
    rescale baked in - EACH split's own weight sum has already been
    rescaled (per era/sample - see dataset_prep.rescale_split_weights_
    per_key) to match the FULL, pre-split dataset's weight sum. Concatenating
    train+val+test as-is would therefore roughly TRIPLE-COUNT the full-
    sample weight. Instead of reconstructing the pre-rescale weight by
    dividing the rescale back out, raw_data["rel_w_premech2_train"/"_val"/
    "_test"] already hold each split's weight EXACTLY as it was right after
    mechanism 1 (fraction-sampling), carried through the split unchanged
    (see dataset_prep.build_training_dataset) - concatenating those three
    directly reproduces the true full-sample statistics, with no rescale
    arithmetic involved.
    """
    import pandas as pd

    event_counts_dir = os.path.join(out_dir, "event_counts")
    os.makedirs(event_counts_dir, exist_ok=True)

    scoring_model = EndToEndClassifierGMM(
        classifier, gato, signal_class=signal_class,
        mass_window_cfg=None, nonres_reweight_cfg=None,
        drop_last_class=drop_last_class,
    )

    results = {}

    # ---- per-split tables (mechanism-2-rescaled weights, used as-is) ----
    split_specs = [
        ("train", raw_data["X_train"], raw_data["y_train"], raw_data["rel_w_train"], mass_train),
        ("val", raw_data["X_val"], raw_data["y_val"], raw_data["rel_w_val"], mass_val),
        ("test", raw_data["X_test"], raw_data["y_test"], raw_data["rel_w_test"], mass_test),
    ]
    class_dfs_by_split = {}
    for split_name, X, y_idx, w, mass in split_specs:
        # X is passed through as-is (may be a zero-copy PermutedView over a
        # large memmapped feature matrix) - build_class_score_dataframes
        # reads/casts it chunk-by-chunk internally when chunk_size is given,
        # same memory-saving pattern as make_plot_dataframe_from_model.
        class_dfs = build_class_score_dataframes(
            scoring_model, X, y_idx, w, mass,
            classes, plot_dim, chunk_size=chunk_size,
        )
        class_dfs_by_split[split_name] = class_dfs
        df = make_event_count_table(
            class_dfs,
            output_pdf=os.path.join(event_counts_dir, f"event_counts_{split_name}_only.pdf"),
            title=f"Event Counts by Category ({split_name} split only)",
        )
        results[split_name] = df
        print(f"[INFO] Saved event_counts table ({split_name} split): "
              f"{os.path.join(event_counts_dir, f'event_counts_{split_name}_only.pdf')}")

    # ---- combined table: recombine all 3 splits using the pre-mechanism-2
    # weight each split carried through the split unchanged (raw_data
    # ["rel_w_premech2_*"] - see dataset_prep.build_training_dataset), so no
    # rescale-and-divide-back-out arithmetic is needed here at all.
    rel_w_train_premech2 = np.asarray(raw_data["rel_w_premech2_train"], dtype=np.float64)
    rel_w_val_premech2 = np.asarray(raw_data["rel_w_premech2_val"], dtype=np.float64)
    rel_w_test_premech2 = np.asarray(raw_data["rel_w_premech2_test"], dtype=np.float64)

    # Zero-copy 3-way concat (same ConcatView used for X_all at the end of
    # end_to_end_training.py) instead of np.concatenate, so combining the
    # three splits doesn't materialize a second full copy of X in RAM -
    # build_class_score_dataframes reads it chunk-by-chunk below anyway.
    X_combined = ConcatView(raw_data["X_train"], raw_data["X_val"], raw_data["X_test"])
    y_combined = np.concatenate(
        [np.asarray(raw_data["y_train"]), np.asarray(raw_data["y_val"]), np.asarray(raw_data["y_test"])], axis=0
    )
    w_combined = np.concatenate([rel_w_train_premech2, rel_w_val_premech2, rel_w_test_premech2], axis=0)
    mass_combined = np.concatenate(
        [np.asarray(mass_train), np.asarray(mass_val), np.asarray(mass_test)], axis=0
    )

    class_dfs_combined = build_class_score_dataframes(
        scoring_model, X_combined, y_combined, w_combined, mass_combined,
        classes, plot_dim, chunk_size=chunk_size,
    )
    combined_path = os.path.join(event_counts_dir, "event_counts_combined.pdf")
    results["combined"] = make_event_count_table(
        class_dfs_combined, output_pdf=combined_path,
        title="Event Counts by Category (combined: train+val+test)",
    )
    print(f"[INFO] Saved event_counts table (combined): {combined_path}")

    # ---- test-only, WITH nonres reweight, FULL range (no mass cut) ----
    # This is the ONE special test-set table (5 total, alongside combined/
    # train/val/test above): full preselected mass range, no [120,130] cut
    # applied - the reweight and the mass cut are two independent
    # operations, and this table applies ONLY the reweight (an earlier
    # version of this function wrongly applied both to one table, then a
    # second wrongly kept both a mass-cut-only AND a reweight-only table,
    # 6 total - this replaces both with the single correct table). The
    # ONLY transformation relative to the plain "test_only" table above is
    # multiplying the non-resonant background class's weights by the
    # per-category sideband-fit factors. See docstring above for the full
    # rationale.
    nonres_reweight_cfg = nonres_reweight_cfg or {}
    if nonres_reweight_cfg and "nonres_bkg_class" in nonres_reweight_cfg:
        nonres_class_idx = int(nonres_reweight_cfg["nonres_bkg_class"])
        nonres_class_name = classes[nonres_class_idx]

        diagnostics_dir = os.path.join(event_counts_dir, "nonres_fit_diagnostics")

        factors = compute_nonres_reweight_factors(
            scoring_model, raw_data["X_test"], raw_data["y_test"], raw_data["rel_w_test"], mass_test,
            nonres_class=nonres_class_idx, n_cats=gato.n_cats,
            mass_sb_low=nonres_reweight_cfg.get("mass_sb_low", 100.0),
            mass_sb_high=nonres_reweight_cfg.get("mass_sb_high", 180.0),
            mass_sig_low=nonres_reweight_cfg.get("mass_sig_low", 120.0),
            mass_sig_high=nonres_reweight_cfg.get("mass_sig_high", 130.0),
            nbins=resolve_nonres_nbins(nonres_reweight_cfg, "test", data_prep_cfg),
            default_factor=float(nonres_reweight_cfg.get("default_factor", 0.062)),
            chunk_size=chunk_size,
            # Always saved, not config-gated (see build_event_count_tables'
            # docstring) - this is a fit on the TEST dataset, so both the
            # diagnostic CSV and the per-category plots are unconditional.
            diagnostics_csv_path=os.path.join(diagnostics_dir, "nonres_fit_diagnostics_test.csv"),
            save_fit_plots=True,
            plot_dir=diagnostics_dir,
            plot_dtype=plot_dtype,
            call_label="event_counts_test_with_reweight",
        )
        print(f"[INFO] Non-resonant reweight factors (fit on TEST sideband, full range): {factors}")

        # Apply factors[cat_index] to ONLY the nonres-bkg class's weights,
        # post-hoc on the hard cat_index already assigned in
        # class_dfs_by_split["test"] (built above, full range, no cut) -
        # mirrors gato_categorization_merged.py's
        # apply_nonres_reweight_to_dict.
        class_dfs_test_reweighted = {cls: df.copy() for cls, df in class_dfs_by_split["test"].items()}
        nonres_df = class_dfs_test_reweighted[nonres_class_name]
        cat_idx = nonres_df["cat_index"].to_numpy()
        new_w = nonres_df["weights"].to_numpy().astype(np.float64)
        valid = (cat_idx >= 0) & (cat_idx < len(factors))
        new_w[valid] = new_w[valid] * np.asarray(factors)[cat_idx[valid]]
        nonres_df["weights"] = new_w

        with_reweight_path = os.path.join(event_counts_dir, "event_counts_test_only_with_nonres_reweight.pdf")
        results["test_with_nonres_reweight"] = make_event_count_table(
            class_dfs_test_reweighted, output_pdf=with_reweight_path,
            title="Event Counts by Category (test split, full range, WITH non-resonant reweight)",
        )
        print(
            f"[INFO] Saved event_counts table (test split, full range, with nonres reweight): "
            f"{with_reweight_path}"
        )
    else:
        print(
            "[INFO] event_counts: no usable nonres_reweight config (missing or no "
            "nonres_bkg_class) - skipping the 'test_with_nonres_reweight' table."
        )

    return results


def plot_category_binning(model, X, y, out_path, title):
    scores, _ = model.scores(tf.constant(X, dtype=tf.float32), training=False)
    scores = scores.numpy()
    if scores.shape[1] < 2:
        scores = np.pad(scores, ((0, 0), (0, 2 - scores.shape[1])), mode="constant")

    # print Info
    print("Info on assignment operation")
    print(f"x: {type(scores[:, :2])}, {scores[:, :2].shape}")
    print(type(tf.constant(tf.constant(scores[:, :2], dtype=tf.float32))))
    print(tf.constant(tf.constant(scores[:, :2], dtype=tf.float32)).shape)

    assignments = np.argmax(model.gato.get_probs(tf.constant(scores[:, :2], dtype=tf.float32)).numpy(), axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(scores[:, 0], scores[:, 1], c=assignments, cmap="tab10", s=18, alpha=0.4, edgecolors="none")
    if hasattr(model.gato, "means"):
        means = model.gato.means.numpy() if hasattr(model.gato.means, "numpy") else model.gato.means
        means = np.asarray(means)
        if means.ndim == 2 and means.shape[1] >= 2:
            ax.scatter(means[:, 0], means[:, 1], c="black", marker="x", s=80, linewidths=1.5)
    ax.set_xlabel("score 1")
    ax.set_ylabel("score 2")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def get_unique_out_dir(base_out_dir):
    """
    If base_out_dir doesn't exist yet, returns it unchanged. Otherwise
    appends an ascending integer directly to the last path component until
    a path is found that doesn't exist yet, e.g. "test" -> "test1" ->
    "test2" ... This guarantees every fresh training run gets its own
    output folder and never silently overwrites a previous run's outputs.

    Not applied when continue_training.enabled is true (see main()) - that
    mode deliberately points out-dir at an existing folder to write into.
    """
    base_out_dir = base_out_dir.rstrip("/")
    if not os.path.exists(base_out_dir):
        return base_out_dir
    i = 1
    while True:
        candidate = f"{base_out_dir}{i}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def build_run_name(training_config):
    """
    Descriptive run-folder name encoding the settings that most affect
    what a run actually IS, so out-dir names are self-explanatory without
    opening training_config.yaml, e.g.:

        3D_010_500_epochs_5_bins_soft_class_loss

    Fields, in order (only the trailing two are conditional - omitted
    entirely for the common/default case, matching the example above,
    which has neither):

      1. "{dim}D"            - gato_model.dim (the GMM's actual working
                                dimensionality - see gmm_contour_diagnostics.py's
                                module docstring for why this is
                                n_classes - 1, not n_classes, when
                                drop_last_softmax_class is on).
      2. "{pct:03d}"          - data_prep.fraction as a zero-padded
                                percentage (1.0 -> "100", 0.10 -> "010").
      3. "{n_epochs}_epochs"  - training.n_epochs (Stage B/finetuning
                                epochs - the number usually varied
                                between runs; Stage A's separate
                                pretrain.epochs isn't encoded here).
      4. "{n_cats}_bins"      - gato_model.n-cats (number of GMM
                                categories).
      5. loss label           - "soft_class_loss" if training.ce_mode ==
                                "multiclass" (the classifier is pulled
                                towards its full n-way soft distribution,
                                same loss Stage A uses), else
                                "hard_class_loss" (ce_mode in
                                "binary_plain"/"binary_grouped" - CE is
                                collapsed to a hard signal-vs-background
                                decision; see make_ce_loss_fn's docstring
                                in training.py). This mapping matches the
                                example string above (current config's
                                ce_mode="multiclass" -> "soft_class_loss")
                                - flag it if that's not what you meant by
                                "soft/hard class loss" and I'll remap it.
      6. "continued"          - appended only if continue_training.enabled
                                is true (this run resumes from a previous
                                run's weights/checkpoint).

    Does NOT itself guarantee uniqueness - pass the result through
    get_unique_out_dir (see build_descriptive_out_dir) so two runs with
    identical settings still get separate folders (name -> name1 -> ...).
    """
    gato_cfg = training_config.get("gato_model", {}) or {}
    train_cfg = training_config.get("training", {}) or {}
    data_prep_cfg = training_config.get("data_prep", {}) or {}
    continue_cfg = training_config.get("continue_training", {}) or {}

    dim = int(gato_cfg["dim"])
    n_cats = int(gato_cfg["n-cats"])

    fraction = float(data_prep_cfg.get("fraction", 1.0))
    pct = int(round(fraction * 100))

    n_epochs = int(train_cfg.get("n_epochs", 0))

    ce_mode = str(train_cfg.get("ce_mode", "binary_plain")).lower()
    loss_label = "soft_class_loss" if ce_mode == "multiclass" else "hard_class_loss"

    parts = [
        f"{dim}D",
        f"{pct:03d}",
        f"{n_epochs}_epochs",
        f"{n_cats}_bins",
        loss_label,
    ]

    if bool(continue_cfg.get("enabled", False)):
        parts.append("continued")

    return "_".join(parts)


def build_descriptive_out_dir(training_config):
    """
    Replaces training_config["out-dir"]'s last path component with
    build_run_name(training_config)'s descriptive slug, keeping the same
    parent directory, then runs the result through get_unique_out_dir so
    two runs with identical settings (a genuine re-run, a hyperparameter
    re-sweep, ...) still land in distinct folders instead of colliding -
    e.g. "3D_010_500_epochs_5_bins_soft_class_loss" ->
    "..._soft_class_loss1" -> "..._soft_class_loss2" on repeats, exactly
    like get_unique_out_dir already does for the plain out-dir string.
    """
    base_out_dir = training_config["out-dir"].rstrip("/")
    parent_dir = os.path.dirname(base_out_dir)
    run_name = build_run_name(training_config)
    candidate = os.path.join(parent_dir, run_name) if parent_dir else run_name
    return get_unique_out_dir(candidate)


def write_run_summary(out_dir, training_config, args, run_info):
    """
    Writes a single, human-readable training_info.txt into out_dir with
    every parameter relevant to reproducing/understanding this run: the
    full effective config, CLI args, and a handful of runtime-computed
    values (resolved out_dir, best losses, learned mass window, ...) that
    aren't in the config itself.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("TRAINING RUN SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Timestamp:            {run_info.get('timestamp')}")
    lines.append(f"Resolved out-dir:     {out_dir}")
    lines.append(f"Configured out-dir:   {training_config.get('out-dir')}")
    lines.append(f"individual_samples base-dir: {args.base_dir}")
    lines.append(f"Data config path:     {(training_config.get('data_prep') or {}).get('data_config_path')}")
    lines.append(f"Training config path: {args.training_config_path}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("CLI ARGUMENTS")
    lines.append("-" * 70)
    for k, v in vars(args).items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("RUNTIME INFO")
    lines.append("-" * 70)
    for k, v in run_info.items():
        if k == "timestamp":
            continue
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("EFFECTIVE TRAINING CONFIG (training_config.yaml)")
    lines.append("-" * 70)
    lines.append(yaml.dump(training_config, sort_keys=False, default_flow_style=False))

    summary_path = os.path.join(out_dir, "training_info.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Saved training run summary: {summary_path}")
    return summary_path


def _rescale_hist_by_category(h, factors):
    """
    Returns a COPY of the 1-D, one-bin-per-GMM-category Hist `h` with each
    bin's value (and, for Weight() storage, variance) scaled by the
    corresponding entry of `factors` (length == number of bins/categories).

    Used to apply the per-category non-resonant-background sideband
    reweight factor to the "background_nonres" category histogram before
    plotting (see evaluate_on_test_set) - Hist*array is NOT a guaranteed
    boost_histogram/hist operation (only Hist*scalar, already used
    elsewhere in this file for `scale_signal`, is), so this goes through
    `.view()` instead - the same portable API compute_nonres_reweight_factors
    and hist_values already rely on. The result is then combined with the
    "background_other" histogram via ordinary Hist+Hist addition, which IS
    a standard operation.
    """
    h2 = h.copy()
    view = h2.view(flow=False)
    factors = np.asarray(factors, dtype=np.float64)
    if view.dtype.names and "value" in view.dtype.names:
        view["value"] = view["value"] * factors
        if "variance" in view.dtype.names:
            view["variance"] = view["variance"] * (factors ** 2)
    else:
        view[...] = view * factors
    return h2


def hist_values(h):
    """Extracts plain per-bin (weighted) yields from whatever
    fill_histogram_from_assignments returns (see evaluate_model.py)."""
    if hasattr(h, "values"):
        return np.asarray(h.values(), dtype=np.float64)
    arr = np.asarray(h)
    if arr.dtype.names and "value" in arr.dtype.names:
        return arr["value"].astype(np.float64)
    return arr.astype(np.float64)


def hist_variances(h):
    """Per-bin VARIANCE counterpart of hist_values() above - the MC
    statistical uncertainty input (sum of squared weights per bin) needed
    to put an error bar on a hard-binned Asimov Z. Only meaningful for
    Weight() storage (see _rescale_hist_by_category's docstring, which
    already relies on the same "value"/"variance" field convention); for
    a plain (unweighted-variance) histogram this returns zeros rather than
    raising, so a caller that always asks for it doesn't need its own
    storage-type branch - callers should treat an all-zero return as "no
    uncertainty information available" rather than "genuinely zero
    uncertainty".
    """
    if hasattr(h, "variances"):
        var = h.variances()
        if var is not None:
            return np.asarray(var, dtype=np.float64)
    arr = np.asarray(h)
    if arr.dtype.names and "variance" in arr.dtype.names:
        return arr["variance"].astype(np.float64)
    return np.zeros_like(hist_values(h))


def _asimov_z_with_uncertainty(S, S_w2, B, B_w2):
    """
    Per-category Asimov significance PLUS its MC-statistical uncertainty,
    propagated from the per-category signal/background sum-of-squared-
    weights (S_w2/B_w2) via ordinary error propagation:

        sigma_Z_k = sqrt( (dZ_k/dS_k * sigma_S_k)^2 + (dZ_k/dB_k * sigma_B_k)^2 )

    with sigma_S_k = sqrt(S_w2_k), sigma_B_k = sqrt(B_w2_k) - the same MC-
    statistical-uncertainty convention as gatohep.losses.
    high_bkg_uncertainty_penalty and training.py's B_w2 additions.

    The partial derivatives dZ_k/dS_k, dZ_k/dB_k are obtained from a single
    small tf.GradientTape call on `tf.reduce_sum(asymptotic_significance(S,
    B))` w.r.t. S and B - valid ONLY because asymptotic_significance is a
    purely elementwise function of S and B (z_k depends on S_k, B_k alone,
    never on S_j/B_j for j != k - see its definition in gatohep.utils), so
    d(sum_k z_k)/dS_j collapses to exactly dz_j/dS_j. This is the same
    "gradient-caching" style trick used throughout training.py, just
    applied here to a NumPy-side diagnostic instead of an actual training
    gradient - no optimizer ever sees this tape.

    The COMBINED Z = sqrt(sum_k z_k^2) is then propagated through that same
    formula: d(Z_combined)/dz_k = z_k / Z_combined, combined in quadrature
    over the already-computed per-category sigma_Z_k.

    Returns (z_per_cat, z_unc_per_cat, Z_combined, Z_combined_unc), all
    plain numpy floats/arrays - safe to use directly in a summary table
    with no further TF objects involved.
    """
    S_t = tf.constant(np.asarray(S, dtype=np.float32))
    B_t = tf.constant(np.asarray(B, dtype=np.float32))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(S_t)
        tape.watch(B_t)
        z_per_cat_t = asymptotic_significance(S_t, B_t)
        z_sum_t = tf.reduce_sum(z_per_cat_t)
    dZdS = tape.gradient(z_sum_t, S_t).numpy().astype(np.float64)
    dZdB = tape.gradient(z_sum_t, B_t).numpy().astype(np.float64)
    del tape

    sigma_S = np.sqrt(np.maximum(np.asarray(S_w2, dtype=np.float64), 0.0))
    sigma_B = np.sqrt(np.maximum(np.asarray(B_w2, dtype=np.float64), 0.0))
    z_per_cat = z_per_cat_t.numpy().astype(np.float64)
    z_unc_per_cat = np.sqrt((dZdS * sigma_S) ** 2 + (dZdB * sigma_B) ** 2)

    Z_combined = float(np.sqrt(np.sum(z_per_cat ** 2)))
    if Z_combined > 0.0:
        Z_combined_unc = float(np.sqrt(np.sum((z_per_cat / Z_combined * z_unc_per_cat) ** 2)))
    else:
        Z_combined_unc = 0.0

    return z_per_cat, z_unc_per_cat, Z_combined, Z_combined_unc


def write_split_variant_events_table(
    split_name, variant, S_soft, B_soft, z_soft, Z_soft,
    S_hard, S_hard_unc, B_hard, B_hard_unc, z_hard, z_hard_unc, Z_hard, Z_hard_unc,
    out_dir, plot_dtype="pdf",
):
    """
    Writes ONE CSV + ONE PDF under out_dir, for a single (split, variant)
    combination - e.g. (train, without_nonres_reweight) - breaking down
    per-category signal/background yields and Asimov significance, both
    "soft" (the differentiable GMM soft-assignment yields Stage B actually
    optimizes - see GMMFromScores.call) and "hard" (argmax-category yields,
    with their MC-statistical uncertainty - see _asimov_z_with_uncertainty).

    Six rows: S_soft, B_soft, Z_soft (no uncertainty - the soft assignment
    has no associated statistical-uncertainty concept in this framework,
    it is the plain differentiable quantity), then S_hard, B_hard, Z_hard
    (each "value +/- uncertainty" per category and combined).

    Files: events_table_{split_name}_{variant}.csv /
    .{plot_dtype} under out_dir.

    Grey-shaded PDF table (reportlab), matching make_event_count_table's
    styling: bold light-grey header row, plus alternating light shading on
    every other data row so a reader's eye can follow a row across many
    category columns without losing their place - the concrete ask behind
    "grey shading to make navigation inside the table easier".

    Returns (df, csv_path, pdf_path).
    """
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    n_cats = len(S_soft)
    cat_cols = [f"cat_{k}" for k in range(n_cats)]
    columns = ["quantity"] + cat_cols + ["combined"]

    def fmt(v):
        return f"{v:.4f}"

    def fmt_unc(v, u):
        return f"{v:.4f} +/- {u:.4f}"

    rows = [
        ["S_soft"] + [fmt(v) for v in S_soft] + [fmt(float(np.sum(S_soft)))],
        ["B_soft"] + [fmt(v) for v in B_soft] + [fmt(float(np.sum(B_soft)))],
        ["Z_soft"] + [fmt(v) for v in z_soft] + [fmt(Z_soft)],
        ["S_hard"] + [fmt_unc(v, u) for v, u in zip(S_hard, S_hard_unc)]
        + [fmt_unc(float(np.sum(S_hard)), float(np.sqrt(np.sum(np.asarray(S_hard_unc) ** 2))))],
        ["B_hard"] + [fmt_unc(v, u) for v, u in zip(B_hard, B_hard_unc)]
        + [fmt_unc(float(np.sum(B_hard)), float(np.sqrt(np.sum(np.asarray(B_hard_unc) ** 2))))],
        ["Z_hard"] + [fmt_unc(v, u) for v, u in zip(z_hard, z_hard_unc)] + [fmt_unc(Z_hard, Z_hard_unc)],
    ]

    df = pd.DataFrame(rows, columns=columns)

    csv_path = os.path.join(out_dir, f"events_table_{split_name}_{variant}.csv")
    df.to_csv(csv_path, index=False)

    pdf_path = os.path.join(out_dir, f"events_table_{split_name}_{variant}.{plot_dtype}")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=landscape(A4),
        rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20,
    )
    styles = getSampleStyleSheet()
    title = f"Event yields &amp; significance ({split_name}, {variant})"
    table_rows = [columns] + rows
    table = Table(table_rows, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        # First column (row labels) bold, to match the header emphasis.
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ]
    # Alternating grey shading on data rows (header already shaded above) -
    # the "grey shading to make navigation inside the table easier" ask.
    for i in range(1, len(table_rows)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.whitesmoke))
    table.setStyle(TableStyle(style_cmds))
    doc.build([Paragraph(title, styles["Title"]), Spacer(1, 12), table])

    return df, csv_path, pdf_path


# --------------------------------------------------------------------------
# Test-set evaluation (shared by end_to_end_training.py's automatic
# post-training call and the standalone evaluate_model.py script)
# --------------------------------------------------------------------------
def build_static_preselection_cuts(presel_cfg, arrays):
    """
    Builds the ordered list of (name, full_length_boolean_mask) pairs for
    every STATIC (no model needed) preselection cut that is both enabled in
    presel_cfg and has its column available in `arrays`. Order matches
    gato_categorization_vbf_only_ggh.py's own preselection() function:
    mass window, dijet-mass window, lead/sublead mvaID thresholds, then the
    is_boosted veto - the score-based cut (needs a classifier) is handled
    separately by build_score_preselection_cut, since it can only run once
    a classifier exists (see pretrain.py vs combined_optimization.py).

    presel_cfg: training_config["preselection"]. Each non-mass cut has its
    own nested "enabled" switch (dijet_mass.enabled, lead_mvaid.enabled,
    sublead_mvaid.enabled, is_boosted_veto.enabled) so any of them can be
    turned off independently of the master preselection.enabled flag - see
    pretrain_config.yaml/combined_optimization_config.yaml.

    arrays: {"mass": ..., "dijet_mass": ..., "lead_mvaid": ...,
    "sublead_mvaid": ..., "is_boosted": ...} - any subset; a cut whose
    column isn't present here is silently skipped even if enabled (so
    callers only need to load the columns their own config actually asks
    for - see the extra_columns plumbing in dataset_prep.py).

    Returns a list of (name, mask) - feed straight into
    compute_preselection_cutflow.
    """
    presel_cfg = presel_cfg or {}
    cuts = []
    if not bool(presel_cfg.get("enabled", False)):
        return cuts

    if "mass" in arrays and arrays["mass"] is not None:
        mass = np.asarray(arrays["mass"])
        low, high = presel_cfg["mass_low"], presel_cfg["mass_high"]
        cuts.append(("mass_window", (mass >= low) & (mass <= high)))

    dijet_cfg = presel_cfg.get("dijet_mass", {}) or {}
    if bool(dijet_cfg.get("enabled", False)) and "dijet_mass" in arrays and arrays["dijet_mass"] is not None:
        dijet_mass = np.asarray(arrays["dijet_mass"])
        low, high = dijet_cfg["low"], dijet_cfg["high"]
        cuts.append(("dijet_mass_window", (dijet_mass > low) & (dijet_mass < high)))

    lead_cfg = presel_cfg.get("lead_mvaid", {}) or {}
    if bool(lead_cfg.get("enabled", False)) and "lead_mvaid" in arrays and arrays["lead_mvaid"] is not None:
        lead_mvaid = np.asarray(arrays["lead_mvaid"])
        cuts.append(("lead_mvaid_min", lead_mvaid > lead_cfg["min"]))

    sublead_cfg = presel_cfg.get("sublead_mvaid", {}) or {}
    if bool(sublead_cfg.get("enabled", False)) and "sublead_mvaid" in arrays and arrays["sublead_mvaid"] is not None:
        sublead_mvaid = np.asarray(arrays["sublead_mvaid"])
        cuts.append(("sublead_mvaid_min", sublead_mvaid > sublead_cfg["min"]))

    boosted_cfg = presel_cfg.get("is_boosted_veto", {}) or {}
    if bool(boosted_cfg.get("enabled", False)) and "is_boosted" in arrays and arrays["is_boosted"] is not None:
        is_boosted = np.asarray(arrays["is_boosted"]).astype(bool)
        cuts.append(("is_boosted_veto", ~is_boosted))

    return cuts


def static_preselection_columns(presel_cfg):
    """
    Which extra events_boostedCat.parquet columns build_static_preselection_
    cuts will actually need, given presel_cfg - so callers can pass exactly
    this to dataset_prep's extra_columns instead of guessing/hardcoding it.
    Returns a tuple of parquet column names (empty if preselection or every
    individual sub-cut is disabled).
    """
    presel_cfg = presel_cfg or {}
    if not bool(presel_cfg.get("enabled", False)):
        return ()
    cols = []
    if bool((presel_cfg.get("dijet_mass", {}) or {}).get("enabled", False)):
        cols.append("nonResReg_vbfpair_dijet_mass_DNNreg")
    if bool((presel_cfg.get("lead_mvaid", {}) or {}).get("enabled", False)):
        cols.append("lead_mvaID")
    if bool((presel_cfg.get("sublead_mvaid", {}) or {}).get("enabled", False)):
        cols.append("sublead_mvaID")
    if bool((presel_cfg.get("is_boosted_veto", {}) or {}).get("enabled", False)):
        cols.append("is_boosted")
    return tuple(cols)


def build_score_preselection_cut(presel_cfg, score_column):
    """
    (name, mask) for the classifier-score-based cut (gato_categorization_
    vbf_only_ggh.py's `score_bool = scores[:, 3] < 0.774`), or None if
    presel_cfg["score_cut"]["enabled"] is not true. Needs an already-
    computed 1-D array of one softmax class's score per (surviving) event -
    see compute_class_scores below for how that's produced.

    presel_cfg["score_cut"]: {"enabled", "class_index", "min", "max"} -
    "min"/"max" are independently optional (null/omit = no bound on that
    side), so this can reproduce the reference's one-sided "< 0.774" cut
    (max: 0.774, min omitted) or a two-sided window.
    """
    cfg = (presel_cfg or {}).get("score_cut", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return None
    score_column = np.asarray(score_column)
    mask = np.ones(score_column.shape[0], dtype=bool)
    if cfg.get("min") is not None:
        mask &= (score_column >= float(cfg["min"]))
    if cfg.get("max") is not None:
        mask &= (score_column < float(cfg["max"]))
    return (f"classifier_score_class{cfg.get('class_index', 0)}", mask)


def compute_class_scores(classifier, X, class_index, chunk_size=None):
    """
    Chunked classifier(X) -> softmax -> one class's score column, for the
    score-based preselection cut above. Uses the raw classifier alone (its
    own softmax), NOT EndToEndClassifierGMM.scores() - this runs before the
    GMM categorizer even exists in combined_optimization.py (the cut is
    meant to shape what Stage B ever trains/fine-tunes on), and mirrors the
    reference script's use of a plain, already-trained classifier's
    NN_output directly.
    """
    n = X.shape[0]
    step = chunk_size or n
    out = []
    for a in range(0, n, step):
        b = min(a + step, n)
        X_chunk = np.asarray(X[a:b], dtype=np.float32)
        logits = classifier(X_chunk, training=False)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        out.append(probs[:, class_index])
    return np.concatenate(out, axis=0) if out else np.zeros(0, dtype=np.float32)


def compute_preselection_cutflow(cuts, n_rows, split_name="train", verbose=True):
    """
    Sequentially intersects an ordered list of (name, full_length_mask)
    pairs (see build_static_preselection_cuts/build_score_preselection_cut)
    into one final set of surviving row indices, tracking how many events
    each individual cut removes given everything before it has already
    been applied (a standard HEP "cutflow" - order-dependent, unlike a
    single combined AND with no accounting) - this is the "counter" behind
    the preselection_cutflow.csv each script writes.

    Returns (keep_idx, cutflow_rows) - keep_idx indexes into the ORIGINAL
    (pre-cutflow) n_rows-length arrays directly, so it composes correctly
    when this is called again later for a further cut (e.g. the score cut,
    run after keep_idx from the static cuts has already been applied - see
    combined_optimization.py). cutflow_rows is a list of dicts with keys
    split/cut/n_before/n_removed/n_after/frac_kept, ready to append to a
    CSV (see write_preselection_cutflow_csv).
    """
    keep_idx = np.arange(n_rows)
    cutflow_rows = []
    for name, full_mask in cuts:
        full_mask = np.asarray(full_mask)
        n_before = keep_idx.size
        keep_idx = keep_idx[full_mask[keep_idx]]
        n_after = keep_idx.size
        frac = (n_after / n_before) if n_before else 1.0
        cutflow_rows.append({
            "split": split_name, "cut": name,
            "n_before": int(n_before), "n_removed": int(n_before - n_after),
            "n_after": int(n_after), "frac_kept": frac,
        })
        if verbose:
            print(
                f"[PRESELECTION][{split_name}] {name}: kept {n_after:,}/{n_before:,} "
                f"({100.0 * frac:.1f}%), removed {n_before - n_after:,}"
            )
    return keep_idx, cutflow_rows


def write_preselection_cutflow_csv(path, cutflow_rows, append=True):
    """
    Appends (default) or overwrites cutflow_rows (see
    compute_preselection_cutflow) to a CSV at `path` - one growing table
    across every split/stage a script calls this with (pretrain.py writes
    its train/val rows first; combined_optimization.py's own run appends
    its train/val/test rows, including the separate score-cut rows, to the
    same file if pointed at the same out_dir), so the full cutflow for a
    run lives in one place.
    """
    import csv as _csv
    if not cutflow_rows:
        return
    file_exists = append and os.path.exists(path)
    mode = "a" if append else "w"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["split", "cut", "n_before", "n_removed", "n_after", "frac_kept"])
        if not file_exists:
            writer.writeheader()
        for row in cutflow_rows:
            writer.writerow(row)


def apply_mass_preselection(mass_arr, presel_cfg, arrays, split_name="test", verbose=True):
    """
    Applies the same hard, non-differentiable preselection mass cut used on
    the train/val splits in end_to_end_training.py's PRESELECTION block, to
    an arbitrary split's row-aligned arrays.

    mass_arr: 1-D mass array for this split.
    presel_cfg: training_config["preselection"] (or {} / None - a no-op if
    "enabled" is not true).
    arrays: dict of {name: array}, every array row-aligned with mass_arr
    (e.g. {"X": X_test, "y": y_test, "rel_w": rel_w_test, "cw": cw_test}).

    Returns (mass_arr_cut, {name: array_cut}) - inputs are returned
    unchanged (not copied) if preselection is disabled.

    Uses plain fancy indexing (not the zero-copy PermutedView machinery
    used for train/val in end_to_end_training.py) since test-set evaluation
    is a one-off pass, not a per-epoch operation - the extra copy is
    negligible here.

    NOTE: this only ever applies the plain mass window - it predates the
    dijet-mass/mvaID/is_boosted/score cuts (see build_static_preselection_
    cuts/build_score_preselection_cut + compute_preselection_cutflow for
    those) and is kept around only because evaluate_model_update.py /
    evaluate_model_detailed_contours.py still call it for their own
    from-scratch data rebuild.
    """
    presel_cfg = presel_cfg or {}
    if not bool(presel_cfg.get("enabled", False)):
        return mass_arr, arrays

    low, high = presel_cfg["mass_low"], presel_cfg["mass_high"]
    mass_arr = np.asarray(mass_arr)
    n_raw = mass_arr.shape[0]
    for name, arr in arrays.items():
        if arr is not None and np.asarray(arr).shape[0] != n_raw:
            raise ValueError(
                f"Row-count mismatch applying preselection to '{split_name}': "
                f"mass has {n_raw} rows but '{name}' has {np.asarray(arr).shape[0]}."
            )

    keep = np.flatnonzero((mass_arr >= low) & (mass_arr <= high))
    if keep.size == 0:
        raise ValueError(
            f"preselection [{low}, {high}] removes ALL '{split_name}' events."
        )
    if verbose:
        print(
            f"[INFO] Preselection ({split_name}): hard cut mass in [{low}, {high}] "
            f"kept {keep.size:,}/{n_raw:,} ({100.0 * keep.size / max(n_raw, 1):.1f}%) events."
        )

    cut_arrays = {name: (np.asarray(arr)[keep] if arr is not None else None) for name, arr in arrays.items()}
    return mass_arr[keep], cut_arrays


def evaluate_split_significance(
    classifier, gato, signal_class, plot_dim, X, y, w, mass, training_config, classes, out_dir,
    split_name, nonres_reweight_fn=compute_nonres_reweight_factors,
    chunk_size=None, plot_dtype="pdf", verbose=True,
):
    """
    Split-agnostic sibling of evaluate_on_test_set, focused ONLY on
    per-category Z_soft/Z_hard (+uncertainty) and the events tables built
    from them - deliberately NOT reproducing evaluate_on_test_set's JSON/
    optimized-histograms-plot/bias-diagnostic output (those stay TEST-only,
    unchanged, in that function) so this can run identically on train/val/
    test without touching that already-exercised code path.

    X/y/w/mass are assumed to already have the broad PRESELECTION cut
    applied (same assumption as evaluate_on_test_set) - `w` should be the
    split's own rel_w (raw_data["rel_w_<split>"]), not a class-weight
    array.

    Reuses the exact same building blocks as evaluate_on_test_set: fits
    non-resonant reweight factors on this split's own sideband (fully
    independent of the factors used at the end of Stage B or in any other
    split's evaluation - see compute_nonres_reweight_factors), applies the
    same variant-dependent hard mgg cut (the "with_nonres_reweight" variant
    exempts the non-resonant class from the cut - see
    evaluate_on_test_set's docstring for the full rationale, identical
    here), then for each variant:

      - Z_soft: from the SAME soft (GMM-membership-weighted) S/B this
        split's Stage-B monitoring would use - obtained via the chunked
        additive-yields accumulation (model.chunk_yields, see its
        docstring), i.e. numerically the differentiable quantity Stage B
        actually optimizes. No uncertainty is reported for this - the soft
        assignment has no associated statistical-uncertainty concept in
        this framework.
      - Z_hard: from the argmax-category yields (same
        assign_bins_and_order/fill_histogram_from_assignments path
        evaluate_on_test_set's "optimized histograms" plot uses), WITH a
        per-category MC-statistical uncertainty from hist_variances via
        _asimov_z_with_uncertainty.

    Writes events_table_{split_name}_{variant}.csv/.{plot_dtype} for each
    variant that actually ran (see write_split_variant_events_table).

    Returns {variant: {"Z_soft_per_cat", "Z_soft", "Z_hard_per_cat",
    "Z_hard_per_cat_unc", "Z_hard", "Z_hard_unc", "S_soft", "B_soft",
    "S_hard", "S_hard_unc", "B_hard", "B_hard_unc"}} - everything
    evaluate_all_splits needs to build the master summary table, with no
    file I/O of its own beyond the events tables above.
    """
    drop_last_class = bool(training_config.get("gato_model", {}).get("drop_last_softmax_class", False))
    n_cats = gato.n_cats

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float32)
    mass = np.asarray(mass, dtype=np.float32)
    n_preselected = int(X.shape[0])

    nonres_full_cfg = training_config.get("nonres_reweight", {}) or {}
    nonres_class = (
        int(nonres_full_cfg["nonres_bkg_class"]) if "nonres_bkg_class" in nonres_full_cfg else None
    )

    # ---- non-resonant reweight factors: fit on THIS split's own wide,
    # preselected-only mass range - independent of whatever factors were
    # last used in training or in any other split's evaluation. See
    # evaluate_on_test_set's docstring for why this must happen BEFORE the
    # hard mgg cut. ----
    factors = None
    if nonres_full_cfg and nonres_class is not None:
        fit_model = EndToEndClassifierGMM(
            classifier, gato, signal_class=signal_class,
            mass_window_cfg=None, nonres_reweight_cfg=None,
            drop_last_class=drop_last_class,
        )
        factors = nonres_reweight_fn(
            fit_model, X, y, w, mass,
            nonres_class=nonres_class, n_cats=n_cats,
            mass_sb_low=nonres_full_cfg.get("mass_sb_low", 100.0),
            mass_sb_high=nonres_full_cfg.get("mass_sb_high", 180.0),
            mass_sig_low=nonres_full_cfg.get("mass_sig_low", 120.0),
            mass_sig_high=nonres_full_cfg.get("mass_sig_high", 130.0),
            nbins=resolve_nonres_nbins(nonres_full_cfg, split_name, training_config.get("data_prep")),
            default_factor=float(nonres_full_cfg.get("default_factor", 0.062)),
            chunk_size=chunk_size,
            diagnostics_csv_path=os.path.join(out_dir, f"nonres_fit_diagnostics_{split_name}.csv"),
            save_fit_plots=True,
            plot_dir=os.path.join(out_dir, "nonres_fit_diagnostics"),
            plot_dtype=plot_dtype,
            call_label=f"evaluate_split_significance_{split_name}",
        )
        if verbose:
            print(f"[evaluate][{split_name}] Non-resonant reweight factors (own sideband): {factors}")
    elif verbose:
        print(
            f"[evaluate][{split_name}] No usable nonres_reweight section - only "
            "'without_nonres_reweight' will be run."
        )

    # ---- hard mgg cut - see evaluate_on_test_set's docstring for the full
    # without/with_nonres_reweight row-selection rationale (identical here,
    # just parameterized by split_name instead of hardcoded "test"). ----
    eval_cfg = training_config.get("evaluation", {}) or {}
    mw_low, mw_high = eval_cfg.get("mass_window", [None, None])
    if mw_low is None or mw_high is None:
        raise ValueError("training_config['evaluation']['mass_window'] = [low, high] must be set.")
    in_window = (mass >= mw_low) & (mass <= mw_high)
    keep_without = np.flatnonzero(in_window)
    if keep_without.size == 0:
        raise ValueError(f"Hard mgg cut [{mw_low}, {mw_high}] removes ALL preselected {split_name} events.")
    keep_with = (
        np.flatnonzero(in_window | (y == nonres_class)) if nonres_class is not None else keep_without
    )
    keep_by_variant = {"without_nonres_reweight": keep_without, "with_nonres_reweight": keep_with}
    if verbose:
        print(
            f"[evaluate][{split_name}] Hard mgg cut [{mw_low}, {mw_high}] kept "
            f"{keep_without.size:,}/{n_preselected:,} events for 'without_nonres_reweight'"
            + (
                f"; {keep_with.size:,}/{n_preselected:,} for 'with_nonres_reweight'"
                if nonres_class is not None else ""
            ) + "."
        )

    results = {}
    for variant, use_nonres in (("without_nonres_reweight", False), ("with_nonres_reweight", True)):
        if use_nonres and factors is None:
            if verbose:
                print(f"[evaluate][{split_name}][{variant}] SKIPPED - no nonres reweight factors.")
            continue

        keep = keep_by_variant[variant]
        X_final, y_final, w_final = X[keep], y[keep], w[keep]
        rewt_tf = tf.constant(factors, dtype=tf.float32) if use_nonres else None

        model = EndToEndClassifierGMM(
            classifier, gato, signal_class=signal_class,
            mass_window_cfg=None,
            nonres_reweight_cfg=(nonres_full_cfg if use_nonres else None),
            drop_last_class=drop_last_class,
        )

        # ---- Z_soft: same additive chunked S/B (+B_w2, unused here)
        # accumulation the Stage-B training/monitoring code path uses (see
        # model.chunk_yields' docstring for the additivity argument) - the
        # differentiable quantity actually optimized. ----
        n_final = int(X_final.shape[0])
        cs = chunk_size if chunk_size else n_final
        S_soft = np.zeros(n_cats, dtype=np.float64)
        B_soft = np.zeros(n_cats, dtype=np.float64)
        for a in range(0, n_final, cs):
            b = min(a + cs, n_final)
            Sc, Bc, _B_w2c, _logits_c = model.chunk_yields(
                X_final[a:b], y_final[a:b], w_final[a:b], mass=None, training=False, rewt=rewt_tf
            )
            S_soft += Sc.numpy().astype(np.float64)
            B_soft += Bc.numpy().astype(np.float64)
        z_soft_per_cat = asymptotic_significance(
            tf.constant(S_soft, dtype=tf.float32), tf.constant(B_soft, dtype=tf.float32)
        ).numpy().astype(np.float64)
        Z_soft = float(np.sqrt(np.sum(z_soft_per_cat ** 2)))

        # ---- Z_hard: argmax-category yields, same path as
        # evaluate_on_test_set's "optimized histograms" plot, plus MC-
        # statistical uncertainty via hist_variances (only available
        # because fill_histogram_from_assignments uses Weight() storage -
        # see _rescale_hist_by_category's docstring). ----
        plot_data = make_plot_dataframe_from_model(
            model, X_final, y_final, w_final, signal_class, plot_dim, chunk_size=chunk_size,
            nonres_class=(nonres_class if use_nonres else None),
        )
        assign, order, _, inv = assign_bins_and_order(gato, plot_data, reduce=True)
        filled = {
            proc: fill_histogram_from_assignments(assign[proc], plot_data[proc]["weight"], n_cats)
            for proc in plot_data
        }
        if use_nonres:
            background_hist = (
                _rescale_hist_by_category(filled["background_nonres"], factors)
                + filled["background_other"]
            )
        else:
            background_hist = filled["background"]

        S_hard = hist_values(filled["signal"])
        S_hard_w2 = hist_variances(filled["signal"])
        B_hard = hist_values(background_hist)
        B_hard_w2 = hist_variances(background_hist)

        z_hard_per_cat, z_hard_unc_per_cat, Z_hard, Z_hard_unc = _asimov_z_with_uncertainty(
            S_hard, S_hard_w2, B_hard, B_hard_w2
        )
        S_hard_unc = np.sqrt(np.maximum(S_hard_w2, 0.0))
        B_hard_unc = np.sqrt(np.maximum(B_hard_w2, 0.0))

        if verbose:
            print(
                f"[evaluate][{split_name}][{variant}] Z_soft={Z_soft:.4f} "
                f"Z_hard={Z_hard:.4f} +/- {Z_hard_unc:.4f}"
            )

        write_split_variant_events_table(
            split_name, variant, S_soft, B_soft, z_soft_per_cat, Z_soft,
            S_hard, S_hard_unc, B_hard, B_hard_unc, z_hard_per_cat, z_hard_unc_per_cat, Z_hard, Z_hard_unc,
            out_dir, plot_dtype=plot_dtype,
        )

        results.setdefault(variant, {})
        results[variant].update({
            "S_soft": S_soft, "B_soft": B_soft,
            "z_soft_per_cat": z_soft_per_cat, "Z_soft": Z_soft,
            "S_hard": S_hard, "S_hard_unc": S_hard_unc,
            "B_hard": B_hard, "B_hard_unc": B_hard_unc,
            "z_hard_per_cat": z_hard_per_cat, "z_hard_per_cat_unc": z_hard_unc_per_cat,
            "Z_hard": Z_hard, "Z_hard_unc": Z_hard_unc,
        })

    return results


def evaluate_on_test_set(
    classifier, gato, signal_class, plot_dim,
    X_test, y_test, w_test, mass_test, cw_test,
    training_config, classes, out_dir,
    nonres_reweight_fn=compute_nonres_reweight_factors,
    chunk_size=None,
    plot_dtype="png",
    verbose=True,
):
    """
    Evaluates a trained classifier+GMM on the held-out TEST split only:
    significance (Z), penalty, cross-entropy and mean bias (the same four
    quantities monitored on the val split during Stage B - see
    training.evaluate_complete_model_chunked / the diagnostics block in
    train_complete_model), plus the hard-bin "optimized histograms" plot
    (the same style produced at the end of end_to_end_training.py, but
    built from X_test/y_test/w_test only).

    X_test/y_test/w_test/mass_test/cw_test are assumed to already have the
    broad PRESELECTION cut applied (see end_to_end_training.py's/
    evaluate_model.py's apply_mass_preselection call on the test split) -
    "before passing the test data into the classifier".

    Unlike Stage B (which applies a *differentiable* soft mgg window,
    EndToEndClassifierGMM's sigmoid-taper mass_window_weighting, so its
    edges stay optimizable), this evaluation applies a genuine HARD mgg cut
    instead - training_config["evaluation"]["mass_window"] = [low, high] -
    since nothing here needs a gradient (this is a forward-only pass). The
    model built inside this function therefore always uses
    mass_window_cfg=None; the cut is a row-selection done directly on
    X_test/y_test/w_test/cw_test, not a per-event weight factor.

    Ordering: the non-resonant reweight factors (if
    training_config["nonres_reweight"] is present) are fit BEFORE the hard
    mgg cut, from the full preselected mass range - the sideband fit needs
    that wider range (mass_sb_low..mass_sb_high, typically 100-180) to have
    anything to fit; cutting to the narrow signal window first would leave
    no sideband. Z/penalty/cross-entropy/bias/histograms are all then
    computed AFTER the hard mgg cut, on the signal-region-only subset.

    IMPORTANT: the hard cut is NOT the same row-selection for both variants
    (see keep_by_variant below). For "with_nonres_reweight", the
    non-resonant background class is exempted from the cut - it is
    represented by its FULL (uncut) preselected-range yield, multiplied by
    the fitted per-category `factors` instead, exactly mirroring
    EndToEndClassifierGMM.mass_window_factor's training-time exemption (see
    that docstring): the sideband fit already predicts the in-window yield
    from the full range, so also cutting those events to the window before
    applying `factors` would double-suppress the non-resonant background
    and inflate Z. Every other class is still cut to the window in both
    variants - only the non-resonant class's row-selection differs.

    Runs the post-cut evaluation TWICE - "without_nonres_reweight"
    (background is one plain bucket, no reweighting, ordinary hard cut on
    every class) and "with_nonres_reweight" (background is split into
    background_nonres/background_other, the non-resonant class keeps its
    full uncut yield as described above, and the nonres reweight factors
    fit above are applied to the background_nonres bucket only) - mirrors
    compute_nonres_reweight_factors' use in training, but self-contained on
    the test set alone, independent of whatever factors were last used at
    the end of Stage B. classifier and gato are shared between both passes
    (only the wrapping EndToEndClassifierGMM's nonres_reweight_cfg and the
    row-selection differ), so this never touches the trained weights.

    training_config["nonres_reweight"] (its sideband/window/nbins/
    default_factor fields, NOT its "enabled" flag - the with/without
    comparison is produced regardless of whether nonres reweighting was
    actually enabled during training) must be present for the
    "with_nonres_reweight" pass; it is skipped (with a message) otherwise.

    Writes, under out_dir/<evaluation.output_subdir>/:
        test_evaluation.json                                (both variants)
        test_evaluation.csv                                  (both variants)
        optimized_histograms_test_<variant>.<plot_dtype>      (per variant)
        nonres_fit_diagnostics_test.csv                       (if the
            with-reweight sideband fit ran at all - see below - this is
            ALWAYS written, not config-gated: "the fit that happens on the
            test dataset" should always leave a record)
        nonres_fit_diagnostics/nonres_sideband_fit_cat<k>_evaluate_on_test_set.<plot_dtype>
            (one per GMM category, ALWAYS written alongside the CSV, same
            reasoning)

    Returns the same dict that's written to the JSON file.
    """
    eval_cfg = training_config.get("evaluation", {}) or {}
    output_subdir = eval_cfg.get("output_subdir", "evaluation")
    eval_out_dir = os.path.join(out_dir, output_subdir)
    os.makedirs(eval_out_dir, exist_ok=True)

    # Must match whatever the trained model actually used (see
    # EndToEndClassifierGMM's drop_last_class docstring) - read from the
    # same training_config["gato_model"] key end_to_end_training.py checks
    # at startup, so a mismatched config can't silently score with the
    # wrong number of GMM input dimensions here.
    drop_last_class = bool(training_config.get("gato_model", {}).get("drop_last_softmax_class", False))

    train_cfg = training_config["training"]
    reg_cfg = training_config["regularization"]
    lam_penalty = reg_cfg["low_bkg_penalty"]["lam"]
    penalty_thr = reg_cfg["low_bkg_penalty"]["threshold"]
    # Same "enabled forces lam to 0.0" convention as train_complete_model's
    # read of this same config block - see its comment for why.
    unc_cfg = reg_cfg.get("high_bkg_uncertainty_penalty", {}) or {}
    lam_unc = float(unc_cfg.get("lam", 0.0)) if unc_cfg.get("enabled", False) else 0.0
    bkg_unc_threshold = float(unc_cfg.get("rel_threshold", 0.2))
    ce_mode = train_cfg.get("ce_mode", "binary_plain")
    n_classes = len(classes)
    ce_loss_fn = make_ce_loss_fn(ce_mode, signal_class, n_classes)

    chunk_size = chunk_size if chunk_size else eval_cfg.get("chunk_size", None)

    n_cats = gato.n_cats
    nonres_full_cfg = training_config.get("nonres_reweight", {}) or {}
    nonres_class = (
        int(nonres_full_cfg["nonres_bkg_class"]) if "nonres_bkg_class" in nonres_full_cfg else None
    )

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)
    w_test = np.asarray(w_test, dtype=np.float32)
    mass_test = np.asarray(mass_test, dtype=np.float32) if mass_test is not None else None
    cw_test = np.asarray(cw_test) if cw_test is not None else None
    if mass_test is None:
        raise ValueError(
            "evaluate_on_test_set requires mass_test - needed both for the nonres-reweight "
            "sideband fit and for the hard mgg cut below."
        )
    n_test_preselected = int(X_test.shape[0])

    # ---- non-resonant reweight factors: fit on the WIDE, preselected-only
    # mass range, i.e. BEFORE the hard mgg cut below narrows everything to
    # the signal window (see docstring). Uses a plain model
    # (mass_window_cfg=None, nonres_reweight_cfg=None) since
    # compute_nonres_reweight_factors only ever calls model.scores() /
    # model.gato.get_bin_indices() - never build_data_dict - so no bucket
    # splitting is needed for this fit. ----
    factors = None
    if nonres_full_cfg and nonres_class is not None:
        fit_model = EndToEndClassifierGMM(
            classifier, gato, signal_class=signal_class,
            mass_window_cfg=None, nonres_reweight_cfg=None,
            drop_last_class=drop_last_class,
        )
        factors = nonres_reweight_fn(
            fit_model, X_test, y_test, w_test, mass_test,
            nonres_class=nonres_class, n_cats=n_cats,
            mass_sb_low=nonres_full_cfg.get("mass_sb_low", 100.0),
            mass_sb_high=nonres_full_cfg.get("mass_sb_high", 180.0),
            mass_sig_low=nonres_full_cfg.get("mass_sig_low", 120.0),
            mass_sig_high=nonres_full_cfg.get("mass_sig_high", 130.0),
            nbins=resolve_nonres_nbins(nonres_full_cfg, "test", training_config.get("data_prep")),
            default_factor=float(nonres_full_cfg.get("default_factor", 0.062)),
            chunk_size=chunk_size,
            # Always saved, not config-gated - this is THE test-set fit
            # (as opposed to the periodic in-training refit on the train
            # split, see train_complete_model's out_dir param, or the
            # separate refit build_event_count_tables does for its own
            # with-reweight table) that training_config["evaluation"]'s
            # numbers (Z/S/B with/without reweight) are actually based on,
            # so unlike the other two call sites this one is never silent.
            diagnostics_csv_path=os.path.join(eval_out_dir, "nonres_fit_diagnostics_test.csv"),
            save_fit_plots=True,
            plot_dir=os.path.join(eval_out_dir, "nonres_fit_diagnostics"),
            plot_dtype=plot_dtype,
            call_label="evaluate_on_test_set",
        )
        if verbose:
            print(f"[evaluate] Non-resonant reweight factors (fit on TEST sideband, preselected range): {factors}")
    elif verbose:
        print(
            "[evaluate] No usable nonres_reweight section in training_config (missing "
            "or no nonres_bkg_class) - only the 'without_nonres_reweight' variant will be run."
        )

    # ---- hard mgg cut: replaces the differentiable soft window used during
    # training - see docstring. Applied to the PRESELECTED test arrays.
    #
    # TWO different row-selections are used below, one per variant:
    #   - "without_nonres_reweight": the ordinary cut, applied to EVERY
    #     class (keep_without) - unchanged from before.
    #   - "with_nonres_reweight": the non-resonant background class is
    #     instead represented by its FULL (uncut) preselected-range yield,
    #     multiplied by the per-category `factors` fit above (keep_with) -
    #     that is exactly what the sideband fit is FOR: predicting the
    #     in-window yield from a smooth exponential fit to the full range,
    #     rather than trusting the (often low-statistics) raw MC events
    #     that happen to land directly in the narrow window. Cutting those
    #     same non-resonant events to the window AND THEN also multiplying
    #     by `factors` would double-suppress the non-resonant background -
    #     once by the cut, once by the fit - artificially shrinking B and
    #     inflating Z, exactly the training-time bug described in
    #     EndToEndClassifierGMM.mass_window_factor's docstring, just via a
    #     hard cut here instead of the soft window. Every other class
    #     (signal, other backgrounds) is still cut to the window normally
    #     in both selections. ----
    mw_low, mw_high = eval_cfg.get("mass_window", [None, None])
    if mw_low is None or mw_high is None:
        raise ValueError("training_config['evaluation']['mass_window'] = [low, high] must be set.")
    in_window = (mass_test >= mw_low) & (mass_test <= mw_high)
    keep_without = np.flatnonzero(in_window)
    if keep_without.size == 0:
        raise ValueError(f"Hard mgg cut [{mw_low}, {mw_high}] removes ALL preselected test events.")
    keep_with = (
        np.flatnonzero(in_window | (y_test == nonres_class))
        if nonres_class is not None else keep_without
    )
    keep_by_variant = {"without_nonres_reweight": keep_without, "with_nonres_reweight": keep_with}
    if verbose:
        print(
            f"[evaluate] Hard mgg cut [{mw_low}, {mw_high}] kept "
            f"{keep_without.size:,}/{n_test_preselected:,} "
            f"({100.0 * keep_without.size / max(n_test_preselected, 1):.1f}%) preselected test "
            f"events for 'without_nonres_reweight'"
            + (
                f"; {keep_with.size:,}/{n_test_preselected:,} "
                f"({100.0 * keep_with.size / max(n_test_preselected, 1):.1f}%) for "
                f"'with_nonres_reweight' (non-resonant class exempted from the cut - see above)"
                if nonres_class is not None else ""
            )
            + "."
        )

    results = {}
    for variant, use_nonres in (("without_nonres_reweight", False), ("with_nonres_reweight", True)):
        if use_nonres and factors is None:
            print(f"[evaluate][{variant}] SKIPPED - no nonres reweight factors were computed (see above).")
            continue

        keep = keep_by_variant[variant]
        X_final = X_test[keep]
        y_final = y_test[keep]
        w_final = w_test[keep]
        cw_final = cw_test[keep] if cw_test is not None else None
        n_test_final = int(X_final.shape[0])
        dd_chunk_size = chunk_size if chunk_size else n_test_final  # build_data_dict_chunked needs a concrete size

        # mass_window_cfg=None here too: the hard mgg cut has ALREADY been
        # applied to X_final/y_final/w_final/cw_final above (variant-
        # dependent - see keep_by_variant), so there is nothing left for a
        # soft window to do.
        model = EndToEndClassifierGMM(
            classifier, gato, signal_class=signal_class,
            mass_window_cfg=None,
            nonres_reweight_cfg=(nonres_full_cfg if use_nonres else None),
            drop_last_class=drop_last_class,
        )
        rewt_tf = tf.constant(factors, dtype=tf.float32) if use_nonres else None

        # ---- scalar metrics: Z, penalty, cross-entropy (same code path used
        # for the Stage-B val-split monitoring every epoch); mass=None since
        # the model's soft window is disabled and the hard cut is already
        # baked into X_final/y_final/w_final. ----
        sig_loss, penalty, scalar_loss, Z, ce_mean, unc_penalty = evaluate_complete_model_chunked(
            model, X_final, y_final, w_final, None,
            chunk_size, lam_penalty, penalty_thr,
            ce_loss_fn=ce_loss_fn, cw=cw_final, rewt=rewt_tf,
            lam_unc=lam_unc, bkg_unc_threshold=bkg_unc_threshold,
        )

        # ---- bias: same diagnostic path used every `diagnostics_every`
        # epochs in train_complete_model (build_data_dict + gato.get_bias) ----
        data_dict = build_data_dict_chunked(
            model, X_final, y_final, w_final, dd_chunk_size, mass_all=None
        )
        bias_vec = model.gato.get_bias(data_dict)
        mean_bias = float(np.mean(np.abs(bias_vec)))

        if verbose:
            print(
                f"[evaluate][{variant}] Z={Z:.4f} sig_loss={sig_loss:.4f} "
                f"penalty={penalty:.4f} ce={ce_mean:.4f} mean_bias={mean_bias:.4f}"
            )

        # ---- hard-bin "optimized histograms", from the TEST set only ----
        plot_data = make_plot_dataframe_from_model(
            model, X_final, y_final, w_final, signal_class, plot_dim, chunk_size=chunk_size,
            nonres_class=(nonres_class if use_nonres else None),
        )
        assign, order, _, inv = assign_bins_and_order(gato, plot_data, reduce=True)
        filled = {
            proc: fill_histogram_from_assignments(assign[proc], plot_data[proc]["weight"], n_cats)
            for proc in plot_data
        }

        if use_nonres:
            # Only the background_nonres bucket is rescaled by the fitted
            # per-category factors, exactly like GMMFromScores.call/
            # chunk_yields do for the differentiable loss - via
            # _rescale_hist_by_category (Hist*array isn't a guaranteed
            # operation) plus ordinary Hist+Hist addition.
            background_hist = (
                _rescale_hist_by_category(filled["background_nonres"], factors)
                + filled["background_other"]
            )
        else:
            background_hist = filled["background"]

        S_hard = hist_values(filled["signal"])
        B_hard = hist_values(background_hist)
        z_hard = float(np.sqrt(np.sum(np.asarray(asymptotic_significance(S_hard, B_hard)) ** 2)))
        if verbose:
            print(f"[evaluate][{variant}] Hard-bin Z (test only) = {z_hard:.4f}")

        hist_plot_path = os.path.join(eval_out_dir, f"optimized_histograms_test_{variant}.{plot_dtype}")
        scale_signal = 100
        plot_stacked_histograms(
            stacked_hists=[background_hist],
            process_labels=["background"],
            signal_hists=[filled["signal"] * scale_signal],
            signal_labels=[f"{scale_signal} x signal"],
            output_filename=hist_plot_path,
            axis_labels=("Category index", "Weighted events"),
            normalize=False,
            log=True,
        )

        results[variant] = {
            "Z": float(Z),
            "Z_hard": z_hard,
            "sig_loss": float(sig_loss),
            "penalty": float(penalty),
            "unc_penalty": float(unc_penalty),
            "cross_entropy": float(ce_mean),
            "mean_bias": mean_bias,
            "n_test_events_after_cut": n_test_final,
            "S_hard_per_category": S_hard.tolist(),
            "B_hard_per_category": B_hard.tolist(),
            "nonres_reweight_factors": factors.tolist() if (use_nonres and factors is not None) else None,
            "histogram_plot": hist_plot_path,
        }

    json_path = os.path.join(eval_out_dir, "test_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "n_test_events_preselected": n_test_preselected,
                # Per-variant counts now (not a single shared number): the
                # "with_nonres_reweight" variant exempts the non-resonant
                # class from the hard mgg cut (see keep_by_variant above),
                # so its count can differ from "without_nonres_reweight" -
                # see each entry in "results" for "n_test_events_after_cut".
                "n_test_events_after_hard_mgg_cut_without_nonres_reweight": int(keep_without.size),
                "n_test_events_after_hard_mgg_cut_with_nonres_reweight": int(keep_with.size),
                "mgg_window": [mw_low, mw_high],
                "results": results,
            },
            f, indent=2,
        )

    csv_path = os.path.join(eval_out_dir, "test_evaluation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["variant", "Z", "Z_hard", "sig_loss", "penalty", "unc_penalty", "cross_entropy",
             "mean_bias", "n_test_events_after_cut"]
        )
        for variant, r in results.items():
            writer.writerow([
                variant, r["Z"], r["Z_hard"], r["sig_loss"], r["penalty"], r["unc_penalty"], r["cross_entropy"],
                r["mean_bias"], r["n_test_events_after_cut"],
            ])

    print(f"[INFO] Saved test-set evaluation: {json_path}")
    print(f"[INFO] Saved test-set evaluation summary CSV: {csv_path}")

    return {
        "n_test_events_preselected": n_test_preselected,
        "n_test_events_after_hard_mgg_cut_without_nonres_reweight": int(keep_without.size),
        "n_test_events_after_hard_mgg_cut_with_nonres_reweight": int(keep_with.size),
        "results": results,
    }


def evaluate_all_splits(
    classifier, gato, signal_class, plot_dim, raw_data, mass_train, mass_val, mass_test,
    training_config, classes, out_dir,
    nonres_reweight_fn=compute_nonres_reweight_factors,
    chunk_size=None, plot_dtype="pdf", verbose=True,
):
    """
    Runs evaluate_split_significance on train, val, AND test (each split
    fit against its OWN sideband, cut with its OWN hard mgg window - fully
    independent per split, same as evaluate_on_test_set's test-only
    treatment just repeated 3x), then assembles ONE master summary table
    across every (split, variant, metric) combination:

        split,variant,metric,cat_0,...,cat_<n_cats-1>,combined

    with metric in {Z_soft, Z_hard} - Z_soft as plain per-category floats
    (no uncertainty - see evaluate_split_significance's docstring), Z_hard
    as "value +/- uncertainty" strings, matching the CSV format supplied
    when this function was requested.

    Writes, under out_dir/<evaluation.output_subdir>/:
        events_table_{split}_{variant}.csv / .{plot_dtype}   (6 files -
            one CSV + one PDF per split x variant combination, from
            evaluate_split_significance/write_split_variant_events_table)
        significance_summary.csv                              (exact
            format above)
        significance_summary.{plot_dtype}                     (grey-
            shaded reportlab rendering of the same table)

    Returns the summary as a pandas DataFrame (same rows/columns written
    to significance_summary.csv).
    """
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    eval_cfg = training_config.get("evaluation", {}) or {}
    output_subdir = eval_cfg.get("output_subdir", "evaluation")
    eval_out_dir = os.path.join(out_dir, output_subdir)
    os.makedirs(eval_out_dir, exist_ok=True)

    split_specs = [
        ("train", raw_data["X_train"], raw_data["y_train"], raw_data["rel_w_train"], mass_train),
        ("val", raw_data["X_val"], raw_data["y_val"], raw_data["rel_w_val"], mass_val),
        ("test", raw_data["X_test"], raw_data["y_test"], raw_data["rel_w_test"], mass_test),
    ]

    all_results = {}
    for split_name, X, y, w, mass in split_specs:
        if verbose:
            print(f"\n[evaluate_all_splits] --- {split_name} ---")
        all_results[split_name] = evaluate_split_significance(
            classifier, gato, signal_class, plot_dim, X, y, w, mass,
            training_config, classes, eval_out_dir, split_name,
            nonres_reweight_fn=nonres_reweight_fn,
            chunk_size=chunk_size, plot_dtype=plot_dtype, verbose=verbose,
        )

    # ---- master summary rows: split, variant, metric, cat_0.. combined ----
    n_cats = gato.n_cats
    cat_cols = [f"cat_{k}" for k in range(n_cats)]
    columns = ["split", "variant", "metric"] + cat_cols + ["combined"]
    rows = []
    for split_name, _, _, _, _ in split_specs:
        for variant in ("without_nonres_reweight", "with_nonres_reweight"):
            r = all_results[split_name].get(variant)
            if r is None:
                continue  # this variant was skipped for this split (no nonres factors)

            z_soft = r["z_soft_per_cat"]
            rows.append(
                [split_name, variant, "Z_soft"]
                + [f"{v:.3f}" for v in z_soft]
                + [f"{r['Z_soft']:.3f}"]
            )

            z_hard = r["z_hard_per_cat"]
            z_hard_unc = r["z_hard_per_cat_unc"]
            rows.append(
                [split_name, variant, "Z_hard"]
                + [f"{v:.3f} +/- {u:.3f}" for v, u in zip(z_hard, z_hard_unc)]
                + [f"{r['Z_hard']:.3f} +/- {r['Z_hard_unc']:.3f}"]
            )

    summary_df = pd.DataFrame(rows, columns=columns)

    csv_path = os.path.join(eval_out_dir, "significance_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved significance summary CSV: {csv_path}")

    # ---- master summary PDF: grey-shaded reportlab table, same styling
    # convention as write_split_variant_events_table/make_event_count_table
    # ("grey shading to make navigation inside the table easier") - here
    # additionally shading whole (split, variant) BLOCKS (2 rows each:
    # Z_soft then Z_hard) in alternation, since that's the natural unit a
    # reader scans one at a time in this particular table, on top of the
    # header shading every table in this file already uses. ----
    pdf_path = os.path.join(eval_out_dir, f"significance_summary.{plot_dtype}")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=landscape(A4),
        rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20,
    )
    styles = getSampleStyleSheet()
    table_rows = [columns] + rows
    table = Table(table_rows, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    # Shade every other (split, variant) BLOCK of 2 rows (Z_soft + Z_hard),
    # not every other single row - a Z_soft/Z_hard pair belongs together.
    for block_idx in range(len(rows) // 2):
        if block_idx % 2 == 1:
            row_start = 1 + block_idx * 2
            row_end = row_start + 1
            style_cmds.append(("BACKGROUND", (0, row_start), (-1, row_end), colors.whitesmoke))
    table.setStyle(TableStyle(style_cmds))
    doc.build([
        Paragraph("Significance summary (soft vs. hard, all splits &amp; variants)", styles["Title"]),
        Spacer(1, 12),
        table,
    ])
    print(f"[INFO] Saved significance summary PDF: {pdf_path}")

    return summary_df