#!/usr/bin/env python3
"""
discriminant_density_diagnostics.py

Tracks how the CLASSIFIER's raw score outputs are actually spread out in
discriminant space, per true class, during Stage-B finetuning - a more
detailed complement to the scalar cross-entropy loss. For every pair of
score dimensions, draws contours enclosing 68.27% / 95.45% / 99.73% of
each class's weighted density ("1/2/3-sigma-EQUIVALENT" contours - the
same enclosed-probability-mass convention corner.py/getdist use for
non-Gaussian posteriors), estimated directly from a weighted 2D histogram
of the data. No shape is assumed - multi-modal, skewed, banana-shaped
clusters all come out looking like themselves, not like forced ellipses.

--------------------------------------------------------------------------
Full softmax, not model.scores() - all C(n_classes, 2) pairs, always
--------------------------------------------------------------------------
EndToEndClassifierGMM.scores() returns the softmax with the last column
DISCARDED when drop_last_softmax_class is on (gato never sees it). This
module intentionally does NOT call model.scores() - it recomputes the
full, untruncated softmax directly from model.classifier's logits (see
compute_class_dim_histograms), so every one of the classifier's
n_classes raw outputs is tracked, including the one gato ignores. That
dimension still moves during finetuning (the classifier keeps producing
it - the softmax normalization means every output weight, including the
dropped column's, keeps getting gradient - see EndToEndClassifierGMM's
drop_last_class docstring), so its evolution is a legitimate thing to be
able to spot problems in even though it plays no role in categorization.

dims therefore defaults to ALL C(n_classes, 2) pairs (6 for 4 classes),
not just the (n_classes - 1) gato actually uses. Axis labels use the
actual class name (from `classes`, e.g. "P(is_ggHH_sig)") since each
dimension now corresponds 1:1 to one of the real classes, not to an
arbitrary kept/dropped GMM index.

--------------------------------------------------------------------------
Why a histogram, not raw event storage, and not a Gaussian fit
--------------------------------------------------------------------------
A weighted 2D histogram is a fixed-size (bins x bins) sufficient
statistic for "what did the density look like", with no assumption at
all about its shape, reconstructable to any contour level after the fact
with no access to the original data or model. Costs, both controllable:

  1. Compute: one classifier forward pass per split per snapshot,
     independent of `bins` (see compute_class_dim_histograms - increasing
     bins only adds ~O(bins) histogram-fill overhead, not another forward
     pass). Bounded via `max_events_per_class` and this diagnostic's own
     `recompute_every` cadence.
  2. Storage: sparse long-format CSV - only bins with nonzero weight are
     written, since softmax scores only populate part of the unit
     square/simplex. Grows roughly as bins**~1.5 in practice (measured),
     not bins**2, because finer bins slice existing support into more
     cells rather than covering more area.

--------------------------------------------------------------------------
Splits: train / val (periodic) and test (one-off)
--------------------------------------------------------------------------
All three splits share ONE CSV schema (adds a `split` column) so a
single file/meta pair can hold the full picture. append_class_dim_
histograms is called with split="train"/"val" every `recompute_every`
epochs during Stage-B (see the integration notes), and
save_test_set_histograms is called ONCE, after training, on the held-out
test split (mirrors evaluate_model.py's test-only, always-unconditional
outputs - see training_config.yaml's `evaluation` block for that
convention). Test rows use epoch=-1 as a sentinel so they're never
mistaken for "the latest training epoch" by the default plotting call.
"""

import csv
import itertools
import json
import os

import numpy as np

# Only needed for the training-time forward pass (compute_class_dim_
# histograms); guarded so the plotting-only path (plot_discriminant_
# density_from_csv) still works without a TensorFlow install.
try:
    import tensorflow as tf
except ImportError:
    tf = None

TEST_SPLIT_EPOCH_SENTINEL = -1

# Single source of truth for the "safe ceiling" forward-pass chunk size,
# used as BOTH compute_class_dim_histograms'/save_test_set_histograms'
# own default parameter AND the fallback in training_config.yaml's
# discriminant_density_diagnostics.chunk_size (see integration notes) -
# so the two can't silently drift apart. Matches training.py's own
# documented reference point: ~40kB/event retained activations for this
# classifier shape, i.e. ~800MB at this chunk size. Deliberately NOT
# derived from training_config.yaml's `training.chunk_size` (Stage B's
# own, separately-tuned step - see the OOM discussion this was born
# from: that config had chunk_size=500000, which is not a safe default
# to inherit here).
DEFAULT_CHUNK_SIZE = 20_000


# --------------------------------------------------------------------------
# TRAINING-TIME call: chunked forward pass -> weighted 2D histograms
# --------------------------------------------------------------------------
def _full_softmax_chunked(model, X, chunk_size):
    """Recomputes the FULL (untruncated) softmax directly from
    model.classifier's logits, in chunks - deliberately bypasses
    model.scores(), which would drop the last column when
    drop_last_softmax_class is on (see module docstring).

    chunk_size naturally never allocates more than len(X) rows in a
    single forward pass: each chunk's end index is min(start + chunk_size,
    len(X)), so whenever chunk_size is bigger than the data actually being
    scored here (a small `fraction` run, a rare class's max_events_per_
    class-capped subsample, a small val/test split, ...), the whole thing
    just becomes ONE chunk of len(X) rows - never chunk_size rows padded
    or over-allocated. This is why compute_class_dim_histograms' default
    below is a fixed, GPU-memory-safe reference size (see the ~40kB/event,
    ~800MB-at-20000 note in training.py) rather than something inherited
    from training_config.yaml's own (possibly much larger) `chunk_size` -
    it only ever gets smaller than this default, automatically, never
    bigger."""
    score_chunks = []
    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        logits = model.classifier(tf.constant(X[start:end], dtype=tf.float32), training=False)
        s_chunk = tf.nn.softmax(logits, axis=1)
        score_chunks.append(s_chunk.numpy())
    return np.concatenate(score_chunks, axis=0) if score_chunks else np.zeros((0, 0))


def compute_class_dim_histograms(model, X, y, weights, classes, dims=None,
                                  bins=160, score_range=(0.0, 1.0),
                                  chunk_size=DEFAULT_CHUNK_SIZE, max_events_per_class=None,
                                  rng_seed=0):
    """
    Builds one weighted 2D histogram per (true class, dim pair), using
    the classifier's FULL softmax output (all n_classes dims - see
    module docstring for why this is not model.scores()).

    chunk_size: deliberately NOT tied to training_config.yaml's own Stage-B
    `chunk_size` (which can be tuned much larger for a bigger GPU - see the
    OOM discussion in chat). 20_000 matches training.py's own documented
    safe reference point (~800MB of retained activations for this
    classifier shape at that size). Since _full_softmax_chunked clamps
    each chunk to min(chunk_size, rows remaining), this is a CEILING, not
    a fixed cost - a small `fraction` run or a rare class's subsample
    below max_events_per_class automatically uses smaller (cheaper)
    chunks with no extra logic needed. Raise this only once you've
    confirmed real headroom (e.g. via nvidia-smi) beyond what Stage-B's
    own training step is already using.

    max_events_per_class: if given, each class's events are randomly
    subsampled (plain uniform subsampling of rows; the kept rows' own
    physics weights are then used to fill the histogram) down to at most
    this many rows before scoring, bounding the forward-pass cost
    independent of dataset size. None -> use every event of every class.

    dims: list of (i, j) pairs. None -> ALL C(n_classes, 2) pairs (e.g. 6
    for 4 classes) - inferred from the classifier's actual output width,
    not from gato's dim.

    Returns {(class_name, i, j): (H, xedges, yedges)}, H shape (bins, bins).
    """
    y = np.asarray(y)
    weights = np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)

    probe = _full_softmax_chunked(model, X[:1], chunk_size=1)
    n_classes_out = probe.shape[1]
    if dims is None:
        dims = list(itertools.combinations(range(n_classes_out), 2))

    histograms = {}
    for cls_idx, cls_name in enumerate(classes):
        idx = np.where(y == cls_idx)[0]
        if max_events_per_class is not None and len(idx) > max_events_per_class:
            idx = rng.choice(idx, size=max_events_per_class, replace=False)
            idx.sort()  # memory-friendly ascending-order reads

        if len(idx) == 0:
            for (i, j) in dims:
                edges = np.linspace(score_range[0], score_range[1], bins + 1)
                histograms[(cls_name, i, j)] = (np.zeros((bins, bins)), edges, edges)
            continue

        scores = _full_softmax_chunked(model, X[idx], chunk_size=chunk_size)
        wc = weights[idx]

        for (i, j) in dims:
            H, xedges, yedges = np.histogram2d(
                scores[:, i], scores[:, j], bins=bins,
                range=[list(score_range), list(score_range)], weights=wc,
            )
            histograms[(cls_name, i, j)] = (H, xedges, yedges)

    return histograms


# --------------------------------------------------------------------------
# Sparse CSV history (append-only, ONE file for train + val + test)
# --------------------------------------------------------------------------
SPARSE_CSV_COLUMNS = ["epoch", "split", "class", "dim_i", "dim_j", "bin_x", "bin_y", "weight"]


def append_class_dim_histograms(csv_path, meta_path, epoch, split, histograms, bins, score_range,
                                 classes=None):
    """
    Appends the nonzero bins of every histogram in `histograms` (as
    returned by compute_class_dim_histograms) to csv_path, in sparse
    long format (SPARSE_CSV_COLUMNS).

    split: "train", "val", or "test" - written as its own column so one
    CSV/meta pair covers all three (see module docstring). Call this once
    per split per snapshot (e.g. once with split="train" and once with
    split="val" every `recompute_every` epochs; see save_test_set_
    histograms for the one-off "test" call).

    classes: pass the same `classes` list used to compute `histograms`
    the FIRST time this csv_path/meta_path pair is written, so the
    sidecar meta JSON can record class-name<->index for axis labels at
    plot time. Ignored on subsequent calls (already recorded).
    """
    if not os.path.exists(meta_path):
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({"bins": bins, "score_range": list(score_range), "classes": classes}, f)

    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(SPARSE_CSV_COLUMNS)
        for (cls_name, i, j), (H, _, _) in histograms.items():
            nz_x, nz_y = np.nonzero(H)
            for bx, by in zip(nz_x, nz_y):
                w = float(H[bx, by])
                if w == 0.0:
                    continue
                writer.writerow([epoch, split, cls_name, i, j, int(bx), int(by), w])
    return csv_path


def save_test_set_histograms(model, X_test, y_test, w_test, classes, csv_path, meta_path,
                              bins=160, score_range=(0.0, 1.0), chunk_size=DEFAULT_CHUNK_SIZE,
                              max_events_per_class=None, final_epoch=None):
    """
    ONE-OFF (not periodic) test-set snapshot, meant to be called once
    after Stage-B finishes (e.g. alongside evaluate_model.py's other
    test-only, always-unconditional outputs - see training_config.yaml's
    `evaluation` block). Writes with split="test" and
    epoch=final_epoch if given, else the TEST_SPLIT_EPOCH_SENTINEL (-1)
    - either way, distinguishable from the periodic train/val history via
    the `split` column, and the -1 sentinel specifically means "don't
    mistake this for the latest training-epoch snapshot" if final_epoch
    isn't tracked at the call site.
    """
    hists = compute_class_dim_histograms(
        model, X_test, y_test, w_test, classes, bins=bins, score_range=score_range,
        chunk_size=chunk_size, max_events_per_class=max_events_per_class,
    )
    epoch = final_epoch if final_epoch is not None else TEST_SPLIT_EPOCH_SENTINEL
    return append_class_dim_histograms(
        csv_path, meta_path, epoch, "test", hists, bins, score_range, classes=classes,
    )


# --------------------------------------------------------------------------
# Enclosed-probability ("sigma-equivalent") contour levels
# --------------------------------------------------------------------------
def hdi_levels_from_hist(H, quantiles=(0.6827, 0.9545, 0.9973)):
    """
    Density thresholds such that summing all bins with H >= threshold
    encloses each of `quantiles`' fraction of the total weighted mass -
    the corner.py/getdist convention for "1/2/3-sigma-equivalent"
    contours of an arbitrarily-shaped 2D distribution (no Gaussian
    assumption). Returns levels ascending. Zero-mass histograms return
    an all-zero level list (nothing plotted, rather than raising).
    """
    total = H.sum()
    if total <= 0:
        return [0.0 for _ in quantiles]
    flat_sorted = np.sort(H.flatten())[::-1]
    cumsum = np.cumsum(flat_sorted) / total
    levels = []
    for q in quantiles:
        idx = np.searchsorted(cumsum, q)
        idx = min(idx, len(flat_sorted) - 1)
        levels.append(flat_sorted[idx])
    return sorted(set(levels)) or [0.0]


# --------------------------------------------------------------------------
# Replotting from the CSV (no model/checkpoint/data needed)
# --------------------------------------------------------------------------
def _load_meta(meta_path):
    with open(meta_path) as f:
        return json.load(f)


def _dense_hist_from_sparse(df_bins, bins):
    H = np.zeros((bins, bins))
    for bx, by, w in zip(df_bins["bin_x"], df_bins["bin_y"], df_bins["weight"]):
        H[int(bx), int(by)] = w
    return H


def _merge_bins(H, centers, merge):
    """
    Coarsens H (bins, bins) and its 1D bin `centers` by block-summing
    merge x merge cells into one, e.g. bins=160, merge=4 -> 40x40.

    This is the right remedy for a SPARSE histogram (few events per bin,
    isolated non-adjacent populated cells -> contours trace tiny loops
    around individual cells, i.e. "dots" - see module usage notes): it
    aggregates real neighboring cells into one before computing sigma
    levels, recovering an actual local density estimate instead of
    smoothing (blurring) already-sparse individual counts, which only
    softens each dot's edges without bridging the gaps between them.
    Can be combined with a modest smooth_sigma on top of the coarsened
    grid for an even less blocky result.

    `bins` must be evenly divisible by `merge` - trailing rows/cols that
    don't fill a full block are dropped (with a warning) rather than
    silently padded with zeros, which would dilute the merged bins'
    density.
    """
    merge = int(merge)
    if merge <= 1:
        return H, centers
    bins = H.shape[0]
    usable = (bins // merge) * merge
    if usable != bins:
        print(f"[WARNING] merge_bins={merge} does not evenly divide bins={bins} - "
              f"dropping the last {bins - usable} row(s)/col(s) rather than diluting "
              f"a partial block with zeros.")
    H = H[:usable, :usable]
    centers = centers[:usable]
    n = usable // merge
    H_merged = H.reshape(n, merge, n, merge).sum(axis=(1, 3))
    centers_merged = centers.reshape(n, merge).mean(axis=1)
    return H_merged, centers_merged


def _grid_shape(n_panels, max_cols=3):
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


# --------------------------------------------------------------------------
# Shared "reference-style" per-class contour styling
# --------------------------------------------------------------------------
def styled_class_contours(ax, H, centers, color, quantiles=(0.6827, 0.9545), fill_alpha=0.35,
                           linewidth=1.8):
    """
    Draws ONE class's density the way corner.py/getdist-style plots (and
    the reference plot this was modeled on) do it: a translucent FILLED
    patch for the innermost level (default 0.6827 -> "1-sigma-equivalent"
    / 68% enclosed density) and a DASHED outline for the next level out
    (default 0.9545 -> "2-sigma-equivalent" / 95%). This reads far better
    than N overlapping unfilled line contours once classes overlap a lot,
    which is exactly the "not very well readable" case a solid/dashed/
    dotted line-only version runs into.

    quantiles: exactly 2 entries, ascending (inner, outer). Kept to 2
    (not 3) by default per the same readability concern - see module
    usage notes for why 3-sigma-equivalent tends to just add clutter
    once classes already overlap heavily.

    Returns a list of (handle, level_label) pairs for the caller to
    build a combined legend from - e.g. [(patch, "68%"), (dashed_line,
    "95%")] - rather than drawing its own legend here, since the caller
    is the one deciding the class name prefix and whether to show a
    legend on this particular panel at all.

    If the two quantiles' density thresholds collapse to a single value
    (a real, common case for a confident/well-separated class - see
    hdi_levels_from_hist's docstring - meaning the top bin/cluster alone
    already holds >=95% of the mass), only the filled patch is drawn and
    only one handle is returned: there is no meaningful "outer minus
    inner" ring left to distinguish with a dashed line at that point.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if H.max() <= 0:
        return []
    levels = hdi_levels_from_hist(H, quantiles=quantiles)
    top = H.max() * (1.0 + 1e-6)

    if len(levels) == 1:
        ax.contourf(centers, centers, H.T, levels=[levels[0], top], colors=[color], alpha=fill_alpha)
        return [(Patch(facecolor=color, alpha=fill_alpha), f"{quantiles[0] * 100:.0f}%")]

    outer_level, inner_level = levels[0], levels[-1]
    ax.contourf(centers, centers, H.T, levels=[inner_level, top], colors=[color], alpha=fill_alpha)
    ax.contour(centers, centers, H.T, levels=[outer_level], colors=[color],
               linestyles="dashed", linewidths=linewidth)
    return [
        (Patch(facecolor=color, alpha=fill_alpha), f"{quantiles[0] * 100:.0f}%"),
        (Line2D([0], [0], color=color, linestyle="dashed", linewidth=linewidth), f"{quantiles[1] * 100:.0f}%"),
    ]


def plot_discriminant_density_from_csv(csv_path, meta_path, output_path, split="train",
                                        epoch=None, dims=None, quantiles=(0.6827, 0.9545),
                                        smooth_sigma=0.0, merge_bins=1):
    """
    Reads the sparse histogram history CSV + its sidecar meta JSON and
    draws ONE combined figure: a grid of subplots (one per dim pair,
    all C(n_classes, 2) of them by default - see module docstring), each
    showing every true class's 1/2/3-sigma-equivalent density contours.

    split: "train", "val", or "test" - which rows to read (the CSV holds
    all three; see module docstring). Required, since epoch numbering
    only makes sense within one split (test uses a sentinel epoch).

    epoch: which epoch's rows to plot, within `split`. None -> the
    latest epoch present for that split (for split="test" there's
    normally only one snapshot, so this just picks it up automatically).

    merge_bins: block-sum this many x this many bins into one before
    computing contours (see _merge_bins) - use this FIRST if contours
    come out looking like isolated "dots" instead of smooth regions (a
    sparse-histogram symptom: too many bins for the available event
    count, especially for max_events_per_class-capped or naturally rare
    classes). 1 (default) = no merging.

    smooth_sigma: optional Gaussian smoothing (bin units, applied AFTER
    merge_bins so it operates on the already-coarsened grid) for a less
    blocky-looking contour. 0 (default) = no smoothing. Note this does
    NOT fix the "dots" symptom above by itself at small values (e.g. 0.5)
    - it softens each isolated bin's edges without bridging the gaps
    between them; merge_bins is the right first lever for that, larger
    smooth_sigma (2-4) can help further on top.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    meta = _load_meta(meta_path)
    bins = meta["bins"]
    score_range = meta["score_range"]
    class_names = meta.get("classes")
    edges = np.linspace(score_range[0], score_range[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    df = pd.read_csv(csv_path)
    df_split = df[df["split"] == split]
    if df_split.empty:
        available = sorted(df["split"].unique().tolist())
        raise ValueError(f"No rows for split={split!r} in {csv_path}. Available splits: {available}")

    if epoch is None:
        epoch = int(df_split["epoch"].max())
    df_ep = df_split[df_split["epoch"] == epoch]
    if df_ep.empty:
        available = sorted(df_split["epoch"].unique().tolist())
        raise ValueError(f"No rows for split={split!r}, epoch={epoch} in {csv_path}. Available epochs: {available}")

    if dims is None:
        pairs = sorted(df_ep[["dim_i", "dim_j"]].drop_duplicates().itertuples(index=False, name=None))
    else:
        pairs = dims
    classes_present = sorted(df_ep["class"].unique().tolist())
    cmap = plt.get_cmap("tab10")

    def axis_label(dim_idx):
        if class_names and dim_idx < len(class_names):
            return f"P({class_names[dim_idx]})"
        return f"score dim {dim_idx}"

    nrows, ncols = _grid_shape(len(pairs))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    legend_handles, legend_labels = [], []
    for panel_idx, (i, j) in enumerate(pairs):
        ax = axes_flat[panel_idx]
        df_pair = df_ep[(df_ep["dim_i"] == i) & (df_ep["dim_j"] == j)]
        for c_idx, cls in enumerate(classes_present):
            color = cmap(c_idx % 10)
            df_cls = df_pair[df_pair["class"] == cls]
            if df_cls.empty:
                continue
            H = _dense_hist_from_sparse(df_cls, bins)
            plot_centers = centers
            if merge_bins > 1:
                H, plot_centers = _merge_bins(H, centers, merge_bins)
            if smooth_sigma > 0:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=smooth_sigma)
            if H.max() <= 0:
                print(f"[WARNING] dims {i} vs {j}, class {cls!r}: no weight in this histogram "
                      f"(0 events reached this bin range) - skipping this class's contours.")
                continue
            # Filled inner-level patch + dashed outer-level line (see
            # styled_class_contours docstring for why this replaces the
            # old dotted/dashed/solid unfilled-line-only version: once
            # classes overlap a lot, semi-transparent fills read far
            # better than stacked line contours). Levels CAN legitimately
            # collapse to one (a confident, well-separated class puts
            # >=95% of the mass in the top bin/cluster already) - that's
            # handled inside styled_class_contours, not skipped here.
            pair_handles = styled_class_contours(ax, H, plot_centers, color, quantiles=quantiles)
            if panel_idx == 0:
                for handle, level_label in pair_handles:
                    legend_handles.append(handle)
                    legend_labels.append(f"{cls} ({level_label} contour)")
        ax.set_xlabel(axis_label(i), fontsize=14)
        ax.set_ylabel(axis_label(j), fontsize=14)
        ax.set_title(f"dims {i} vs {j}", fontsize=14)
        ax.set_xlim(score_range)
        ax.set_ylim(score_range)
        ax.grid(alpha=0.3)
        if panel_idx == 0 and legend_handles:
            ax.legend(legend_handles, legend_labels, fontsize=8, ncol=1)

    for k in range(len(pairs), len(axes_flat)):
        axes_flat[k].axis("off")

    epoch_label = "test set" if split == "test" else f"epoch {epoch}"
    level_pct = [f"{q * 100:.0f}%" for q in quantiles]
    fig.suptitle(
        f"Discriminant-space density per true class - {split} split, {epoch_label}\n"
        f"(filled = {level_pct[0]} / dashed = {level_pct[-1]} enclosed density, "
        f"1/2$\\sigma$-equivalent)",
        fontsize=20,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved: {output_path}")
    return output_path