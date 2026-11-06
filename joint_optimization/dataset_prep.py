#!/usr/bin/env python3
"""
dataset_prep.py

Shared dataset-building logic, used by BOTH:
  - this module's own CLI (`python dataset_prep.py --base-dir ... --output-dir
    ...` - see main() at the bottom): samples once, writes X_train.npy/etc.
    to disk - still useful for evaluate_model.py, which expects a cached
    --input-path-style directory, and for quick repeated experimentation
    without re-reading individual_samples/ every run. This used to be a
    separate script, prepare_inputs.py; it was folded into this file
    (2024, see git history) since it was a thin (~130-line) CLI wrapper
    around build_training_dataset() below with no logic of its own - two
    files were harder to keep in sync than one, for no benefit now that
    end_to_end_training.py no longer requires the separate offline step
    either (see next bullet). Run it as
    `python dataset_prep.py --base-dir ... --output-dir ... --config ...
    --fraction ...` - same flags prepare_inputs.py used to take.
  - end_to_end_training.py (calls build_training_dataset() directly, in
    process, every run - no separate offline caching step required)

so the two never drift apart on how samples are loaded, rescaled, split, or
weighted.

Reads the per-era/per-sample outputs of the upstream skimming step
(individual_samples/<era>/<sample>/{X.npy, rel_w.npy, events.parquet}),
builds one-hot class labels and process numbers from a training config,
samples a --fraction of each file (rescaled per era/sample back to a full-
dataset-equivalent weight sum - MECHANISM 1), splits everything into a
train/val/test set stratified by era/sample so every era/sample is split
at (as close as possible to) the same ratio, rescales each split's weights
per era/sample back to the full-dataset-equivalent weight sum (MECHANISM
2, the same granularity as mechanism 1), OPTIONALLY applies the classifier
(Stage A) preselection (see classifier_preselection_mask below - a fixed
mass/dijet-mass/mvaID window, DIFFERENT from the "gato model" (Stage B)
preselection in end_to_end_helpers.py, which is broader and fully
config-driven, and is applied downstream in combined_optimization.py
instead, never here), and computes the class weight arrays needed for
classifier (Stage A) training - AFTER the preselection, if one was applied,
so the "two normal backgrounds sum to one each, the two di-Higgs processes
combined sum to one" weight normalization is computed on the exact events
the classifier will actually train/validate/test on, not on a
pre-preselection population that then gets partially cut away underneath it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# Fixed, superset list of every era subdirectory name that has ever existed
# under individual_samples/ for this analysis - kept in sync with
# gato_categorization_merged.py's load_samples(), which uses this exact
# same tuple as ITS default for the same reason (see that function's
# comment). Used as the DEFAULT for data_prep.eras in training_config.yaml
# (and for prepare_inputs.py's --eras) instead of discover_eras()'s
# auto-detection: trying a fixed, known-superset list every run makes data
# loading independent of whatever era directories happen to exist under
# --base-dir right now (auto-detection would silently pick up or drop an
# era if base-dir's contents change between runs). This is always safe to
# try in full: build_dataset() below already skips (and logs under
# "missing_dirs") any era - or any era/sample combination - that isn't
# actually present on disk, rather than failing, so eras not yet produced
# (e.g. "2025") are simply skipped, not an error.
DEFAULT_ERAS = (
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025",
    "preEE", "postEE", "preBPix", "postBPix",
)


# ----------------------------------------------------------------------------
# CLASSIFIER (Stage A / pretrain.py) PRESELECTION
#
# TWO SEPARATE preselection functions now exist in this pipeline, and they
# are NOT the same cut:
#   - THIS one (classifier_preselection_mask) - used ONLY for classifier
#     (Stage A) pretraining, applied right here in build_training_dataset,
#     BEFORE the classifier weight (class_weights_for_set) computation. It
#     is a fixed, unconditional 4-cut mass/dijet-mass/mvaID window - EXACTLY
#     prepare_inputs.py's preselection() - with no per-cut enable switches.
#   - The "gato model" (Stage B / combined_optimization.py) preselection -
#     build_static_preselection_cuts/build_score_preselection_cut in
#     end_to_end_helpers.py - a broader, fully config-driven set of cuts
#     (adds an is_boosted veto and a classifier-score cut on top of the same
#     mass/dijet-mass/mvaID window, each independently toggleable via
#     training_config["preselection"]) applied downstream in
#     combined_optimization.py itself, AFTER the classifier weight is
#     already computed (that pipeline only DROPS rows via the cut, it never
#     recomputes/renormalizes class weights afterward - see
#     combined_optimization.py's PRESELECTION block).
#
# Column-name constants below match the parquet column names combined_
# optimization.py's own extra_column_logical_name dict already uses for the
# same physical quantities, so both pipelines read identical columns.
CLASSIFIER_PRESELECTION_DIJET_MASS_COLUMN = "nonResReg_vbfpair_dijet_mass_DNNreg"
CLASSIFIER_PRESELECTION_LEAD_MVAID_COLUMN = "lead_mvaID"
CLASSIFIER_PRESELECTION_SUBLEAD_MVAID_COLUMN = "sublead_mvaID"
CLASSIFIER_PRESELECTION_COLUMNS = (
    CLASSIFIER_PRESELECTION_DIJET_MASS_COLUMN,
    CLASSIFIER_PRESELECTION_LEAD_MVAID_COLUMN,
    CLASSIFIER_PRESELECTION_SUBLEAD_MVAID_COLUMN,
)


def classifier_preselection_mask(mass, dijet_mass, lead_mvaid, sublead_mvaid):
    """EXACTLY prepare_inputs.py's preselection() - the classifier (Stage A)
    pretraining preselection. See the module-level comment above this
    function for how this differs from the "gato model" (Stage B)
    preselection in end_to_end_helpers.py."""
    mass = np.asarray(mass)
    dijet_mass = np.asarray(dijet_mass)
    lead_mvaid = np.asarray(lead_mvaid)
    sublead_mvaid = np.asarray(sublead_mvaid)
    return (
        (mass > 100) & (mass < 180)
        & (dijet_mass > 70) & (dijet_mass < 190)
        & (lead_mvaid > -0.7) & (sublead_mvaid > -0.7)
    )


def classifier_preselection_kwargs(data_prep_cfg):
    """Given a training_config["data_prep"] dict, resolves the kwargs to
    forward to build_training_dataset/build_raw_dataset_from_samples for
    Stage-A's own preselection: {"apply_classifier_preselection": bool,
    "extra_columns": tuple}. Centralizes "is it on, and which extra_columns
    does it need" so pretrain.py and anything that reconstructs the exact
    same Stage-A dataset (evaluate_pretrain_suite.py, ...) never drift out
    of sync on it.

    data_prep_cfg["apply_classifier_preselection"]: defaults to True - the
    classifier IS preselected now, by design (see module docstring). Set to
    false to restore the old "pretrain on the full, unpreselected sample"
    behavior.
    """
    enabled = bool(data_prep_cfg.get("apply_classifier_preselection", True))
    return {
        "apply_classifier_preselection": enabled,
        "extra_columns": CLASSIFIER_PRESELECTION_COLUMNS if enabled else (),
    }


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _to_jsonable(obj):
    """Recursively converts numpy scalars/arrays (and Path objects) inside a
    nested dict/list structure into plain, json.dump-able Python types, so
    the data_info dict built up below (which freely mixes numpy floats/ints/
    arrays with plain values) can be dumped as-is."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_data_info(data_info, path):
    """Writes a build_training_dataset() result's "data_info" entry out as a
    JSON file (creating parent directories as needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(data_info), f, indent=2)
    return path


def process_numbers_from_config(sample_to_class):
    """Index of each sample within its class (same logic as prepare_inputs.py),
    plus the number of processes per class."""
    num_process_each_class = {}
    process_number = {}
    for sample, cls in sample_to_class.items():
        process_number[sample] = num_process_each_class.get(cls, 0)
        num_process_each_class[cls] = num_process_each_class.get(cls, 0) + 1
    return process_number, num_process_each_class


def load_and_sample_one(sample_dir, fraction, rng, extra_columns=()):
    """Loads X.npy, rel_w.npy and the 'mass' column (plus any requested
    extra_columns, e.g. for additional preselection cuts - see
    combined_optimization_config.yaml's preselection.dijet_mass/lead_mvaid/
    sublead_mvaid/is_boosted_veto) of events.parquet for one era/sample
    directory, returns a consistently-sampled subset of all of them (same
    random row indices throughout), rescaled to a full-dataset-equivalent
    weight sum.

    MECHANISM 1 (fraction-sampling rescale): --fraction samples EVERY
    era/sample directory at the exact same fraction, but a random subsample
    of a finite-statistics sample still fluctuates around that fraction in
    total weight (more so for low-count samples). This rescales THIS ONE
    sample's subsampled rel_w so its sum matches THIS SAME sample's own
    full (pre-sampling) weight sum exactly:

        rescale = sum(rel_w_full) / sum(rel_w_sampled)

    applied at the same per-sample granularity the sampling itself happens
    at (one sample_dir = one physical process, one era). This keeps
    cross-section ratios BETWEEN processes (weight_scheme_process: "xsec"
    in class_weights_for_set) and any significance computed from rel_w
    unaffected by which random subset --fraction happened to draw - the
    per-SAMPLE analogue of rescale_split_weights_per_key below (mechanism
    2), just applied here instead of after the train/val/test split.

    Returns (X_sub, rel_w_sub, mass_sub, extra_sub, full_weight_sum,
    fraction_rescale) - extra_sub is {column_name: array}, one entry per
    name in extra_columns, sampled/row-aligned exactly like mass_sub.
    """
    X = np.load(sample_dir / "X.npy", mmap_mode="r")
    rel_w = np.load(sample_dir / "rel_w.npy", mmap_mode="r")
    parquet_cols = ["mass"] + [c for c in extra_columns if c != "mass"]
    events = pd.read_parquet(sample_dir / "events_boostedCat.parquet", columns=parquet_cols)
    mass = events["mass"].to_numpy()

    n_rows = X.shape[0]
    if not (rel_w.shape[0] == n_rows == mass.shape[0]):
        raise ValueError(
            f"{sample_dir}: row count mismatch "
            f"(X={n_rows}, rel_w={rel_w.shape[0]}, mass={mass.shape[0]})"
        )

    full_weight_sum = float(np.asarray(rel_w, dtype=np.float64).sum())

    n_samples = max(1, int(round(n_rows * fraction)))
    idx = rng.choice(n_rows, size=n_samples, replace=False)

    X_sub = np.asarray(X[idx])
    rel_w_sub = np.asarray(rel_w[idx])
    mass_sub = mass[idx]
    extra_sub = {c: events[c].to_numpy()[idx] for c in extra_columns if c != "mass"}

    sampled_weight_sum = float(rel_w_sub.astype(np.float64).sum())
    fraction_rescale = (full_weight_sum / sampled_weight_sum) if sampled_weight_sum != 0 else 1.0
    rel_w_sub = (rel_w_sub.astype(np.float64) * fraction_rescale).astype(rel_w_sub.dtype)

    return X_sub, rel_w_sub, mass_sub, extra_sub, full_weight_sum, fraction_rescale


def discover_eras(base_dir):
    """All subdirectories directly under base_dir (ignores files like
    mean_std_dict.pkl). NOTE: no longer the default era-resolution strategy
    for prepare_inputs.py/end_to_end_training.py - both now default to the
    fixed DEFAULT_ERAS list above instead (see its docstring for why); this
    function is still available and used automatically whenever --eras/
    data_prep.eras is left unset AND the caller explicitly opts into
    auto-detection instead of the fixed list."""
    return sorted(p.name for p in Path(base_dir).iterdir() if p.is_dir())


def build_dataset(base_dir, config, fraction, fraction_seed, eras, verbose=True, info=None, extra_columns=()):
    """Loads + fraction-samples every era/sample directory listed in
    config["sample_to_class"], applies MECHANISM 1, and pools everything
    into flat arrays. Returns (X, y, rel_w, mass, proc_num, sample_name,
    sample_key, classes, num_process_each_class, extra_data) - extra_data
    is {column_name: array}, one entry per name in extra_columns (e.g.
    "nonResReg_vbfpair_dijet_mass_DNNreg", "lead_mvaID", "is_boosted" - see
    combined_optimization_config.yaml's preselection block), read from
    events_boostedCat.parquet alongside mass. y is one-hot, sample_name is
    a per-row object array of physical sample names (pooled across eras)
    used for stratified splitting downstream.

    fraction_seed: seeds ONLY the --fraction row-subsampling done here
    (rng = np.random.default_rng(fraction_seed) below) - i.e. which random
    subset of each era/sample file's rows gets pulled into the pool at
    all. This is deliberately a separate seed from the later train/val/
    test SPLIT of that pool (see build_training_dataset's split_seed) -
    varying one while holding the other fixed lets you isolate which
    source of randomness actually drives a result.

    info (optional): an existing dict to fill IN PLACE with everything this
    function otherwise only prints (per era/sample sampling stats,
    found/missing/ignored directories, raw row counts) - passed in by
    build_training_dataset so that data can be written to data_info.json
    alongside the terminal output, without changing this function's return
    signature or re-reading anything from disk.
    """
    sample_to_class = config["sample_to_class"]
    classes = config["classes"]
    process_number_map, num_process_each_class = process_numbers_from_config(sample_to_class)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    rng = np.random.default_rng(fraction_seed)

    X_list, rel_w_list, mass_list, y_list, proc_num_list = [], [], [], [], []
    sample_name_list, sample_key_list = [], []
    extra_lists = {c: [] for c in extra_columns}
    found_dirs, missing_dirs, ignored_dirs = [], [], []
    total_raw_rows = 0
    sample_info = []

    def log(msg):
        if verbose:
            print(msg)

    for era in eras:
        era_dir = Path(base_dir) / era
        on_disk_samples = {p.name for p in era_dir.iterdir() if p.is_dir()} if era_dir.exists() else set()

        for sample, cls in sample_to_class.items():
            sample_dir = era_dir / sample
            if not sample_dir.exists():
                missing_dirs.append(str(sample_dir))
                continue
            found_dirs.append(str(sample_dir))

            X, rel_w, mass, extra, full_w_sum, frac_rescale = load_and_sample_one(
                sample_dir, fraction, rng, extra_columns=extra_columns
            )
            n = X.shape[0]
            n_raw = np.load(sample_dir / "X.npy", mmap_mode="r").shape[0]
            total_raw_rows += n_raw
            log(
                f"  {era}/{sample}: {n_raw:,} rows total -> sampled {n:,} rows "
                f"(full weight sum {full_w_sum:.6g}, fraction-rescale factor {frac_rescale:.6g})"
            )
            sample_info.append({
                "era": era,
                "sample": sample,
                "class": cls,
                "raw_rows": int(n_raw),
                "sampled_rows": int(n),
                "full_weight_sum": full_w_sum,
                "fraction_rescale_factor": frac_rescale,
            })

            y = np.zeros((n, len(classes)), dtype=np.float32)
            y[:, class_to_idx[cls]] = 1.0

            X_list.append(X)
            rel_w_list.append(rel_w)
            mass_list.append(mass)
            y_list.append(y)
            proc_num_list.append(np.full(n, process_number_map[sample], dtype=np.int64))
            for c in extra_columns:
                extra_lists[c].append(extra[c])
            # Physical-sample identity per row, pooled across eras (kept for
            # any caller that still wants the coarser grouping, e.g.
            # plotting by sample regardless of era).
            sample_name_list.append(np.full(n, sample, dtype=object))
            # era/sample identity per row - the actual granularity the split
            # and the weight-sum rescale (mechanism 2) now both operate at,
            # so the same era+sample combination is balanced/rescaled on its
            # own instead of being pooled with itself across other eras.
            sample_key_list.append(np.full(n, f"{era}/{sample}", dtype=object))

        # Samples present on disk for this era but NOT listed in sample_to_class
        # (config): these are silently excluded from the dataset - flagged here
        # so a missing/forgotten process shows up instead of just a smaller total.
        unmatched = on_disk_samples - set(sample_to_class.keys())
        ignored_dirs.extend(f"{era}/{s}" for s in sorted(unmatched))

    log(f"\nFound {len(found_dirs)} era/sample directories, {len(missing_dirs)} missing.")
    if missing_dirs:
        log("Missing (skipped, listed in config but not found on disk):")
        for d in missing_dirs:
            log(f"  - {d}")
    if ignored_dirs:
        log("On disk but NOT in sample_to_class (skipped, check if these should be added to the config):")
        for d in ignored_dirs:
            log(f"  - {d}")
    log(f"Raw rows across all found directories (pre-sampling): {total_raw_rows:,}")

    if not X_list:
        raise RuntimeError(
            f"No matching era/sample directories found under base_dir={base_dir} "
            f"for eras={eras}."
        )

    X = np.concatenate(X_list, axis=0)
    rel_w = np.concatenate(rel_w_list, axis=0)
    mass = np.concatenate(mass_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    proc_num = np.concatenate(proc_num_list, axis=0)
    sample_name = np.concatenate(sample_name_list, axis=0)
    sample_key = np.concatenate(sample_key_list, axis=0)
    extra_data = {c: np.concatenate(extra_lists[c], axis=0) for c in extra_columns}

    if info is not None:
        info.update({
            "found_dirs": found_dirs,
            "missing_dirs": missing_dirs,
            "ignored_dirs": ignored_dirs,
            "total_raw_rows": total_raw_rows,
            "total_sampled_rows": int(X.shape[0]),
            "sample_info": sample_info,
        })

    return X, y, rel_w, mass, proc_num, sample_name, sample_key, classes, num_process_each_class, extra_data


def rescale_split_weights_per_key(key_full, w_full, key_split, w_split):
    """MECHANISM 2 (train/val/test split rescale), per era/sample.

    Lowest-granularity analogue of gato_categorization_merged2.py's
    rescale_split_weights(): mechanism 1 (load_and_sample_one) already
    corrects each era/sample file's own subsample back to its own full
    weight sum, BEFORE the split. This does the same correction again
    AFTER the split, at that same era/sample granularity (key = "era/
    sample"), instead of pooling everything by class first - a
    class-level rescale can't fix an individual era/sample landing off
    its ratio while the class total happens to average out.

        factor[key] = sum(w_full[key_full==key]) / sum(w_split[key_split==key])

    key_full/key_split: per-row "era/sample" identity arrays.
    w_full/w_split: matching per-event weight arrays (rel_w).

    Returns (w_split_rescaled, row_factors, factor_by_key):
      row_factors    - the factor actually applied to each row of
                        w_split (same length as w_split); dividing it back
                        out reconstructs the pre-mechanism-2 weight exactly,
                        which is what build_event_count_tables' combined-
                        table "undo" now uses instead of a per-class factor.
      factor_by_key   - {key: factor} for logging/data_info.json (1.0 for
                        any key with zero split weight sum).
    """
    w_split_rescaled = np.array(w_split, dtype=np.float64, copy=True)
    row_factors = np.ones(len(w_split), dtype=np.float64)
    factor_by_key = {}
    for key in np.unique(key_full):
        sum_full = w_full[key_full == key].sum()
        mask_split = (key_split == key)
        sum_split = w_split[mask_split].sum()
        factor = (sum_full / sum_split) if sum_split != 0 else 1.0
        factor_by_key[str(key)] = float(factor)
        row_factors[mask_split] = factor
        w_split_rescaled[mask_split] = w_split[mask_split] * factor

    return w_split_rescaled.astype(w_split.dtype), row_factors, factor_by_key


def stratified_train_test_split(arrays, sample_key, train_size, seed, verbose=True):
    """train_test_split(*arrays, train_size=..., stratify=sample_key), with
    a fallback to an UNSTRATIFIED split (with a warning) if stratification
    is impossible - sklearn's stratify requires every distinct value in
    sample_key to have at least 2 rows (so both output splits can receive
    at least 1), which a very small --fraction can violate for low-count
    era/sample combinations.

    This is what makes mechanism 2's split "take equally from every
    era/sample": each one is split at (as close to) the same train_size
    ratio, instead of relying on a single global shuffle to happen to
    preserve that ratio - what rescale_split_weights_per_key then has left
    to correct is only the residual fluctuation, not systematic skew.
    """
    try:
        return train_test_split(
            *arrays, train_size=train_size, shuffle=True, random_state=seed,
            stratify=sample_key,
        )
    except ValueError as e:
        if verbose:
            print(
                f"[WARNING] Could not stratify the split by era/sample ({e}). "
                f"Falling back to an unstratified split for this stage - "
                f"typically means --fraction is small enough that some "
                f"individual era/sample has <2 rows. The mechanism-2 rescale "
                f"(rescale_split_weights_per_key) still corrects the "
                f"resulting weight-sum dilution, just without the split "
                f"itself already being balanced per era/sample."
            )
        return train_test_split(
            *arrays, train_size=train_size, shuffle=True, random_state=seed,
        )


def class_weights_for_set(y, base_weight, proc_num, classes, weight_scheme_process,
                           num_process_each_class, class_groups=None):
    """Generic version of prepare_inputs.py's get_weights_for_training /
    get_weights_for_val_test: computes one weight per event such that, per
    class, the weights sum to 1.

    - scheme == "xsec": all events of the class are weighted by base_weight
      directly (processes with a larger cross section dominate the class).
    - scheme == "equal_weight": each process within the class is first
      normalized to sum to 1 on its own, then the (already normalized)
      per-process weights are summed and the whole class renormalized to 1
      (every process contributes equally to the class, regardless of its
      cross section or event count).

    class_groups (optional): list of {"classes": [...], "scheme": "xsec"}
    dicts. Classes are still separate one-hot columns in y/labels, but any
    class listed in a group is normalized TOGETHER with the other members of
    that group - i.e. the group's combined weight sums to 1, split between
    member classes according to their raw relative weight - instead of each
    being independently normalized to 1. Classes not listed in any group
    fall back to the per-class weight_scheme_process behavior above.
    """
    class_groups = class_groups or []
    grouped_classes = {c for g in class_groups for c in g["classes"]}
    class_to_idx = {c: i for i, c in enumerate(classes)}
    weights = np.zeros_like(base_weight, dtype=np.float64)

    for g in class_groups:
        members = g["classes"]
        scheme = g.get("scheme", "xsec")
        if scheme != "xsec":
            raise ValueError(
                f"Grouped normalization currently only supports 'xsec', got '{scheme}' for group {members}"
            )
        cls_bool = np.zeros(y.shape[0], dtype=bool)
        for c in members:
            cls_bool |= (y[:, class_to_idx[c]] == 1)
        w = base_weight * cls_bool
        s = w.sum()
        if s != 0:
            weights += w / s

    for i, cls in enumerate(classes):
        if cls in grouped_classes:
            continue
        cls_bool = (y[:, i] == 1)
        scheme = weight_scheme_process[cls]

        if scheme == "equal_weight":
            cls_weights = np.zeros_like(base_weight, dtype=np.float64)
            for proc in range(num_process_each_class[cls]):
                w = base_weight * cls_bool * (proc_num == proc)
                s = w.sum()
                if s != 0:
                    cls_weights += w / s
            s = cls_weights.sum()
            if s != 0:
                weights += cls_weights / s

        elif scheme == "xsec":
            w = base_weight * cls_bool
            s = w.sum()
            if s != 0:
                weights += w / s

        else:
            raise ValueError(f"Unknown weight_scheme_process '{scheme}' for class '{cls}'")

    return weights


def summarize_classes(y, weights, classes):
    """Per-class (n_events, weight_sum) summary, as a plain dict - the data
    backing print_class_summary's terminal output, reused here so the same
    numbers can also be written to data_info.json."""
    summary = {}
    for i, cls in enumerate(classes):
        cls_bool = (y[:, i] == 1)
        summary[cls] = {
            "n_events": int(cls_bool.sum()),
            "weight_sum": float(weights[cls_bool].sum()),
        }
    return summary


def print_class_summary(name, y, weights, classes, verbose=True):
    summary = summarize_classes(y, weights, classes)
    if verbose:
        print(f"  {name}:")
        for cls, s in summary.items():
            print(f"    {cls}: {s['n_events']:,} events, weight sum = {s['weight_sum']:.4f}")
    return summary


def build_training_dataset(base_dir, config, fraction, seed, eras,
                            train_fraction=0.7, val_fraction=0.15,
                            class_groups="auto", verbose=True,
                            info_out_path=None, extra_columns=(),
                            split_seed=None,
                            apply_classifier_preselection=False,
                            dijet_mass_column=CLASSIFIER_PRESELECTION_DIJET_MASS_COLUMN,
                            lead_mvaid_column=CLASSIFIER_PRESELECTION_LEAD_MVAID_COLUMN,
                            sublead_mvaid_column=CLASSIFIER_PRESELECTION_SUBLEAD_MVAID_COLUMN):
    """
    Top-level entry point: raw individual_samples/ -> fully split, rescaled,
    weighted training dataset, all in memory (no intermediate files).

    This is the SAME pipeline this module's own CLI runs (see main() at the
    bottom of this file: fraction sampling + mechanism 1 -> stratified
    split + mechanism 2 -> class weights) refactored into a single
    callable, so end_to_end_training.py can call it directly instead of
    requiring a separate offline CLI run + cached .npy files.

    seed: the FRACTION-SAMPLING seed (mechanism 1 - see build_dataset's
    fraction_seed) - which random --fraction subset of each era/sample
    file's rows gets pulled into the pool at all.
    split_seed: OPTIONAL, separate seed for the train/val/test stratified
    split (stratified_train_test_split) below. Defaults to `seed` if not
    given, preserving the old behavior of one seed driving both - so any
    existing caller that only ever passed `seed` is unaffected. Give a
    DIFFERENT split_seed while holding `seed` fixed to keep the exact same
    --fraction-sampled event pool across runs and vary ONLY how that pool
    gets partitioned into train/val/test - this isolates "does the split
    assignment matter" from "does which random subset of the full sample
    we happened to draw matter", instead of conflating both under one
    seed (see end_to_end_helpers.build_raw_dataset_from_samples, which
    exposes this as seeds.data / seeds.data_split).

    class_groups: "auto" (default) -> use config.get("class_groups") if the
    config has one; None -> force independent per-class normalization
    (equivalent to prepare_inputs.py's --no-class-groups); an explicit list
    -> used as-is.

    Returns a dict:
        "classes": [...],
        "X_train"/"y_train"/"rel_w_train"/"mass_train"/"proc_num_train"/"sample_name_train",
        "X_val"/... , "X_test"/... (same keys, val/test splits),
        "true_class_weights"          (train, original rel_w values)
        "class_weights_abs"           (train, abs(rel_w))
        "class_weights_only_positive" (train, negative rel_w set to 0)
        "class_weights_val"           (val, original rel_w values, incl. negatives)
        "class_weights_test"          (test, original rel_w values, incl. negatives)
        "train_rescale_factors"/"val_rescale_factors"/"test_rescale_factors" - mechanism-2
            factors, ONE PER ROW (diagnostic only - not needed to reconstruct the
            combined table anymore, see rel_w_premech2_* below).
        "rel_w_premech2_train"/"_val"/"_test" - each split's rel_w EXACTLY as it
            was right after mechanism 1 (fraction-sampling rescale), carried
            through the train/val/test split unchanged, alongside (but
            independent of) the mechanism-2-rescaled "rel_w_train"/etc.
            Concatenating these three directly reproduces the full, pre-split
            dataset's weights row-for-row - this is what build_event_count_
            tables' combined (train+val+test) table reads, instead of
            reconstructing them by dividing the mechanism-2-rescaled weights
            back out.
        "data_info": a plain (json.dump-able) dict mirroring everything this
            function otherwise only prints to the terminal - per era/sample
            sampling stats, found/missing/ignored directories, split sizes,
            mechanism-2 rescale factors, and the class-weight summaries. Callers
            that know their own output directory (this module's own CLI - see
            main() - or end_to_end_training.py) are expected to write this out
            themselves, e.g. via save_data_info(result["data_info"], out_dir / "data_info.json").
            If info_out_path is given, it's written here too, as a convenience.

    info_out_path: optional path (str or Path) to write data_info to as JSON
        directly from within this call, in addition to returning it.

    extra_columns: extra events_boostedCat.parquet columns to load and carry
        through the split alongside mass (e.g. for additional preselection
        cuts - see combined_optimization_config.yaml's preselection block).
        Each name c ends up in the returned dict as f"{c}_train"/"_val"/
        "_test", row-aligned exactly like mass_train/mass_val/mass_test.

    apply_classifier_preselection: if True, applies classifier_preselection_
        mask (EXACTLY prepare_inputs.py's preselection() - a fixed mass/
        dijet-mass/mvaID window) to each of train/val/test INDEPENDENTLY,
        right after mechanism 2's split-weight-sum rescale and BEFORE the
        class weight (true_class_weights/class_weights_abs/
        class_weights_only_positive/class_weights_val/class_weights_test)
        computation below - so those weight sums are correct for the
        actually-preselected classifier training population, not the full
        unpreselected one. This is DIFFERENT from (and independent of) the
        "gato model" (Stage B) preselection applied downstream in
        combined_optimization.py - see the module-level comment above
        classifier_preselection_mask. Requires dijet_mass_column/
        lead_mvaid_column/sublead_mvaid_column to each be present in
        extra_columns (raises ValueError otherwise) - see
        classifier_preselection_kwargs for the usual way to set both
        consistently. Default False preserves the previous behavior
        (no preselection applied inside this function) for any existing
        caller (e.g. combined_optimization.py) that doesn't pass it.

    dijet_mass_column/lead_mvaid_column/sublead_mvaid_column: parquet column
        names (as loaded via extra_columns) that classifier_preselection_
        mask's dijet_mass/lead_mvaid/sublead_mvaid arguments are read from.
        Only used when apply_classifier_preselection=True. Defaults match
        combined_optimization.py's own extra_column_logical_name mapping.

    y_* arrays are one-hot (shape (N, n_classes)), matching prepare_inputs.py's
    y_train.npy/etc. - callers that need class-index labels (e.g.
    end_to_end_helpers.py's raw_data["y_train"]) should np.argmax(y, axis=1)
    themselves, same as load_raw_dataset() already does for the cached-file
    path.
    """
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not (0 < train_fraction < 1):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not (0 < val_fraction < 1):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if train_fraction + val_fraction >= 1:
        raise ValueError(
            "train_fraction + val_fraction must be < 1 (the remainder becomes the test fraction)."
        )
    test_fraction = 1.0 - train_fraction - val_fraction
    if split_seed is None:
        split_seed = seed

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 70)
    log(f"Sampling {fraction:.1%} of each era/sample file from '{base_dir}' "
        f"(fraction_seed={seed}, split_seed={split_seed})")
    log("MECHANISM 1: each era/sample file's subsample is rescaled (per")
    log("sample) so its weight sum matches that SAME sample's own full,")
    log("pre-sampling weight sum - see load_and_sample_one.")
    log("=" * 70)
    dataset_info = {}
    X, y, rel_w, mass, proc_num, sample_name, sample_key, classes, num_process_each_class, extra_data = build_dataset(
        base_dir, config, fraction, seed, eras, verbose=verbose, info=dataset_info, extra_columns=extra_columns
    )
    log(f"\nTotal sampled events: {X.shape[0]:,}")

    log("\nSplitting into train/val/test ...")
    log(f"  train={train_fraction:.1%}, val={val_fraction:.1%}, test={test_fraction:.1%}")
    log("  Split is stratified by era/sample (lowest available granularity),")
    log("  so every individual era/sample file is divided at (as close as")
    log("  possible to) this same ratio, not left to a single global shuffle.")

    # Stage 1: split off train vs. (val + test) combined - stratified by
    # sample_key ("era/sample"), the lowest granularity available.
    # rel_w is passed through TWICE on purpose: once as the array mechanism
    # 2 will rescale in place below, and once (rel_w_premech2) that is never
    # touched again - a plain, exact, row-aligned copy of each row's
    # pre-mechanism-2 weight, carried through the split instead of derived
    # afterward. This is what build_event_count_tables' combined
    # (train+val+test) table now reads directly - no rescale-then-divide-
    # back-out round trip needed to reconstruct it.
    # extra_columns' arrays are appended at the end of the split-input list
    # (fixed count varies with what's requested), so they are unpacked
    # dynamically below rather than by a fixed positional name each.
    n_fixed = 8  # X, y, rel_w, rel_w_premech2, mass, proc_num, sample_name, sample_key
    split_arrays = [X, y, rel_w, rel_w.copy(), mass, proc_num, sample_name, sample_key] + [
        extra_data[c] for c in extra_columns
    ]
    split_out = stratified_train_test_split(
        split_arrays, sample_key, train_size=train_fraction, seed=split_seed, verbose=verbose,
    )
    train_out, rest_out = split_out[0::2], split_out[1::2]
    (X_train, y_train, rel_w_train, rel_w_premech2_train,
     mass_train, proc_num_train, sample_name_train, sample_key_train) = train_out[:n_fixed]
    (X_rest, y_rest, rel_w_rest, rel_w_premech2_rest,
     mass_rest, proc_num_rest, sample_name_rest, sample_key_rest) = rest_out[:n_fixed]
    extra_train = {c: arr for c, arr in zip(extra_columns, train_out[n_fixed:])}
    extra_rest = {c: arr for c, arr in zip(extra_columns, rest_out[n_fixed:])}

    # Stage 2: split the remainder into val vs. test, preserving the requested
    # val:test ratio, again stratified by sample_key.
    val_share_of_rest = val_fraction / (val_fraction + test_fraction)
    split_arrays_rest = [
        X_rest, y_rest, rel_w_rest, rel_w_premech2_rest, mass_rest, proc_num_rest,
        sample_name_rest, sample_key_rest,
    ] + [extra_rest[c] for c in extra_columns]
    split_out2 = stratified_train_test_split(
        split_arrays_rest, sample_key_rest, train_size=val_share_of_rest, seed=split_seed, verbose=verbose,
    )
    val_out, test_out = split_out2[0::2], split_out2[1::2]
    (X_val, y_val, rel_w_val, rel_w_premech2_val,
     mass_val, proc_num_val, sample_name_val, sample_key_val) = val_out[:n_fixed]
    (X_test, y_test, rel_w_test, rel_w_premech2_test,
     mass_test, proc_num_test, sample_name_test, sample_key_test) = test_out[:n_fixed]
    extra_val = {c: arr for c, arr in zip(extra_columns, val_out[n_fixed:])}
    extra_test = {c: arr for c, arr in zip(extra_columns, test_out[n_fixed:])}

    log(f"  train: {X_train.shape[0]:,} rows, val: {X_val.shape[0]:,} rows, test: {X_test.shape[0]:,} rows")

    split_sizes = {
        "train": int(X_train.shape[0]),
        "val": int(X_val.shape[0]),
        "test": int(X_test.shape[0]),
    }

    # MECHANISM 2: rescale rel_w (the GATO significance weight - NOT the
    # classifier CE weights below, which get independently renormalized to
    # sum=1 per class regardless of any constant rescale, so this has no
    # effect on them) so each split's weight sum matches the full (pre-
    # split, mechanism-1-rescaled) dataset's weight sum, at the SAME
    # era/sample granularity mechanism 1 already uses - see
    # rescale_split_weights_per_key's docstring. This replaces the previous
    # per-class version: a per-class rescale can't catch one era/sample
    # landing off-ratio while the class total happens to average out.
    log("\nRescaling rel_w per era/sample to the full-sample weight sum (mechanism 2) ...")
    rel_w_train, train_rescale, train_rescale_by_key = rescale_split_weights_per_key(
        sample_key, rel_w, sample_key_train, rel_w_train)
    rel_w_val, val_rescale, val_rescale_by_key = rescale_split_weights_per_key(
        sample_key, rel_w, sample_key_val, rel_w_val)
    rel_w_test, test_rescale, test_rescale_by_key = rescale_split_weights_per_key(
        sample_key, rel_w, sample_key_test, rel_w_test)
    if verbose:
        for split_name, factor_by_key in (
            ("train", train_rescale_by_key), ("val", val_rescale_by_key), ("test", test_rescale_by_key)
        ):
            print("  " + split_name + ": " + ", ".join(f"{k}={f:.6g}" for k, f in factor_by_key.items()))
    mechanism2_rescale_factors = {
        "train": train_rescale_by_key,
        "val": val_rescale_by_key,
        "test": test_rescale_by_key,
    }

    # ------------------------------------------------------------------
    # CLASSIFIER (Stage A) PRESELECTION - optional, off by default.
    #
    # Applied HERE: after mechanism 1 (fraction-sampling rescale) AND
    # mechanism 2 (split weight-sum rescale) have already run, but BEFORE
    # the class weight computation just below - so class_weights_for_set
    # normalizes each class (each of the two "normal" backgrounds
    # independently to 1, the two di-Higgs processes combined to 1 via
    # config["class_groups"] - see data_config.yaml) using only the events
    # that actually survive the cut, not the full unpreselected population.
    # Masking (not just zero-weighting) is used, same reasoning as Stage
    # B's preselection in combined_optimization.py: a rejected event should
    # not still carry a nonzero classifier weight.
    #
    # Independent per split (train/val/test each masked on their own, not
    # on a pooled array) - this is safe/expected even though the split
    # itself was already stratified by era/sample before this point: the
    # preselection cut composes on top of that split, it doesn't need to
    # know about it.
    # ------------------------------------------------------------------
    classifier_presel_cutflow = None
    if apply_classifier_preselection:
        for col in (dijet_mass_column, lead_mvaid_column, sublead_mvaid_column):
            if col not in extra_columns:
                raise ValueError(
                    f"apply_classifier_preselection=True requires '{col}' in extra_columns "
                    f"(got extra_columns={extra_columns}). See classifier_preselection_kwargs / "
                    f"CLASSIFIER_PRESELECTION_COLUMNS."
                )

        log("\nApplying CLASSIFIER (Stage A) preselection - mass/dijet-mass window + mvaID "
            "thresholds, EXACTLY prepare_inputs.py's preselection() - independently to each "
            "split, BEFORE computing class weights, so weight sums are correct for what the "
            "classifier will actually train/validate/test on. This is a SEPARATE, fixed cut "
            "from the config-driven 'gato model' (Stage B) preselection applied downstream in "
            "combined_optimization.py.")

        classifier_presel_cutflow = {}

        def _mask_split(split_name, mask_dict, extra_dict):
            mask = classifier_preselection_mask(
                mask_dict["mass"], extra_dict[dijet_mass_column],
                extra_dict[lead_mvaid_column], extra_dict[sublead_mvaid_column],
            )
            n_before = mask.shape[0]
            n_after = int(mask.sum())
            if n_after == 0:
                raise ValueError(
                    f"classifier preselection removes ALL '{split_name}' events."
                )
            classifier_presel_cutflow[split_name] = {"n_before": int(n_before), "n_after": n_after}
            log(f"  {split_name}: kept {n_after:,}/{n_before:,} "
                f"({100.0 * n_after / n_before:.1f}%) events")
            return mask

        mask_train = _mask_split(
            "train", {"mass": mass_train}, extra_train)
        mask_val = _mask_split(
            "val", {"mass": mass_val}, extra_val)
        mask_test = _mask_split(
            "test", {"mass": mass_test}, extra_test)

        X_train, y_train, rel_w_train, rel_w_premech2_train = (
            X_train[mask_train], y_train[mask_train], rel_w_train[mask_train], rel_w_premech2_train[mask_train])
        mass_train, proc_num_train, sample_name_train, sample_key_train = (
            mass_train[mask_train], proc_num_train[mask_train], sample_name_train[mask_train], sample_key_train[mask_train])
        train_rescale = train_rescale[mask_train]
        extra_train = {c: arr[mask_train] for c, arr in extra_train.items()}

        X_val, y_val, rel_w_val, rel_w_premech2_val = (
            X_val[mask_val], y_val[mask_val], rel_w_val[mask_val], rel_w_premech2_val[mask_val])
        mass_val, proc_num_val, sample_name_val, sample_key_val = (
            mass_val[mask_val], proc_num_val[mask_val], sample_name_val[mask_val], sample_key_val[mask_val])
        val_rescale = val_rescale[mask_val]
        extra_val = {c: arr[mask_val] for c, arr in extra_val.items()}

        X_test, y_test, rel_w_test, rel_w_premech2_test = (
            X_test[mask_test], y_test[mask_test], rel_w_test[mask_test], rel_w_premech2_test[mask_test])
        mass_test, proc_num_test, sample_name_test, sample_key_test = (
            mass_test[mask_test], proc_num_test[mask_test], sample_name_test[mask_test], sample_key_test[mask_test])
        test_rescale = test_rescale[mask_test]
        extra_test = {c: arr[mask_test] for c, arr in extra_test.items()}

        split_sizes["train_after_classifier_preselection"] = int(X_train.shape[0])
        split_sizes["val_after_classifier_preselection"] = int(X_val.shape[0])
        split_sizes["test_after_classifier_preselection"] = int(X_test.shape[0])

    weight_scheme_process = config["weight_scheme_process"]
    resolved_class_groups = config.get("class_groups") if class_groups == "auto" else class_groups
    log(f"\nApplying class_groups: {resolved_class_groups if resolved_class_groups else '(none)'}")

    log("Computing class weights ...")
    true_class_weights = class_weights_for_set(
        y_train, rel_w_train, proc_num_train, classes, weight_scheme_process, num_process_each_class,
        class_groups=resolved_class_groups,
    )
    class_weights_abs = class_weights_for_set(
        y_train, np.abs(rel_w_train), proc_num_train, classes, weight_scheme_process, num_process_each_class,
        class_groups=resolved_class_groups,
    )
    class_weights_only_positive = class_weights_for_set(
        y_train, rel_w_train * (rel_w_train > 0), proc_num_train, classes, weight_scheme_process, num_process_each_class,
        class_groups=resolved_class_groups,
    )
    class_weights_val = class_weights_for_set(
        y_val, rel_w_val, proc_num_val, classes, weight_scheme_process, num_process_each_class,
        class_groups=resolved_class_groups,
    )
    class_weights_test = class_weights_for_set(
        y_test, rel_w_test, proc_num_test, classes, weight_scheme_process, num_process_each_class,
        class_groups=resolved_class_groups,
    )

    class_weight_summaries = {
        "true_class_weights_train": print_class_summary(
            "true_class_weights (train)", y_train, true_class_weights, classes, verbose=verbose),
        "class_weights_abs_train": print_class_summary(
            "class_weights_for_training_abs (train)", y_train, class_weights_abs, classes, verbose=verbose),
        "class_weights_only_positive_train": print_class_summary(
            "class_weights_only_positive (train)", y_train, class_weights_only_positive, classes, verbose=verbose),
        "class_weights_val": print_class_summary(
            "class_weights_for_val (val)", y_val, class_weights_val, classes, verbose=verbose),
        "class_weights_test": print_class_summary(
            "class_weights_for_test (test)", y_test, class_weights_test, classes, verbose=verbose),
    }

    data_info = {
        "base_dir": str(base_dir),
        "seed": seed,
        "split_seed": split_seed,
        "fraction": fraction,
        "eras": list(eras),
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "classes": classes,
        "class_groups": resolved_class_groups,
        "weight_scheme_process": weight_scheme_process,
        "found_dirs": dataset_info["found_dirs"],
        "missing_dirs": dataset_info["missing_dirs"],
        "ignored_dirs": dataset_info["ignored_dirs"],
        "total_raw_rows": dataset_info["total_raw_rows"],
        "total_sampled_rows": dataset_info["total_sampled_rows"],
        "sample_info": dataset_info["sample_info"],
        "split_sizes": split_sizes,
        "mechanism2_rescale_factors": mechanism2_rescale_factors,
        "class_weight_summaries": class_weight_summaries,
        "apply_classifier_preselection": apply_classifier_preselection,
        "classifier_preselection_cutflow": classifier_presel_cutflow,
    }
    if info_out_path is not None:
        saved_path = save_data_info(data_info, info_out_path)
        log(f"\n[INFO] Wrote dataset info to '{saved_path}'")

    result = {
        "classes": classes,
        "X_train": X_train, "y_train": y_train, "rel_w_train": rel_w_train,
        "mass_train": mass_train, "proc_num_train": proc_num_train, "sample_name_train": sample_name_train,
        "sample_key_train": sample_key_train, "rel_w_premech2_train": rel_w_premech2_train,
        "X_val": X_val, "y_val": y_val, "rel_w_val": rel_w_val,
        "mass_val": mass_val, "proc_num_val": proc_num_val, "sample_name_val": sample_name_val,
        "sample_key_val": sample_key_val, "rel_w_premech2_val": rel_w_premech2_val,
        "X_test": X_test, "y_test": y_test, "rel_w_test": rel_w_test,
        "mass_test": mass_test, "proc_num_test": proc_num_test, "sample_name_test": sample_name_test,
        "sample_key_test": sample_key_test, "rel_w_premech2_test": rel_w_premech2_test,
        "true_class_weights": true_class_weights,
        "class_weights_abs": class_weights_abs,
        "class_weights_only_positive": class_weights_only_positive,
        "class_weights_val": class_weights_val,
        "class_weights_test": class_weights_test,
        "train_rescale_factors": train_rescale,
        "val_rescale_factors": val_rescale,
        "test_rescale_factors": test_rescale,
        "data_info": data_info,
    }
    for c in extra_columns:
        result[f"{c}_train"] = extra_train[c]
        result[f"{c}_val"] = extra_val[c]
        result[f"{c}_test"] = extra_test[c]
    return result

# --------------------------------------------------------------------------
# CLI (formerly the standalone prepare_inputs.py - see the module docstring
# for why the two files were merged). Samples MC inputs from
# individual_samples/ ONCE and writes the sampled/split .npy files + class
# weights + data_info.json to --output-dir, for callers that specifically
# want a cached, reusable copy of a dataset on disk (e.g. evaluate_model.py,
# which expects an --input-path directory of X_train.npy/X_val.npy/
# events.parquet), or to avoid re-reading individual_samples/ across many
# repeated experiments on the exact same --fraction/--seed.
#
# NOT required by end_to_end_training.py, which calls
# build_training_dataset() above directly, in-process, every run instead.
#
# Outputs (in --output-dir):
#     X_train.npy, X_val.npy, X_test.npy
#     y_train.npy, y_val.npy, y_test.npy
#     rel_w_train.npy, rel_w_val.npy, rel_w_test.npy
#     mass_train.npy, mass_val.npy, mass_test.npy
#     true_class_weights.npy               (train, original rel_w values)
#     class_weights_for_training_abs.npy   (train, abs(rel_w))
#     class_weights_only_positive.npy      (train, negative rel_w set to 0)
#     class_weights_for_val.npy            (val, original rel_w values, incl. negatives)
#     class_weights_for_test.npy           (test, original rel_w values, incl. negatives)
#     data_info.json                       (everything otherwise only printed to the
#                                            terminal: per era/sample sampling stats,
#                                            found/missing/ignored dirs, split sizes,
#                                            mechanism-2 rescale factors, class-weight
#                                            summaries - see build_training_dataset above)
#
# Usage:
#     python dataset_prep.py \
#         --base-dir /path/to/individual_samples \
#         --output-dir /path/to/output \
#         --config training_config_edit.yaml \
#         --fraction 0.1
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sample MC inputs from individual_samples/ and build training data + class weights."
    )
    parser.add_argument("--base-dir", type=Path, required=True,
                         help="Path to individual_samples/ (contains <era>/<sample>/{X.npy,rel_w.npy,events.parquet}).")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Directory to write the sampled/split .npy files to.")
    parser.add_argument("--config", type=Path, default="training_config_edit.yaml",
                         help="YAML with sample_to_class, classes, weight_scheme_process, "
                              "samples_info.eras. Default: training_config_edit.yaml")
    parser.add_argument("--fraction", "-f", type=float, required=True,
                         help="Fraction of rows to sample from each era/sample file, e.g. 0.1 for 10%.")
    parser.add_argument("--train-fraction", type=float, default=0.7,
                         help="Fraction of the sampled data used for training. Default: 0.7")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Fraction of the sampled data used for validation. "
                              "The remainder (1 - train - val) goes to the test set. Default: 0.15")
    parser.add_argument("--eras", type=str, nargs="+", default=None,
                         help="Era subdirectories to use (e.g. --eras preEE postEE). "
                              "Default: DEFAULT_ERAS above - a fixed, known-superset list of "
                              "every era name that has ever existed for this analysis, tried "
                              "unconditionally (missing eras/samples are skipped, not an error; "
                              "see DEFAULT_ERAS' docstring). Pass --auto-discover-eras instead "
                              "to fall back to the old behavior of auto-detecting whatever era "
                              "directories currently exist under --base-dir.")
    parser.add_argument("--auto-discover-eras", action="store_true",
                         help="Ignore DEFAULT_ERAS / --eras and auto-detect era directories "
                              "from --base-dir instead (discover_eras()). Only takes effect if "
                              "--eras is not also given.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Fraction-sampling seed (which random --fraction subset of each "
                              "era/sample file's rows is drawn). Also used for the train/val/test "
                              "split unless --split-seed is given separately. Default: 42")
    parser.add_argument("--split-seed", type=int, default=None,
                         help="Separate seed for the train/val/test split only, independent of "
                              "--seed's fraction-sampling. Defaults to --seed (old behavior: one "
                              "seed drives both) if not given.")
    parser.add_argument("--no-class-groups", action="store_true",
                         help="Ignore any 'class_groups' section in --config and normalize every "
                              "class independently (weight sum always 1.0 per class). By default, "
                              "if the config has a class_groups section, it IS applied.")
    parser.add_argument("--apply-classifier-preselection", action="store_true",
                         help="Apply the classifier (Stage A) preselection - classifier_"
                              "preselection_mask, EXACTLY prepare_inputs.py's preselection() - "
                              "to each of train/val/test before computing class weights. "
                              "Automatically also fetches the extra parquet columns it needs "
                              "(CLASSIFIER_PRESELECTION_COLUMNS). Off by default, matching this "
                              "CLI's previous (unpreselected) behavior. NOTE this is a DIFFERENT "
                              "cut from the 'gato model' (Stage B) preselection applied inside "
                              "combined_optimization.py.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.eras is not None:
        eras = args.eras
    elif args.auto_discover_eras:
        eras = discover_eras(args.base_dir)
    else:
        eras = list(DEFAULT_ERAS)
    print(f"Using eras: {eras}")

    result = build_training_dataset(
        args.base_dir, config, args.fraction, args.seed, eras,
        train_fraction=args.train_fraction, val_fraction=args.val_fraction,
        class_groups=(None if args.no_class_groups else "auto"),
        split_seed=args.split_seed,
        apply_classifier_preselection=args.apply_classifier_preselection,
        extra_columns=(CLASSIFIER_PRESELECTION_COLUMNS if args.apply_classifier_preselection else ()),
        verbose=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to '{args.output_dir}' ...")
    np.save(args.output_dir / "X_train.npy", result["X_train"])
    np.save(args.output_dir / "X_val.npy", result["X_val"])
    np.save(args.output_dir / "X_test.npy", result["X_test"])
    np.save(args.output_dir / "y_train.npy", result["y_train"])
    np.save(args.output_dir / "y_val.npy", result["y_val"])
    np.save(args.output_dir / "y_test.npy", result["y_test"])
    np.save(args.output_dir / "rel_w_train.npy", result["rel_w_train"])
    np.save(args.output_dir / "rel_w_val.npy", result["rel_w_val"])
    np.save(args.output_dir / "rel_w_test.npy", result["rel_w_test"])
    np.save(args.output_dir / "mass_train.npy", result["mass_train"])
    np.save(args.output_dir / "mass_val.npy", result["mass_val"])
    np.save(args.output_dir / "mass_test.npy", result["mass_test"])
    np.save(args.output_dir / "true_class_weights.npy", result["true_class_weights"])
    np.save(args.output_dir / "class_weights_for_training_abs.npy", result["class_weights_abs"])
    np.save(args.output_dir / "class_weights_only_positive.npy", result["class_weights_only_positive"])
    np.save(args.output_dir / "class_weights_for_val.npy", result["class_weights_val"])
    np.save(args.output_dir / "class_weights_for_test.npy", result["class_weights_test"])

    info_path = save_data_info(result["data_info"], args.output_dir / "data_info.json")
    print(f"Saved dataset info to '{info_path}'")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()