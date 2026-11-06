#!/usr/bin/env python3
"""
combined_optimization.py

Stage B only: end-to-end, fully differentiable classifier + GMM
categorization, fine-tuned from an ALREADY-PRETRAINED classifier (see
pretrain.py / pretrain_config.yaml for Stage A, which used to be the first
half of this same script).

    raw features X --[MLP classifier]--> logits --[softmax]--> scores (dim = n_classes)
    scores + weights + process labels --[gato_gmm_model]--> soft category yields S, B
    S, B --[asymptotic_significance]--> Z --> loss = -Z + lam_yield * low_bkg_penalty(B)

Everything above happens inside a single tf.GradientTape context, so both the
classifier's Dense-layer weights and the GMM's means/covariances/temperature
receive gradients from the same significance loss.

What is intentionally kept OUT of the differentiable graph (unchanged from
the original end_to_end_training.py):
    - event preselection / mass-window cuts
    - the per-event "rel_w" weight computation
    - the assignment of an event to "signal" / "background" via its (fixed,
      precomputed) class label. This uses tf.boolean_mask on a label array
      that does not depend on any trainable variable, so it does not block
      gradients for the *values* (scores) that pass through the mask -- it
      only changes which rows are visible to the loss, not how they were
      produced.

Code layout:
    training.py               - HOW one epoch/chunk is trained (train loop
                                 mechanics, memory profiling, dataset builders)
    end_to_end_helpers.py     - WHAT is trained and WHAT DATA it sees (data
                                 loading/shuffling, the combined classifier+GMM
                                 model, run-management/plotting utilities)
    pretrain.py                - Stage A: classifier-only pretraining.
    combined_optimization.py  - Stage B orchestration: this file. Parses
                                 args/config, loads the pretrained classifier,
                                 and fine-tunes classifier + GMM jointly.

combined_optimization_config.yaml's data_prep/class_weight_scheme/
preselection/seeds.data must match whatever pretrain_config.yaml used, so
this run reconstructs the exact train/val/test split the loaded classifier
was pretrained on - see build_raw_dataset_from_samples below. This now also
includes data_prep.apply_classifier_preselection (default true): the
classifier (Stage A) preselection - dataset_prep.classifier_preselection_
mask, a fixed mass/dijet-mass/mvaID window, EXACTLY prepare_inputs.py's
preselection() - is reconstructed here FIRST (same as pretrain.py), and
this script's own 'preselection' section (below) - a SEPARATE, broader,
fully config-driven cut (adds an is_boosted veto and, once a classifier is
loaded, a classifier-score cut) - composes ON TOP of it.

Example:
    python combined_optimization.py \\
        --base-dir /path/to/individual_samples \\
        --training-config-path /path/to/combined_optimization_config.yaml
"""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from gatohep.plotting_utils import (
    assign_bins_and_order,
    fill_histogram_from_assignments,
    plot_bias_history,
    plot_history,
    plot_stacked_histograms,
)
from gatohep.utils import asymptotic_significance

from mlp_tf import MLP

from training import (
    train_complete_model,
    _host_peak_rss_bytes,
    save_with_retry,
)

from end_to_end_helpers import (
    ConcatView,
    PermutedView,
    shuffle_raw_data,
    sequentialize_stage_b_split,
    compute_nonres_reweight_factors,
    build_raw_dataset_from_samples,
    log_gpu_memory,
    GMMFromScores,
    EndToEndClassifierGMM,
    make_plot_dataframe_from_model,
    get_unique_out_dir,
    build_descriptive_out_dir,
    write_run_summary,
    hist_values,
    evaluate_on_test_set,
    evaluate_all_splits,
    build_event_count_tables,
    build_static_preselection_cuts,
    static_preselection_columns,
    build_score_preselection_cut,
    compute_class_scores,
    compute_preselection_cutflow,
    write_preselection_cutflow_csv,
)

from dataset_prep import DEFAULT_ERAS, discover_eras, load_config as load_data_config, classifier_preselection_kwargs

from discriminant_density_diagnostics import save_test_set_histograms, DEFAULT_CHUNK_SIZE as DISC_DENSITY_DEFAULT_CHUNK_SIZE


def main():
    parser = argparse.ArgumentParser(description="Stage-B: fine-tune a pretrained classifier + GMM end-to-end")
    parser.add_argument("--base-dir", required=True,
                         help="Path to individual_samples/ (contains <era>/<sample>/{X.npy,rel_w.npy,events.parquet}).")
    parser.add_argument("--training-config-path", required=True,
                         help="YAML with seeds.*, pretrained_classifier_weights_path, data_prep.*, and "
                              "everything else Stage B needs - see combined_optimization_config.yaml.")
    parser.add_argument("--finetune-checkpoint-every", type=int, default=0,
                         help="Save a PAIRED classifier+gato checkpoint every N Stage-B epochs, under "
                              "out_dir/finetune_checkpoints/ (0 = disabled, default). Kept as a CLI flag "
                              "rather than a config key so the checkpoint cadence can be changed per-run "
                              "without editing the config (e.g. every 10 vs. every 20 epochs) - see "
                              "train_complete_model's finetune_checkpoint_every docstring in training.py "
                              "for what gets saved and why classifier+gato are always paired at the same epoch.")
    args = parser.parse_args()

    # ---------------- config ----------------
    with open(args.training_config_path, "r") as f:
        training_config = yaml.safe_load(f)

    seeds = training_config["seeds"]

    pretrained_classifier_weights_path = training_config.get("pretrained_classifier_weights_path")
    if not pretrained_classifier_weights_path:
        raise SystemExit(
            "combined_optimization_config.yaml is missing pretrained_classifier_weights_path - "
            "point it at the classifier_pretrained.weights.h5 written by pretrain.py, or set "
            "continue_training.enabled + continue_training.weights_path to resume a previous "
            "Stage-B run's classifier instead."
        )

    # -------------- data_prep config (same keys/semantics as pretrain_config.yaml -
    # MUST use the same values there, INCLUDING data_prep.apply_classifier_
    # preselection, so this run reconstructs the identical train/val/test
    # split the loaded classifier was pretrained on) --------------
    data_prep_cfg = training_config.get("data_prep", {}) or {}
    if "data_config_path" not in data_prep_cfg:
        raise SystemExit(
            "combined_optimization_config.yaml is missing data_prep.data_config_path - "
            "point it at your sample_to_class/classes/weight_scheme_process/"
            "samples_info config (e.g. training_config_edit.yaml)."
        )
    if "fraction" not in data_prep_cfg:
        raise SystemExit("combined_optimization_config.yaml is missing data_prep.fraction.")
    data_config_path = data_prep_cfg["data_config_path"]
    fraction = float(data_prep_cfg["fraction"])
    train_fraction = float(data_prep_cfg.get("train_fraction", 0.7))
    val_fraction = float(data_prep_cfg.get("val_fraction", 0.15))
    eras_cfg = data_prep_cfg.get("eras", None)
    use_class_groups = bool(data_prep_cfg.get("use_class_groups", True))
    random_search_dir = data_prep_cfg.get("random_search_dir", None)

    if not (0 < fraction <= 1):
        raise SystemExit("Error: data_prep.fraction must be between 0 and 1.")
    if not (0 < train_fraction < 1):
        raise SystemExit("Error: data_prep.train_fraction must be between 0 and 1.")
    if not (0 < val_fraction < 1):
        raise SystemExit("Error: data_prep.val_fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1:
        raise SystemExit(
            "Error: data_prep.train_fraction + data_prep.val_fraction must be < 1 "
            "(the remainder becomes the test fraction)."
        )

    plot_dim = int(training_config.get("plot_dim", training_config["gato_model"]["dim"]))


    # ---------------- reproducibility/cuda ----------------

    import random

    # Pin TF's thread pools to the number of CPUs HTCondor actually granted
    # this job (via cgroups), not the physical host's core count. Must run
    # BEFORE any other TF op - list_physical_devices()/set_seed() below
    # already count, so this has to come first, immediately after the TF
    # import at module load. os.sched_getaffinity(0) reads the cgroup CPU
    # affinity mask directly, so this tracks request_cpus in the .submit
    # file automatically without needing to hardcode/sync a number here.
    try:
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux fallback
        n_cpus = os.cpu_count() or 1
    tf.config.threading.set_intra_op_parallelism_threads(n_cpus)
    tf.config.threading.set_inter_op_parallelism_threads(n_cpus)
    print(f"[INFO] TF thread pools pinned to {n_cpus} CPUs (from cgroup affinity).")

    # seeds.shuffle covers general Python/NumPy randomness not explicitly
    # seeded elsewhere (data sampling/split use their own explicit seed -
    # see build_raw_dataset_from_samples below - and are unaffected by
    # this), and is also the global TF seed for the rest of the script
    # (dropout during training and any other TF-random op without an
    # explicit seed). seeds.classifier_init is passed explicitly to MLP()
    # below (has no visible effect here since the checkpoint overwrites
    # those weights right after, but kept for a consistent "model init"
    # story). seeds.gmm_init is used ONLY around GMMFromScores'
    # construction further down - gatohep's gato_gmm_model has no explicit
    # seed argument of its own, so that's done by temporarily swapping the
    # global TF seed to seeds.gmm_init right before constructing it, then
    # back to seeds.shuffle right after - see the GMMFromScores call below.
    random.seed(seeds["shuffle"])
    np.random.seed(seeds["shuffle"])
    tf.random.set_seed(seeds["shuffle"])
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # Set device
    gpus = tf.config.list_physical_devices("GPU")
    try:
        cuda_device = int(training_config["cuda_device"])
    except:
        cuda_device = -1

    if gpus:
        if cuda_device < 0 or cuda_device >= len(gpus):
            raise ValueError(
                f"Requested cuda_device={cuda_device}, but only {len(gpus)} GPU(s) available: {gpus}"
            )

        try:
            selected_gpu = gpus[cuda_device]

            # Make only the selected physical GPU visible to TensorFlow
            tf.config.set_visible_devices(selected_gpu, "GPU")

            # Optional but recommended: avoid TensorFlow reserving all memory at once
            tf.config.experimental.set_memory_growth(selected_gpu, True)

            # Important: after set_visible_devices, TensorFlow will see this as logical GPU:0
            logical_gpus = tf.config.list_logical_devices("GPU")

            device_name = "/GPU:0"
        

            print("\nINFO: Physical GPUs originally visible:")
            for i, gpu in enumerate(gpus):
                print(f"  physical index {i}: {gpu}")

            print(f"\nINFO: Requested physical GPU index: {cuda_device}")
            print(f"INFO: Selected physical GPU: {selected_gpu}")
            print(f"INFO: Logical GPUs after restriction: {logical_gpus}")
            print(f"INFO: Use TensorFlow device name: {device_name}\n")

        except RuntimeError as e:
            raise RuntimeError(
                "Could not set visible GPU devices. This usually means TensorFlow "
                "has already initialized the GPU before this code block ran."
            ) from e

    else:
        raise RuntimeError(
            "No CUDA-capable GPU was detected by TensorFlow (tf.config.list_physical_devices('GPU') "
            "returned empty), but this job requested cuda_device="
            f"{cuda_device}. This usually means HTCondor placed the job on a node where the GPU "
            "slot is not actually usable by CUDA (see cuInit/CUDA_ERROR_NO_DEVICE in the .err file) "
            "rather than a genuinely GPU-less machine. Refusing to silently fall back to CPU, since "
            "that would make a ~10h GPU run take many times longer without any visible warning. "
            "If you really want to allow CPU-only runs, set an explicit config flag for it instead "
            "of relying on this fallback."
        )


    # -------------- remaining config parameters --------------

    weight_scheme = training_config["class_weight_scheme"]

    # Every run gets a fresh out-dir, including when continue_training is
    # used to load weights/checkpoints from a PREVIOUS run - that previous
    # run's folder (continue_training.weights_path / gato_checkpoint_dir)
    # is only ever read from, never written into.
    #
    # The out-dir's last path component is replaced with a descriptive
    # slug encoding this run's settings (dim, data fraction, epochs,
    # n-cats, soft/hard class loss, pretrain/continue flags - see
    # build_run_name's docstring), so folders are self-explanatory without
    # opening training_config.yaml. Still collision-safe: two runs with
    # identical settings get "..._1", "..._2", ... via get_unique_out_dir
    # internally, exactly as the plain out-dir string used to.
    out_dir = build_descriptive_out_dir(training_config)
    print(f"[INFO] out-dir '{training_config['out-dir']}' -> using descriptive run folder '{out_dir}' for this run.")
    os.makedirs(out_dir, exist_ok=True)

    import datetime
    run_start_time = datetime.datetime.now().isoformat(timespec="seconds")

    plot_dtype = training_config["plot_dtype"]

    # ---------------- load data ----------------
    # Built directly from individual_samples/ every run - see
    # dataset_prep.build_training_dataset for the fraction-sampling
    # (mechanism 1) and train/val/test-split (mechanism 2) rescaling, and
    # end_to_end_helpers.build_raw_dataset_from_samples for how it's adapted
    # into the raw_data/class_weights/mass shapes the rest of this script
    # expects. mass_train_arr/mass_val_arr are now ALWAYS available (not
    # just when preselection/mass_window_weighting is enabled), since
    # nonres_reweight needs them unconditionally whenever it's enabled.
    # The test split (X_test/y_test/rel_w_test in raw_data, cw_test in
    # class_weights, mass_test_arr here) is never trained/selected on - it
    # is held out purely for evaluate_on_test_set below.
    # eras: data_prep.eras in training_config.yaml, defaulting (when
    # null/omitted) to dataset_prep.DEFAULT_ERAS - a FIXED list of every era
    # name that has ever existed on disk for this analysis (see that
    # constant's docstring, and gato_categorization_merged.py's load_samples,
    # which uses the same fixed list for the same reason). This is
    # deliberately NOT auto-detected via discover_eras(args.base_dir)
    # anymore: auto-detection makes the dataset silently depend on whatever
    # era directories happen to exist under --base-dir *right now*, so the
    # same training_config.yaml can quietly pick up (or drop) an era if
    # base-dir's contents change between runs. Trying every name in a fixed
    # list is safe either way - dataset_prep.build_dataset already skips any
    # era (or era/sample) directory that isn't present on disk, logging it
    # under "missing_dirs" in data_info.json, rather than failing - so an
    # era that doesn't exist yet (e.g. "2025") is simply skipped, not an
    # error. discover_eras() is still available and used automatically if
    # data_prep.eras is explicitly set to something other than null.
    data_config = load_data_config(data_config_path)
    eras = eras_cfg if eras_cfg is not None else DEFAULT_ERAS
    print(f"[INFO] Building raw dataset from '{args.base_dir}' (eras={eras})")
    presel_cfg = training_config.get("preselection", {}) or {}
    # classifier (Stage A) preselection kwargs - MUST resolve the same way
    # pretrain.py's own call did (data_prep.apply_classifier_preselection),
    # or this reconstructs a different (larger) train/val/test split than
    # the one the loaded classifier was actually pretrained on - see the
    # module docstring and the data_prep_cfg comment above.
    classifier_presel_kwargs = classifier_preselection_kwargs(data_prep_cfg)
    print(f"[INFO] Classifier (Stage A) preselection (reconstructing the pretrained "
          f"split): {'ENABLED' if classifier_presel_kwargs['apply_classifier_preselection'] else 'disabled'}")
    # Union of the columns Stage A's own (fixed) preselection needs and
    # whatever this run's own 'preselection' section (static_preselection_
    # columns) needs - deduplicated, since both use the same column names
    # for dijet_mass/lead_mvaID/sublead_mvaID (see extra_column_logical_name
    # below). dict.fromkeys(...) preserves order while dropping duplicates.
    extra_columns = tuple(dict.fromkeys(
        classifier_presel_kwargs["extra_columns"] + static_preselection_columns(presel_cfg)
    ))
    # Maps a parquet column name back to the logical key build_static_
    # preselection_cuts expects - used below for both train/val (PermutedView)
    # and test (plain fancy indexing).
    extra_column_logical_name = {
        "nonResReg_vbfpair_dijet_mass_DNNreg": "dijet_mass",
        "lead_mvaID": "lead_mvaid",
        "sublead_mvaID": "sublead_mvaid",
        "is_boosted": "is_boosted",
    }
    raw_data, class_weights, mass_train_arr, mass_val_arr, mass_test_arr, classes, split_rescale_factors = build_raw_dataset_from_samples(
        args.base_dir, data_config, fraction, seeds["data"], eras,
        train_fraction=train_fraction, val_fraction=val_fraction,
        weight_scheme=weight_scheme,
        class_groups=("auto" if use_class_groups else None),
        info_out_path=os.path.join(out_dir, "data_info.json"),
        extra_columns=extra_columns,
        apply_classifier_preselection=classifier_presel_kwargs["apply_classifier_preselection"],
        # Must match the pretrained-checkpoint's own seeds.data_split
        # EXACTLY (same rule as seeds.data - see this config's header
        # comment) or this reconstructs a DIFFERENT train/val/test split
        # than the one the classifier was actually pretrained on, even
        # with seeds.data matching. Falls back to seeds["data"] (old,
        # single-seed behavior) if seeds.data_split isn't set at all.
        split_seed=seeds.get("data_split"),
    )

    # Shuffle all datasets (X, y, rel_w, class_weights) once, using one
    # fixed permutation per split. This MUST happen before the mass-window
    # block below (which, for any mode other than "off", builds stageB_data
    # directly from raw_data/class_weights) - otherwise stageB_data would
    # capture the pre-shuffle arrays and silently train Stage B on
    # unshuffled data whenever mass_window_weighting.mode != "off". This is
    # a zero-copy reorder (see PermutedView) so it stays cheap even for
    # large memmapped feature matrices, and is likewise safe to run before
    # Stage A (which does its own per-batch shuffling internally anyway).
    raw_data, class_weights, train_perm, val_perm = shuffle_raw_data(raw_data, class_weights, seeds["shuffle"])

    # mass_train_arr/mass_val_arr were built in the ORIGINAL (pre-shuffle)
    # row order alongside X/y/rel_w; apply the same permutation here so they
    # stay row-aligned with the now-shuffled raw_data.
    mass_train_arr = np.asarray(mass_train_arr)[train_perm]
    mass_val_arr = np.asarray(mass_val_arr)[val_perm]
    raw_data["rel_w_premech2_train"] = raw_data["rel_w_premech2_train"][train_perm]
    raw_data["rel_w_premech2_val"] = raw_data["rel_w_premech2_val"][val_perm]
    for c in extra_columns:
        raw_data[f"{c}_train"] = raw_data[f"{c}_train"][train_perm]
        raw_data[f"{c}_val"] = raw_data[f"{c}_val"][val_perm]

    # ----------------------------------------------------------------------
    # PRESELECTION: rough, NON-differentiable hard cuts (mass window, and -
    # if their own switches are on - dijet-mass window, lead/sublead mvaID
    # thresholds, is_boosted veto; the score-based cut needs a classifier,
    # so it runs separately, further below, right after one is loaded). This
    # mirrors gato_categorization_vbf_only_ggh.py's preselection(): a broad
    # acceptance region defining which events are part of the analysis at
    # all. It is deliberately NOT differentiable and NOT trainable - there
    # is nothing to optimize about it, and no gradient should ever move it.
    #
    # Because these are genuine hard cuts, rejected events are DROPPED (via
    # zero-copy index views for train/val) rather than merely given weight
    # 0. Dropping them is strictly better than zero-weighting: zero-weighted
    # events would still cost a full classifier forward/backward pass every
    # epoch, and - more importantly - would still carry a non-zero class
    # weight `cw` and therefore still train the classifier through the
    # cross-entropy term, even though they are outside the analysis region.
    #
    # Each individual cut's removal count (a standard HEP cutflow) is
    # written to out_dir/preselection_cutflow.csv - see
    # compute_preselection_cutflow/write_preselection_cutflow_csv.
    cutflow_csv_path = os.path.join(out_dir, "preselection_cutflow.csv")
    any_static_cutflow = False
    for split_name, mass_arr in (("train", mass_train_arr), ("val", mass_val_arr)):
        arrays = {"mass": mass_arr}
        for c in extra_columns:
            arrays[extra_column_logical_name[c]] = raw_data[f"{c}_{split_name}"]
        cuts = build_static_preselection_cuts(presel_cfg, arrays)
        if not cuts:
            continue
        n_before_split = mass_arr.shape[0]
        keep_idx, cutflow_rows = compute_preselection_cutflow(
            cuts, n_rows=n_before_split, split_name=split_name,
        )
        if keep_idx.size == 0:
            raise ValueError(f"preselection removes ALL '{split_name}' events - see {cutflow_csv_path}.")
        write_preselection_cutflow_csv(cutflow_csv_path, cutflow_rows)
        any_static_cutflow = True

        raw_data[f"X_{split_name}"] = PermutedView(raw_data[f"X_{split_name}"], keep_idx)
        raw_data[f"y_{split_name}"] = PermutedView(raw_data[f"y_{split_name}"], keep_idx)
        raw_data[f"rel_w_{split_name}"] = PermutedView(raw_data[f"rel_w_{split_name}"], keep_idx)
        class_weights[f"cw_{split_name}"] = PermutedView(class_weights[f"cw_{split_name}"], keep_idx)
        raw_data[f"rel_w_premech2_{split_name}"] = raw_data[f"rel_w_premech2_{split_name}"][keep_idx]
        if split_name == "train":
            class_weights["cw_train_no_abs"] = PermutedView(class_weights["cw_train_no_abs"], keep_idx)
            mass_train_arr = mass_train_arr[keep_idx]
        else:
            mass_val_arr = mass_val_arr[keep_idx]
        for c in extra_columns:
            raw_data[f"{c}_{split_name}"] = raw_data[f"{c}_{split_name}"][keep_idx]
    if any_static_cutflow:
        print(f"[INFO] Saved static preselection cutflow to: {cutflow_csv_path}")
    else:
        print(
            "[INFO] Preselection disabled (preselection.enabled=false) - assuming the "
            "rough mass acceptance cut was already applied upstream (individual_samples/)."
        )

    # Test split gets the identical static cuts, applied separately (it
    # isn't part of the train/val PermutedView bookkeeping above since
    # Stage A/B never see it) - see evaluate_on_test_set below. Plain fancy
    # indexing (not PermutedView) since this is a one-off pass, not a
    # per-epoch operation - the extra copy is negligible here.
    test_arrays = {"mass": mass_test_arr}
    for c in extra_columns:
        test_arrays[extra_column_logical_name[c]] = raw_data[f"{c}_test"]
    test_cuts = build_static_preselection_cuts(presel_cfg, test_arrays)
    if test_cuts:
        keep_test, test_cutflow_rows = compute_preselection_cutflow(
            test_cuts, n_rows=mass_test_arr.shape[0], split_name="test",
        )
        if keep_test.size == 0:
            raise ValueError(f"preselection removes ALL 'test' events - see {cutflow_csv_path}.")
        write_preselection_cutflow_csv(cutflow_csv_path, test_cutflow_rows)
        mass_test_arr = mass_test_arr[keep_test]
        raw_data["X_test"] = np.asarray(raw_data["X_test"])[keep_test]
        raw_data["y_test"] = np.asarray(raw_data["y_test"])[keep_test]
        raw_data["rel_w_test"] = np.asarray(raw_data["rel_w_test"])[keep_test]
        class_weights["cw_test"] = np.asarray(class_weights["cw_test"])[keep_test]
        raw_data["rel_w_premech2_test"] = raw_data["rel_w_premech2_test"][keep_test]
        for c in extra_columns:
            raw_data[f"{c}_test"] = raw_data[f"{c}_test"][keep_test]

    log_gpu_memory("after data loading")

    input_size = raw_data["X_train"].shape[1]
    # `[:]` goes through PermutedView's vectorized slice path; np.unique on the
    # view object itself would fall back to a slow element-by-element read.
    output_size = len(np.unique(np.asarray(raw_data["y_train"][:])))
    print(f"[INFO] input_size={input_size}, output_size (n_classes)={output_size}")

    # architecture: reuse random-search best_params.json if present, else config defaults
    best_params_path = (
        os.path.join(random_search_dir, "random_search_1", "best_params.json")
        if random_search_dir else None
    )
    if best_params_path and os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        num_layers = best_params["num_layers"]
        num_nodes = best_params["num_nodes"]
        act_fn_name = best_params["act_fn_name"]
        dropout_prob = best_params["dropout_prob"]
        print(f"[INFO] Using architecture from {best_params_path}: {best_params}")
    else:
        # get parameters from config:
        num_layers, num_nodes, act_fn_name, dropout_prob = training_config["class_model"].values()

    # -------------- classifier weights: load, never train from scratch here --------------
    # continue_training (optional) resumes a PREVIOUS Stage-B run's classifier
    # (and, if gato_checkpoint_dir is set, its GMM checkpoint too) instead of
    # starting fresh from the Stage-A checkpoint. Either way, Stage A itself
    # is never run by this script - see pretrain.py.
    continue_cfg = training_config.get("continue_training", {}) or {}
    continue_training_enabled = continue_cfg.get("enabled", False)
    continue_gato_checkpoint_dir = continue_cfg.get("gato_checkpoint_dir", None)
    continue_finetuning = continue_training_enabled and continue_cfg.get("continue_finetuning", False)
    finetune_suffix = ""

    classifier_weights_path = (
        continue_cfg.get("weights_path") if continue_training_enabled else None
    ) or pretrained_classifier_weights_path
    if not os.path.exists(classifier_weights_path):
        raise FileNotFoundError(f"classifier weights file does not exist: {classifier_weights_path}")

    with tf.device(device_name):
        classifier = MLP(input_size, num_layers, num_nodes, output_size, act_fn_name, dropout_prob,
                          seed=seeds["classifier_init"])
        # build the model's variables before loading weights into it
        classifier(tf.zeros((1, input_size), dtype=tf.float32), training=False)
        classifier.load_weights(classifier_weights_path)
        print(f"[INFO] Loaded classifier weights from {classifier_weights_path}")

    # ----------------------------------------------------------------------
    # SCORE-BASED PRESELECTION: gato_categorization_vbf_only_ggh.py's
    # `scores[:, 3] < 0.774` cut - only possible here, once a classifier
    # actually exists (never in pretrain.py). Uses the plain classifier
    # softmax directly (compute_class_scores), not EndToEndClassifierGMM -
    # the GMM categorizer isn't built until further below, and this cut is
    # meant to shape what Stage B ever trains/fine-tunes on in the first
    # place. Composes on top of the static preselection views above (same
    # PermutedView machinery for train/val; plain fancy indexing for test).
    # Appended to the SAME preselection_cutflow.csv as the static cuts.
    # ----------------------------------------------------------------------
    score_chunk_size = training_config.get("training", {}).get("chunk_size")
    score_cut_cfg = presel_cfg.get("score_cut", {}) or {}
    if bool(score_cut_cfg.get("enabled", False)):
        with tf.device(device_name):
            for split_name in ("train", "val"):
                X_split = raw_data[f"X_{split_name}"]
                class_scores = compute_class_scores(
                    classifier, X_split, int(score_cut_cfg.get("class_index", 0)),
                    chunk_size=score_chunk_size,
                )
                score_cut = build_score_preselection_cut(presel_cfg, class_scores)
                keep_idx, cutflow_rows = compute_preselection_cutflow(
                    [score_cut], n_rows=class_scores.shape[0], split_name=split_name,
                )
                if keep_idx.size == 0:
                    raise ValueError(f"score-based preselection removes ALL '{split_name}' events.")
                write_preselection_cutflow_csv(cutflow_csv_path, cutflow_rows)

                raw_data[f"X_{split_name}"] = PermutedView(raw_data[f"X_{split_name}"], keep_idx)
                raw_data[f"y_{split_name}"] = PermutedView(raw_data[f"y_{split_name}"], keep_idx)
                raw_data[f"rel_w_{split_name}"] = PermutedView(raw_data[f"rel_w_{split_name}"], keep_idx)
                class_weights[f"cw_{split_name}"] = PermutedView(class_weights[f"cw_{split_name}"], keep_idx)
                raw_data[f"rel_w_premech2_{split_name}"] = raw_data[f"rel_w_premech2_{split_name}"][keep_idx]
                for c in extra_columns:
                    raw_data[f"{c}_{split_name}"] = raw_data[f"{c}_{split_name}"][keep_idx]
                if split_name == "train":
                    class_weights["cw_train_no_abs"] = PermutedView(class_weights["cw_train_no_abs"], keep_idx)
                    mass_train_arr = mass_train_arr[keep_idx]
                else:
                    mass_val_arr = mass_val_arr[keep_idx]

            class_scores_test = compute_class_scores(
                classifier, raw_data["X_test"], int(score_cut_cfg.get("class_index", 0)),
                chunk_size=score_chunk_size,
            )
            score_cut_test = build_score_preselection_cut(presel_cfg, class_scores_test)
            keep_test, test_cutflow_rows = compute_preselection_cutflow(
                [score_cut_test], n_rows=class_scores_test.shape[0], split_name="test",
            )
            if keep_test.size == 0:
                raise ValueError("score-based preselection removes ALL 'test' events.")
            write_preselection_cutflow_csv(cutflow_csv_path, test_cutflow_rows)
            mass_test_arr = mass_test_arr[keep_test]
            raw_data["X_test"] = np.asarray(raw_data["X_test"])[keep_test]
            raw_data["y_test"] = np.asarray(raw_data["y_test"])[keep_test]
            raw_data["rel_w_test"] = np.asarray(raw_data["rel_w_test"])[keep_test]
            class_weights["cw_test"] = np.asarray(class_weights["cw_test"])[keep_test]
            raw_data["rel_w_premech2_test"] = raw_data["rel_w_premech2_test"][keep_test]
            for c in extra_columns:
                raw_data[f"{c}_test"] = raw_data[f"{c}_test"][keep_test]
        print(f"[INFO] Saved score-based preselection cutflow to: {cutflow_csv_path}")

    # ----------------------------------------------------------------------
    # m_gammagamma WINDOW: fixed soft window used only inside Stage B
    # ----------------------------------------------------------------------
    # Deliberately built HERE, after every preselection step (static AND
    # score-based) has already reduced raw_data/class_weights/mass_train_arr/
    # mass_val_arr - stageB_data below copies straight out of those, so it
    # would otherwise silently capture the PRE-score-cut arrays if this
    # block still sat before the classifier/score-cut section, wasting the
    # PermutedView/index-cut work above and letting rows the score cut just
    # rejected quietly reappear in Stage B's actual training data.
    mass_window_cfg = dict(training_config.get("training", {}).get("mass_window_weighting", {}) or {})
    mass_window_enabled = bool(mass_window_cfg.get("enabled", True))
    stageB_data = None

    if mass_window_enabled:
        eval_cfg_for_window = training_config.get("evaluation", {}) or {}
        default_low, default_high = eval_cfg_for_window.get("mass_window", [None, None])
        if mass_window_cfg.get("low") is None:
            mass_window_cfg["low"] = default_low
        if mass_window_cfg.get("high") is None:
            mass_window_cfg["high"] = default_high
        if mass_window_cfg["low"] is None or mass_window_cfg["high"] is None:
            raise ValueError(
                "mass_window_weighting is enabled, but low/high are not set."
            )

        mw_low = mass_window_cfg["low"]
        mw_high = mass_window_cfg["high"]
        mw_softness = mass_window_cfg.get("softness", 1.5)

        mass_train, mass_val = mass_train_arr, mass_val_arr

        stageB_data = {
            "X_train": raw_data["X_train"], "X_val": raw_data["X_val"],
            "y_train": raw_data["y_train"], "y_val": raw_data["y_val"],
            "rel_w_train": np.array(raw_data["rel_w_train"][:], dtype=np.float32, copy=True),
            "rel_w_val": np.array(raw_data["rel_w_val"][:], dtype=np.float32, copy=True),
            "mass_train": mass_train, "mass_val": mass_val,
            "cw_train": np.array(class_weights["cw_train"][:], copy=True),
            "cw_val": np.array(class_weights["cw_val"][:], copy=True),
            "cw_train_no_abs": np.array(class_weights["cw_train_no_abs"][:], copy=True),
        }

        print(
            f"[INFO] Stage-B mass window enabled: soft window [{mw_low}, {mw_high}] "
            f"(softness={mw_softness}) will be applied between classifier and GMM."
        )

    # NOTE: raw_data/class_weights (and mass_train/mass_val, and stageB_data
    # if mass_window_enabled) were already shuffled once, consistently,
    # right after data loading - see the shuffle_raw_data() call above,
    # before the mass-window block.

    # ======================================================================
    # STAGE B: joint fine-tuning of classifier + GMM with the Asimov
    #          significance loss. Trains on the train split and monitors on
    #          the val split (same split used by Stage A), instead of
    #          training on the combined train+val set as before.
    # ======================================================================

    print("[STAGE B] Building GMM categorizer and fine-tuning end-to-end\n")

    # load gato model config. NOTE: this used to unpack training_config
    # ["gato_model"].values() POSITIONALLY (relying on the yaml's key
    # order) - switched to explicit key access here since we're now adding
    # a new key (drop_last_softmax_class) to that same dict, and a
    # positional unpack would either break outright (wrong tuple length)
    # or, worse, silently bind the wrong config value to the wrong
    # variable if the new key ever ended up in the "wrong" position.
    gato_cfg = training_config["gato_model"]
    n_cats = gato_cfg["n-cats"]
    signal_class = gato_cfg["signal_class"]
    gato_dim = gato_cfg["dim"]
    temperature_init = gato_cfg["temperature_init"]
    temperature_final = gato_cfg["temperature_final"]

    # drop_last_softmax_class (default False): softmax over output_size
    # classes always sums to 1, so its last column is redundant - see
    # EndToEndClassifierGMM's drop_last_class docstring in
    # end_to_end_helpers.py for the full rationale and the differentiability
    # check backing this up (autodiff vs. finite differences, matched to
    # ~1e-10 - dropping the column does NOT block gradients from reaching
    # the classifier's corresponding output-layer weights, because the
    # softmax's shared normalization still ties every logit together).
    # When enabled, gato_model.dim MUST be output_size - 1 (the GMM then
    # only ever sees the (output_size-1)-dimensional embedding
    # EndToEndClassifierGMM.scores() actually hands it) - checked here so a
    # mismatched config fails fast with a clear message instead of a
    # confusing shape error deep inside gato's own code.
    drop_last_class = bool(gato_cfg.get("drop_last_softmax_class", False))
    if drop_last_class and gato_dim != output_size - 1:
        raise ValueError(
            f"gato_model.drop_last_softmax_class is true, so gato_model.dim must be "
            f"output_size - 1 = {output_size - 1} (output_size={output_size}, from the "
            f"number of distinct classes in y_train), but gato_model.dim={gato_dim} in "
            f"training_config.yaml. Set dim: {output_size - 1}, or turn "
            f"drop_last_softmax_class off."
        )
    elif (not drop_last_class) and gato_dim != output_size:
        print(
            f"[WARNING] gato_model.dim={gato_dim} does not match output_size={output_size} "
            f"(the number of classes) and drop_last_softmax_class is not enabled - this was "
            f"already allowed before (e.g. plot_dim can differ from output_size for other "
            f"reasons), but double-check this is intentional."
        )

    # gato_gmm_model draws its initial mixture_logits/means from tf.random.*
    # with no seed argument of its own, so it can only be given its own
    # independent seed by temporarily pointing the global TF seed at
    # seeds.gmm_init for exactly this construction call, then restoring the
    # general one right after - see the seeds block near the top of main().
    tf.random.set_seed(seeds["gmm_init"])
    gato = GMMFromScores(n_cats=n_cats, dim=gato_dim, temperature=temperature_init)
    tf.random.set_seed(seeds["shuffle"])

    if continue_training_enabled and continue_gato_checkpoint_dir:
        print(f"[INFO] continue_training enabled: restoring GMM categorizer from {continue_gato_checkpoint_dir}")
        gato.restore(continue_gato_checkpoint_dir)

    # Add soft mass window cut if enabled
    model_mass_cfg = None
    if mass_window_enabled:
        model_mass_cfg = {
            "enabled": True,
            "init_low": mass_window_cfg["low"],
            "init_high": mass_window_cfg["high"],
            "softness": mass_window_cfg.get("softness", 1.5),
            # fixed placement: always before GMM
        }
    model = EndToEndClassifierGMM(
        classifier, gato, signal_class=signal_class,
        mass_window_cfg=model_mass_cfg,
        nonres_reweight_cfg=training_config.get("nonres_reweight"),
        drop_last_class=drop_last_class,
    )

    # Stage-B data: raw_data's regular X_train/X_val split either way
    # (mass_window_weighting.enabled=False uses raw_data directly; enabled
    # uses stageB_data, which wraps the SAME X_train/X_val/y_train/y_val
    # plus mass_train.npy/mass_val.npy and its own rel_w - see above). The
    # mass arrays are threaded through to the model, which applies the soft
    # window internally (and, by default, learns its edges) - see
    # EndToEndClassifierGMM/model_mass_cfg above.
    if mass_window_enabled:
        X_train_stageB, y_train_stageB, w_train_stageB = (
            stageB_data["X_train"], stageB_data["y_train"], stageB_data["rel_w_train"]
        )
        X_val_stageB, y_val_stageB, w_val_stageB = (
            stageB_data["X_val"], stageB_data["y_val"], stageB_data["rel_w_val"]
        )
        # The mass window is always applied INSIDE the model (see
        # EndToEndClassifierGMM/model_mass_cfg below) whenever it's enabled,
        # so the real per-event mass arrays must be threaded through here -
        # passing None would make EndToEndClassifierGMM.mass_window_factor()
        # silently skip the window entirely (see its "if mass is None"
        # check), defeating mass_window_weighting.enabled without any error.
        mass_train_stageB, mass_val_stageB = stageB_data["mass_train"], stageB_data["mass_val"]

        # Same source as X_train_stageB/rel_w_train above (stageB_data,
        # built from the already-shuffled raw_data/class_weights - see the
        # shuffle_raw_data() call earlier in main()), so cw_train_stageB
        # stays row-aligned with X_train_stageB/y_train_stageB here.
        cw_train_stageB, cw_val_stageB = stageB_data["cw_train"], stageB_data["cw_val"]
        cw_train_no_abs_stageB = stageB_data["cw_train_no_abs"]
    else:
        X_train_stageB, y_train_stageB, w_train_stageB = (
            raw_data["X_train"], raw_data["y_train"], raw_data["rel_w_train"]
        )
        X_val_stageB, y_val_stageB, w_val_stageB = (
            raw_data["X_val"], raw_data["y_val"], raw_data["rel_w_val"]
        )
        mass_train_stageB, mass_val_stageB = None, None
        cw_train_stageB, cw_val_stageB = class_weights["cw_train"], class_weights["cw_val"]
        cw_train_no_abs_stageB = class_weights["cw_train_no_abs"]

    # ----------------------------------------------------------------------
    # I/O: put the Stage-B splits back into on-disk row order
    # ----------------------------------------------------------------------
    # Stage A is already done at this point and did its own per-batch
    # shuffling, so the random permutation applied at load time has no
    # remaining purpose - while it does turn every Stage-B chunk read into a
    # scattered gather across the whole feature file. Stage B's loss is a sum
    # over all events, so re-ordering rows changes nothing numerically. See
    # sequentialize_stage_b_split() for the full argument.
    #
    # Every array of a split is reordered with the SAME permutation, so the
    # X/y/rel_w/cw/mass row alignment (and hence the alignment of X_all and
    # y_plot/w_plot used for the final histograms below) is preserved.
    sequential_rows = bool(
        (training_config.get("training", {}) or {}).get("sequential_row_order", True)
    )
    X_train_stageB, (y_train_stageB, w_train_stageB, cw_train_stageB, mass_train_stageB, cw_train_no_abs_stageB) = (
        sequentialize_stage_b_split(
            X_train_stageB,
            (y_train_stageB, w_train_stageB, cw_train_stageB, mass_train_stageB, cw_train_no_abs_stageB),
            split_name="train", enabled=sequential_rows,
        )
    )
    X_val_stageB, (y_val_stageB, w_val_stageB, cw_val_stageB, mass_val_stageB) = (
        sequentialize_stage_b_split(
            X_val_stageB,
            (y_val_stageB, w_val_stageB, cw_val_stageB, mass_val_stageB),
            split_name="val", enabled=sequential_rows,
        )
    )

    log_gpu_memory("before Stage B")

    # Stage B now trains on the train split only and monitors on the val
    # split (mirroring Stage A), instead of training on combined train+val.
    (
        loss_history, penalty_history, mean_bias_history, bias_epochs,
        temperature_history, val_loss_history,
        ce_history, val_ce_history,
        ce_no_abs_history,
        unc_penalty_history, val_unc_penalty_history, cov_penalty_history,
    ) = train_complete_model(
        model,
        X_train_stageB, y_train_stageB, w_train_stageB,
        training_config,
        X_val=X_val_stageB, y_val=y_val_stageB, w_val=w_val_stageB,
        mass_train=mass_train_stageB, mass_val=mass_val_stageB,
        cw_train=cw_train_stageB, cw_val=cw_val_stageB,
        cw_train_no_abs=cw_train_no_abs_stageB,
        # Needed to build the collapsed signal-vs-background CE
        # (training.ce_mode) - the classifier still has `output_size` outputs,
        # but the CE groups all non-signal classes into one block.
        n_classes=output_size,
        # Per-epoch wall-clock + memory profile, written next to the other run
        # outputs. compare_profiles.py diffs two of these directly.
        profile_out_path=os.path.join(out_dir, "profile_stageB.json"),
        nonres_reweight_fn=compute_nonres_reweight_factors,
        out_dir=out_dir,
        classes=classes,
        finetune_checkpoint_every=args.finetune_checkpoint_every,
    )
    log_gpu_memory("after Stage B")

    # ---------------- save training histories to CSV ----------------
    # Everything below was previously only printed to the terminal (the
    # per-epoch log_line) and plotted as PNGs; the underlying numbers were
    # never written anywhere the user could reload/replot them later.
    #
    # loss_history / val_loss_history / ce_history / val_ce_history /
    # penalty_history are all recorded once per epoch (see the
    # `for ep in range(n_epochs)` loop in train_complete_model), so they
    # share one row-per-epoch CSV. mean_bias_history / temperature_history
    # are only recorded every `diagnostics_every` epochs (plus the final
    # epoch), so they get their own CSV indexed by bias_epochs.
    history_csv_path = os.path.join(out_dir, f"stageB_history{finetune_suffix}.csv")
    with open(history_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_sig_loss", "val_sig_loss", "train_ce", "val_ce", "train_ce_no_abs",
             "penalty", "unc_penalty", "val_unc_penalty", "cov_penalty"]
        )
        for i in range(len(loss_history)):
            writer.writerow([
                i,
                loss_history[i],
                val_loss_history[i] if i < len(val_loss_history) else "",
                ce_history[i] if i < len(ce_history) else "",
                val_ce_history[i] if i < len(val_ce_history) else "",
                ce_no_abs_history[i] if i < len(ce_no_abs_history) else "",
                penalty_history[i] if i < len(penalty_history) else "",
                unc_penalty_history[i] if i < len(unc_penalty_history) else "",
                val_unc_penalty_history[i] if i < len(val_unc_penalty_history) else "",
                cov_penalty_history[i] if i < len(cov_penalty_history) else "",
            ])
    print(f"[INFO] Saved Stage-B training history to: {history_csv_path}")

    diagnostics_csv_path = os.path.join(out_dir, f"stageB_diagnostics{finetune_suffix}.csv")
    with open(diagnostics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mean_bias", "temperature"])
        for ep_i, bias_v, temp_v in zip(bias_epochs, mean_bias_history, temperature_history):
            writer.writerow([ep_i, bias_v, temp_v])
    print(f"[INFO] Saved Stage-B bias/temperature diagnostics to: {diagnostics_csv_path}")

    # Zero-copy view over train+val, used below only for the final plots /
    # histograms, which should reflect the full available statistics. Uses
    # whichever split Stage B actually trained on (see above).
    X_all = ConcatView(X_train_stageB, X_val_stageB)


    # ---------------- save outputs ----------------
    if continue_finetuning:
        print("[INFO] continue_finetuning enabled: continuing Stage-B optimization of the restored GMM checkpoint.\n")

    save_with_retry(
        classifier.save_weights, os.path.join(out_dir, f"classifier_finetuned{finetune_suffix}.weights.h5"),
        label="final Stage B classifier weights",
    )
    checkpoint_dir = os.path.join(out_dir, f"checkpoint{finetune_suffix}")
    save_with_retry(gato.save, checkpoint_dir, ensure_dir=checkpoint_dir, label="final Stage B gato checkpoint")

    # Loss history (train + val in one plot, mirroring the Stage A plot above),
    # if a validation split was used in Stage B. Falls back to a train-only
    # plot via plot_history if no val history is available.
    loss_plot = os.path.join(out_dir, f"loss_history{finetune_suffix}.{plot_dtype}")
    if val_loss_history:
        plt.figure(figsize=(8, 6))
        plt.plot(loss_history, label="train")
        plt.plot(val_loss_history, label="validation")
        plt.xlabel("epochs")
        plt.ylabel("Asimov loss (-Z)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_plot)
        plt.close()
    else:
        plot_history(np.array(loss_history), loss_plot, x_label="Epoch", y_label="Asimov loss (-Z)")

    # Cross-entropy history: the classifier-only half of the split loss. Worth
    # its own plot now that the CE no longer enters the GMM's objective - it
    # tells you whether the classifier is drifting while Z improves.
    ce_plot = None
    if any(np.isfinite(v) for v in ce_history):
        ce_plot = os.path.join(out_dir, f"ce_history{finetune_suffix}.{plot_dtype}")
        plt.figure(figsize=(8, 6))
        plt.plot(ce_history, label="train (abs weights)")
        if any(np.isfinite(v) for v in ce_no_abs_history):
            plt.plot(ce_no_abs_history, label="train (true weights)", linestyle="--")
        if val_ce_history and any(np.isfinite(v) for v in val_ce_history):
            plt.plot(val_ce_history, label="validation (true weights)")
        plt.xlabel("epochs")
        plt.ylabel("classifier cross entropy (Stage-B regularizer)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ce_plot)
        plt.close()

    # Penalty history
    penalty_plot = os.path.join(out_dir, f"penalty_history{finetune_suffix}.{plot_dtype}")
    plot_history(np.array(penalty_history), penalty_plot, x_label="Epoch", y_label="Low_bkg_penalty")

    # High-background-uncertainty penalty history (train + val, mirroring
    # the CE plot above) - B_w2 is always computed and this is always
    # tracked, but skip the plot entirely when the term is actually
    # disabled (lam=0.0 everywhere in this run), same convention as the CE
    # plot's `any(np.isfinite(...))` guard, since a flat all-same-scale
    # line for a term that never entered the loss adds no information.
    unc_penalty_plot = None
    if any(v != 0.0 for v in unc_penalty_history):
        unc_penalty_plot = os.path.join(out_dir, f"unc_penalty_history{finetune_suffix}.{plot_dtype}")
        plt.figure(figsize=(8, 6))
        plt.plot(unc_penalty_history, label="train")
        if val_unc_penalty_history:
            plt.plot(val_unc_penalty_history, label="validation")
        plt.xlabel("epochs")
        plt.ylabel("high_bkg_uncertainty_penalty (Stage-B regularizer)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(unc_penalty_plot)
        plt.close()

    # Covariance-damping penalty history - train (gato) only, no val
    # counterpart: this term depends solely on the GMM's own parameters
    # (model.gato.get_scale_tril()), not on any data split, so there is no
    # meaningful "validation" version of it - see covariance_damping_
    # penalty's docstring in training.py.
    cov_penalty_plot = None
    if any(v != 0.0 for v in cov_penalty_history):
        cov_penalty_plot = os.path.join(out_dir, f"cov_penalty_history{finetune_suffix}.{plot_dtype}")
        plot_history(
            np.array(cov_penalty_history), cov_penalty_plot,
            x_label="Epoch", y_label="Covariance_damping_penalty",
        )

    bias_plot = os.path.join(out_dir, f"bias_history{finetune_suffix}.{plot_dtype}")
    plot_bias_history(
        mean_bias_history, bias_plot, epochs=bias_epochs,
        temp_points=temperature_history, temp_label="Temperature",
    )

    bias_plot_log = os.path.join(out_dir, f"bias_history_log{finetune_suffix}.{plot_dtype}")
    plot_bias_history(
        mean_bias_history, bias_plot_log, epochs=bias_epochs,
        temp_points=temperature_history, temp_label="Temperature", log_scale=True
    )


    before_plot = os.path.join(out_dir, f"binning_final{finetune_suffix}.{plot_dtype}")
    plot_chunk_size = training_config.get("training", {}).get("chunk_size", None)
    y_plot = np.concatenate([np.asarray(y_train_stageB), np.asarray(y_val_stageB)])
    w_plot = np.concatenate([np.asarray(w_train_stageB), np.asarray(w_val_stageB)])
    # plot_category_binning() is available in end_to_end_helpers.py but is NOT
    # imported/called here by default: unlike make_plot_dataframe_from_model()
    # below, it scores X_all in a single unchunked forward pass, which defeats
    # the chunked/memmap memory-saving approach used everywhere else in this
    # script and can OOM on large datasets. Import and call it manually if you
    # want this diagnostic plot for a small dataset, e.g.:
    #   from end_to_end_helpers import plot_category_binning
    #   plot_category_binning(model, X_all, y_plot, before_plot,
    #                          title="Binning after end-to-end optimization")

    plot_data = make_plot_dataframe_from_model(
        model, X_all, y_plot, w_plot, signal_class, plot_dim, chunk_size=plot_chunk_size
    )
    assign, order, _, inv = assign_bins_and_order(gato, plot_data, reduce=True)
    filled = {
        proc: fill_histogram_from_assignments(assign[proc], plot_data[proc]["weight"], n_cats)
        for proc in plot_data
    }
    S_hard = hist_values(filled["signal"])   # gleiche Helper-Funktion wie in evaluate_model.py
    B_hard = hist_values(filled["background"])
    z_hard = np.sqrt(np.sum(np.asarray(asymptotic_significance(S_hard, B_hard)) ** 2))
    print(f"[CHECK] Hard-bin Z (train+val, same model, no reload) = {z_hard:.4f}")

    hist_plot = os.path.join(out_dir, f"optimized_histograms{finetune_suffix}.{plot_dtype}")
    scale_signal = 100
    plot_stacked_histograms(
        stacked_hists=[filled["background"]],
        process_labels=["background"],
        signal_hists=[filled["signal"] * scale_signal],
        signal_labels=[f"{scale_signal} x signal"],
        output_filename=hist_plot,
        axis_labels=("Category index", "Weighted events"),
        normalize=False,
        log=True,
    )

    print(f"\n[INFO] Saved fine-tuned classifier weights to: {os.path.join(out_dir, f'classifier_finetuned{finetune_suffix}.weights.h5')}")
    print(f"[INFO] Saved GMM checkpoint to: {checkpoint_dir}")
    if val_loss_history:
        print(f"[INFO] Saved loss plot (train+val): {loss_plot}")
    else:
        print(f"[INFO] Saved loss plot: {loss_plot}")
    print(f"[INFO] Saved bias plot: {bias_plot}")
    print(f"[INFO] Binning plot skipped (see plot_category_binning() note above); "
          f"would be saved to: {before_plot}")
    print(f"[INFO] Saved histogram plot: {hist_plot}")

    # ---------------- run summary ----------------
    run_info = {
        "timestamp": run_start_time,
        "classifier_weights_path": classifier_weights_path,
        "continue_training_enabled": continue_training_enabled,
        "continue_finetuning": continue_finetuning,
        "device": device_name,
        "input_size": input_size,
        "output_size": output_size,
        # .shape directly - np.asarray() here would read the ENTIRE feature
        # matrix into RAM just to look at its first dimension.
        "n_train_stageB": int(X_train_stageB.shape[0]),
        "n_val_stageB": int(X_val_stageB.shape[0]),
        "final_stageB_train_sig_loss": loss_history[-1] if loss_history else None,
        "final_stageB_val_sig_loss": val_loss_history[-1] if val_loss_history else None,
        "n_stageB_epochs_run": len(loss_history),
        "mass_window_enabled": mass_window_enabled,
        "profile_stageB": os.path.join(out_dir, "profile_stageB.json"),
        "host_peak_rss_MiB": (None if _host_peak_rss_bytes() is None
                              else round(_host_peak_rss_bytes() / 1024**2, 1)),
        "sequential_row_order": sequential_rows,
        "diagnostics_every": (training_config.get("training", {}) or {}).get(
            "diagnostics_every", 10),
    }
    if mass_window_enabled:
        run_info["mass_window_config"] = [mass_window_cfg["low"], mass_window_cfg["high"]]


    write_run_summary(out_dir, training_config, args, run_info)

    # ---------------- discriminant-density test-set snapshot ----------------
    # ONE-OFF (not periodic - see the training-loop calls inside
    # train_complete_model for the periodic train/val history), on the
    # held-out TEST split. Mirrors nonres_reweight's test-sideband fit:
    # always written unconditionally when the diagnostic itself is
    # enabled, regardless of evaluation.enabled below, since it's cheap
    # (one forward pass) and a natural companion to the training-time
    # history already being collected.
    disc_cfg = training_config.get("discriminant_density_diagnostics", {}) or {}
    if bool(disc_cfg.get("enabled", False)):
        disc_dir = os.path.join(out_dir, "discriminant_density_diagnostics")
        save_test_set_histograms(
            model, raw_data["X_test"], raw_data["y_test"], raw_data["rel_w_test"],
            classes,
            csv_path=os.path.join(disc_dir, disc_cfg.get("output_csv_name", "discriminant_density_history.csv")),
            meta_path=os.path.join(disc_dir, disc_cfg.get("output_meta_name", "discriminant_density_meta.json")),
            bins=int(disc_cfg.get("bins", 160)),
            score_range=tuple(disc_cfg.get("score_range", [0.0, 1.0])),
            chunk_size=int(disc_cfg.get("chunk_size") or DISC_DENSITY_DEFAULT_CHUNK_SIZE),
            max_events_per_class=disc_cfg.get("max_events_per_class", None),
            final_epoch=training_config.get("training", {}).get("n_epochs"),
        )
        print(f"[INFO] Saved discriminant-density test-set snapshot under: {disc_dir}")

    # ---------------- test-set evaluation ----------------
    eval_cfg = training_config.get("evaluation", {}) or {}
    if bool(eval_cfg.get("enabled", False)):
        print("\n[EVALUATION] evaluation.enabled=true: evaluating the just-trained "
              "model on the held-out TEST split.\n")
        evaluate_on_test_set(
            classifier, gato, signal_class, plot_dim,
            raw_data["X_test"], raw_data["y_test"], raw_data["rel_w_test"], mass_test_arr,
            class_weights["cw_test"],
            training_config, classes, out_dir,
            nonres_reweight_fn=compute_nonres_reweight_factors,
            chunk_size=eval_cfg.get("chunk_size", plot_chunk_size),
            plot_dtype=plot_dtype,
        )

        # ---- train/val/test significance comparison (Z_soft/Z_hard, both
        # nonres-reweight variants) - a broader companion to the test-only
        # evaluation above: per-(split, variant) event tables (CSV+PDF) plus
        # one master summary CSV+PDF across all of them, for judging how
        # much the significance moves between train/val/test and whether
        # the non-resonant background fit is behaving consistently across
        # splits. See evaluate_all_splits' docstring for exact file names.
        print("\n[EVALUATION] Building train/val/test significance summary "
              "(Z_soft/Z_hard, both nonres-reweight variants)...\n")
        evaluate_all_splits(
            classifier, gato, signal_class, plot_dim, raw_data,
            mass_train_arr, mass_val_arr, mass_test_arr,
            training_config, classes, out_dir,
            nonres_reweight_fn=compute_nonres_reweight_factors,
            chunk_size=eval_cfg.get("chunk_size", plot_chunk_size),
            plot_dtype=plot_dtype,
        )
    else:
        print(
            "\n[INFO] Test-set evaluation skipped (evaluation.enabled=false). "
            "Run evaluate_model.py separately against this run's out-dir to evaluate it later.\n"
        )

    # ---------------- event_counts tables ----------------
    # Per-category event-count/significance PDF tables, mirroring
    # gato_categorization_merged.py's make_event_count_pdf calls (see
    # build_event_count_tables' docstring in end_to_end_helpers.py for the
    # exact per-CLASS adaptation and the mechanism-2 "undo" used for the
    # combined table). Five tables are written under out_dir/event_counts/:
    # one combined (train+val+test), one per split (train/val/test, full
    # preselected mass range, no reweight), and one test-only table at the
    # FULL mass range (no [120,130] cut) with the non-resonant background
    # reweight factors (fit fresh on the test split's own sideband)
    # applied - the mass cut and the reweight are never combined into one
    # table.
    print("\n[EVENT COUNTS] Building event_counts tables (combined, per-split, "
          "test-with-nonres-reweight)...\n")
    build_event_count_tables(
        classifier, gato, signal_class, plot_dim, classes,
        raw_data, mass_train_arr, mass_val_arr, mass_test_arr,
        out_dir, chunk_size=plot_chunk_size,
        nonres_reweight_cfg=training_config.get("nonres_reweight"),
        plot_dtype=plot_dtype,
        drop_last_class=drop_last_class,
        data_prep_cfg=training_config.get("data_prep"),
    )


if __name__ == "__main__":
    main()