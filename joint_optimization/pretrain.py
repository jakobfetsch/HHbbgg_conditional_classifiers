#!/usr/bin/env python3
"""
pretrain.py

Stage A only: supervised pretraining of the classifier with weighted
cross-entropy, matching training_utils.py/mlp.py's (PyTorch) training
behaviour as closely as possible - see training.py's pretrain() for the
per-epoch mechanics (weighted CE, Adam + manually-coupled weight decay to
match torch.optim.Adam(weight_decay=...) exactly, ReduceLROnPlateau, early
stopping) this script drives.

CLASSIFIER (Stage A) PRESELECTION: by default, the classifier is now
pretrained on a preselected event sample - dataset_prep.
classifier_preselection_mask, EXACTLY prepare_inputs.py's preselection()
(a fixed mass/dijet-mass/mvaID window, unconditionally applied - no
per-cut enable switches). This is a SEPARATE, fixed cut from the broader,
fully config-driven 'preselection' section combined_optimization.py (Stage
B) reads and applies on top of its own is_boosted veto and classifier-
score cut - --training-config-path's 'preselection' section (if any) is
still ignored here. The classifier weight rescaling (class_weights_for_set
- each of the two "normal" backgrounds normalized to sum to 1
independently, the two di-Higgs processes combined via config["class_
groups"] to sum to 1 together) happens AFTER this preselection is applied,
inside dataset_prep.build_training_dataset, so the weight sums going into
Stage A training are correct for the actually-preselected population.
Set data_prep.apply_classifier_preselection: false in
--training-config-path to go back to the old (fully unpreselected)
behavior.

Everything else is config-driven; run as:

    python pretrain.py --base-dir /path/to/individual_samples \
        --training-config-path pretrain_config.yaml

Writes, under pretrain_config.yaml's out-dir:
    classifier_pretrained.weights.h5   - final (best-val-loss) weights;
                                          combined_optimization.py's
                                          pretrained_classifier_weights_path
                                          should point here.
    pretrain_loss_plot.<plot_dtype>, pretrain_loss_plot_no_abs.<plot_dtype>,
                                          pretrain_history.csv
    pretrain_checkpoints/, pretrain_checkpoints_best/,
    pretrain_best_checkpoints/          - see pretrain()'s docstring in
                                          training.py for what each holds.
"""

import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from mlp_tf import MLP
from training import pretrain, save_with_retry
from end_to_end_helpers import build_raw_dataset_from_samples, shuffle_raw_data
from dataset_prep import DEFAULT_ERAS, load_config as load_data_config, classifier_preselection_kwargs


def main():
    parser = argparse.ArgumentParser(description="Stage-A classifier pretraining")
    parser.add_argument("--base-dir", required=True,
                         help="Path to individual_samples/ (contains <era>/<sample>/{X.npy,rel_w.npy,events.parquet}).")
    parser.add_argument("--training-config-path", required=True,
                         help="YAML with seeds.*, cuda_device, data_prep.*, class_weight_scheme, "
                              "class_model, pretrain_optimizer, pretrain.*, profiling - "
                              "see pretrain_config.yaml. Any top-level 'preselection' section is "
                              "still ignored (that is the SEPARATE, Stage-B-only 'gato model' "
                              "cut) - Stage A's own preselection is instead controlled by "
                              "data_prep.apply_classifier_preselection (default true, see NOTE "
                              "below).")
    parser.add_argument("--num-threads", type=int, default=None,
                         help="Override intra-op/inter-op CPU thread count (default: all visible cores, "
                              "the previous unconditional behavior). Set this explicitly to a fraction of "
                              "your core count when running several pretrain.py processes concurrently on "
                              "the same machine (e.g. total_cores // n_concurrent_runs) - otherwise each "
                              "process independently requests every core and they oversubscribe each "
                              "other, which usually makes ALL of them slower than running sequentially. "
                              "See run_pretrain_suite_parallel.py, which sets this automatically.")
    args = parser.parse_args()

    with open(args.training_config_path, "r") as f:
        training_config = yaml.safe_load(f)

    seeds = training_config["seeds"]

    if args.num_threads is not None:
        n_cpus = max(1, int(args.num_threads))
    else:
        try:
            n_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            n_cpus = os.cpu_count() or 1
    tf.config.threading.set_intra_op_parallelism_threads(n_cpus)
    tf.config.threading.set_inter_op_parallelism_threads(n_cpus)
    print(f"[INFO] TF intra-op/inter-op parallelism threads set to {n_cpus}"
          f"{' (from --num-threads)' if args.num_threads is not None else ' (all visible cores)'}.")

    # Also cap BLAS-backed CPU thread pools (numpy/scipy operations - e.g.
    # bincount, softmax-adjacent ops during confusion-matrix computation,
    # any BLAS calls inside dataset prep) that read these env vars at
    # process start and otherwise independently auto-detect every core,
    # same oversubscription risk as the TF intra/inter-op settings above.
    # Must be set before numpy imports its BLAS backend - here that's
    # already too late for THIS process's own numpy import above, but the
    # common backends (OpenBLAS/MKL) additionally consult these env vars
    # at every threaded call, not just at import time, so setting them
    # here still constrains everything numpy does for the rest of this run.
    if args.num_threads is not None:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(n_cpus)

    random.seed(seeds["shuffle"])
    np.random.seed(seeds["shuffle"])
    tf.random.set_seed(seeds["classifier_init"])
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    gpus = tf.config.list_physical_devices("GPU")
    try:
        cuda_device = int(training_config["cuda_device"])
    except (KeyError, ValueError):
        cuda_device = -1
    if not gpus:
        raise RuntimeError(
            "No CUDA-capable GPU was detected by TensorFlow, but this job requested "
            f"cuda_device={cuda_device}. Refusing to silently fall back to CPU."
        )
    if cuda_device < 0 or cuda_device >= len(gpus):
        raise ValueError(f"Requested cuda_device={cuda_device}, but only {len(gpus)} GPU(s) available: {gpus}")
    selected_gpu = gpus[cuda_device]
    tf.config.set_visible_devices(selected_gpu, "GPU")
    tf.config.experimental.set_memory_growth(selected_gpu, True)
    device_name = "/GPU:0"
    print(f"[INFO] Using physical GPU index {cuda_device} ({selected_gpu}) as {device_name}")

    out_dir = training_config["out-dir"]
    os.makedirs(out_dir, exist_ok=True)
    plot_dtype = training_config["plot_dtype"]

    # ---------------- data ----------------
    data_prep_cfg = training_config["data_prep"]
    data_config = load_data_config(data_prep_cfg["data_config_path"])
    fraction = float(data_prep_cfg["fraction"])
    train_fraction = float(data_prep_cfg.get("train_fraction", 0.7))
    val_fraction = float(data_prep_cfg.get("val_fraction", 0.15))
    eras = data_prep_cfg.get("eras") or DEFAULT_ERAS
    use_class_groups = bool(data_prep_cfg.get("use_class_groups", True))

    print(f"[INFO] Building raw dataset from '{args.base_dir}' (eras={eras})")
    # ----------------------------------------------------------------------
    # NOTE - CLASSIFIER (Stage A) PRESELECTION, ON BY DEFAULT.
    # classifier_preselection_kwargs resolves data_prep.
    # apply_classifier_preselection (default true) into the
    # apply_classifier_preselection flag + the extra parquet columns
    # (dijet_mass, lead/sublead mvaID) it needs alongside mass - see
    # dataset_prep.classifier_preselection_mask/build_training_dataset.
    # This is a SEPARATE, fixed cut from the broader, fully config-driven
    # 'preselection' section combined_optimization.py (Stage B) reads -
    # that section, if present in --training-config-path, is still
    # ignored here. (There is also no classifier-score-based cut possible
    # here regardless, since that needs a trained classifier, which
    # doesn't exist yet at this stage.)
    # ----------------------------------------------------------------------
    presel_kwargs = classifier_preselection_kwargs(data_prep_cfg)
    print(f"[INFO] Classifier (Stage A) preselection: "
          f"{'ENABLED' if presel_kwargs['apply_classifier_preselection'] else 'disabled'}")
    raw_data, class_weights, _mass_train_arr, _mass_val_arr, _mass_test_arr, classes, _split_rescale = \
        build_raw_dataset_from_samples(
            args.base_dir, data_config, fraction, seeds["data"], eras,
            train_fraction=train_fraction, val_fraction=val_fraction,
            weight_scheme=training_config["class_weight_scheme"],
            class_groups=("auto" if use_class_groups else None),
            info_out_path=os.path.join(out_dir, "data_info.json"),
            # Separate seed for the train/val/test split, independent of
            # seeds["data"]'s fraction-sampling - see build_raw_dataset_
            # from_samples' docstring. Optional: falls back to seeds["data"]
            # (old behavior, one seed for both) if seeds.data_split isn't
            # in the config at all.
            split_seed=seeds.get("data_split"),
            **presel_kwargs,
        )

    # Same row-order shuffle combined_optimization.py applies, so a
    # continue_training run against this checkpoint sees a train_dataset
    # built from the identical rows in the identical order up to this
    # point - only the seed used differs by design (seeds.shuffle here).
    raw_data, class_weights, train_perm, val_perm = shuffle_raw_data(raw_data, class_weights, seeds["shuffle"])

    input_size = raw_data["X_train"].shape[1]
    output_size = len(np.unique(np.asarray(raw_data["y_train"][:])))
    print(f"[INFO] input_size={input_size}, output_size (n_classes)={output_size}")

    num_layers, num_nodes, act_fn_name, dropout_prob = training_config["class_model"].values()

    with tf.device(device_name):
        classifier = MLP(input_size, num_layers, num_nodes, output_size, act_fn_name, dropout_prob,
                          seed=seeds["classifier_init"])

    # Save initial checkpoint:
    save_with_retry(
                        classifier.save_weights,
                        os.path.join(out_dir, "classifier_initial.weights.h5"),
                        label="Stage A initial model",
                    )

    # ---------------- Stage A ----------------
    best_loss, train_loss_history, val_loss_history, train_loss_no_abs_history = pretrain(
        classifier, raw_data, class_weights, training_config,
        profile_out_path=os.path.join(out_dir, "profile_stageA.json"),
        out_dir=out_dir, classes=classes,
    )

    try:
        save_with_retry(
            classifier.save_weights, os.path.join(out_dir, "classifier_pretrained.weights.h5"),
            label="final Stage A weights",
        )
    except OSError:
        classifier.save_weights("classifier_pretrained.weights.h5")
        print("[WARN] Could not write to out-dir; saved pretrained weights to the current directory instead.")
    print(f"[Stage A] Done. Best val loss = {best_loss:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label="train")
    plt.plot(val_loss_history, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pretrain_loss_plot.{plot_dtype}"))
    plt.close()

    # SECOND, apples-to-apples loss plot: train_loss_no_abs (weighted by
    # class_weights["cw_train_no_abs"] - i.e. true_class_weights, the SIGNED
    # rel_w, computed by train_one_epoch alongside the primary train_loss
    # but never plotted before now) against val_loss. dataset_prep never
    # abs's the val split's class weights (class_weights_val is always built
    # from the signed rel_w_val - see build_training_dataset), so val_loss
    # already uses signed weights throughout - train_loss_no_abs is its
    # true train-side counterpart. The PRIMARY plot above instead compares
    # train_loss (weighted by whatever class_weight_scheme selected -
    # class_weights_abs or class_weights_only_positive, always
    # non-negative) against that same signed-weighted val_loss - useful for
    # tracking what the classifier is actually optimized against, but not a
    # like-for-like comparison. This plot is.
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_no_abs_history, label="train (no abs weights)")
    plt.plot(val_loss_history, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy (signed/no-abs weights)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pretrain_loss_plot_no_abs.{plot_dtype}"))
    plt.close()

    history_csv = os.path.join(out_dir, "pretrain_history.csv")
    with open(history_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_ce", "val_ce", "train_ce_no_abs"])
        for i, (tr, va, tr_no_abs) in enumerate(
            zip(train_loss_history, val_loss_history, train_loss_no_abs_history)
        ):
            writer.writerow([i, tr, va, tr_no_abs])

    print(f"[INFO] Saved pretrained weights to: {os.path.join(out_dir, 'classifier_pretrained.weights.h5')}")
    print(f"[INFO] Saved loss plot to: {os.path.join(out_dir, f'pretrain_loss_plot.{plot_dtype}')}")
    print(f"[INFO] Saved no-abs loss plot to: {os.path.join(out_dir, f'pretrain_loss_plot_no_abs.{plot_dtype}')}")
    print(f"[INFO] Saved history to: {history_csv}")


if __name__ == "__main__":
    main()