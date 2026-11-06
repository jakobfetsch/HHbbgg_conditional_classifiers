#!/usr/bin/env python3
"""
plot_discriminant_density.py

Regenerates the combined discriminant-space density figure (one panel
per pair of the classifier's full softmax dimensions - all C(n_classes, 2)
of them, e.g. 6 for 4 classes - each showing every true class's 1/2/3-
sigma-equivalent enclosed-density contours) from the sparse histogram
history CSV written by discriminant_density_diagnostics.
append_class_dim_histograms() / save_test_set_histograms() during/after
Stage-B training.

No access to the original training run, checkpoints, or data is needed -
everything is reconstructed from the saved histogram bins.

Usage:
    # Train split, latest epoch:
    python plot_discriminant_density.py \
        --csv /path/to/out_dir/discriminant_density_diagnostics/discriminant_density_history.csv \
        --meta /path/to/out_dir/discriminant_density_diagnostics/discriminant_density_meta.json \
        --output-dir /path/to/out_dir/discriminant_density_diagnostics/plots \
        --split train

    # Val split, a specific epoch:
    python plot_discriminant_density.py --csv ... --meta ... --output-dir ... --split val --epoch 120

    # Test split (single one-off snapshot):
    python plot_discriminant_density.py --csv ... --meta ... --output-dir ... --split test

    # Every train epoch present in the CSV:
    python plot_discriminant_density.py --csv ... --meta ... --output-dir ... --split train --all-epochs
"""

import argparse
import os

import pandas as pd

from discriminant_density_diagnostics import plot_discriminant_density_from_csv


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the discriminant-density figure from a discriminant_density_history.csv."
    )
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--meta", required=True, type=str,
                         help="Path to the sidecar discriminant_density_meta.json (bins/score_range/classes).")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"],
                         help="Which split's rows to plot.")
    parser.add_argument("--epoch", default=None, type=int,
                         help="Which epoch to plot (ignored for --split test, which has one snapshot). "
                              "Omit for the latest epoch present.")
    parser.add_argument("--all-epochs", action="store_true",
                         help="Plot every epoch present for --split (ignores --epoch; not meaningful for test).")
    parser.add_argument("--smooth-sigma", default=0.0, type=float,
                         help="Optional Gaussian smoothing (in bin units) applied only at plot time, "
                              "for less blocky/fragmented contours at fine bin counts. "
                              "0 (default) = raw histogram, no shape assumption added. "
                              "Note: small values (e.g. 0.5) won't fix contours that look like "
                              "isolated 'dots' rather than smooth regions - see --merge-bins for that.")
    parser.add_argument("--merge-bins", default=1, type=int,
                         help="Block-sum this many x this many histogram bins into one before "
                              "computing contours (e.g. 4 turns a 160x160 grid into 40x40). Use "
                              "this FIRST if contours come out as isolated 'dots' instead of smooth "
                              "regions - a sparse-histogram symptom (too many bins for the "
                              "available event count). 1 (default) = no merging. Combine with a "
                              "modest --smooth-sigma on top of the merged grid for an even less "
                              "blocky result.")
    parser.add_argument("--plot-dtype", default="png", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all_epochs:
        df = pd.read_csv(args.csv, usecols=["epoch", "split"])
        epochs = sorted(df[df["split"] == args.split]["epoch"].unique().tolist())
        print(f"[INFO] Found {len(epochs)} epochs for split={args.split!r} in {args.csv}: {epochs}")
        for ep in epochs:
            out_path = os.path.join(args.output_dir, f"discriminant_density_{args.split}_epoch{ep}.{args.plot_dtype}")
            plot_discriminant_density_from_csv(args.csv, args.meta, out_path, split=args.split, epoch=ep,
                                                smooth_sigma=args.smooth_sigma, merge_bins=args.merge_bins)
    else:
        epoch = args.epoch
        suffix = "latest" if epoch is None else str(epoch)
        out_path = os.path.join(args.output_dir, f"discriminant_density_{args.split}_epoch{suffix}.{args.plot_dtype}")
        plot_discriminant_density_from_csv(args.csv, args.meta, out_path, split=args.split, epoch=epoch,
                                            smooth_sigma=args.smooth_sigma, merge_bins=args.merge_bins)

    print(f"\n[INFO] Done. Plot(s) written to: {args.output_dir}")


if __name__ == "__main__":
    main()