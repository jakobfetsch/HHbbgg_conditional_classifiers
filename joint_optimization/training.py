import contextlib
import csv
import json
import os
import shutil
import sys
import tempfile
import time

import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
from gatohep.utils import TemperatureScheduler, asymptotic_significance
from gatohep.losses import low_bkg_penalty, high_bkg_uncertainty_penalty

def make_schedule(lr, steps=50, decay=0.5):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=steps,
        decay_rate=decay,
        staircase=True,
    )

def scaled_nbins_for_split(base_nbins, split_name, data_prep_cfg, min_bins=3):
    """
    Scales a sideband-fit bin count (e.g. nonres_reweight.nbins, which is
    effectively tuned for however many TRAIN-split events land in the
    non-resonant-background sideband) down for a split with proportionally
    fewer events, so val/test fits aren't left with the same nbins as
    train despite having far fewer events per bin to fit against.

    Actual per-split event counts aren't known at every call site (some
    only have `nonres_reweight_cfg`, not the full loaded dataset), but the
    split itself is always drawn at the fixed train_fraction/val_fraction
    ratios in training_config["data_prep"] (see dataset_prep.
    stratified_train_test_split) - independent of `fraction` (which
    shrinks every split by the same factor and so cancels out of the
    ratio). That ratio is therefore a reliable, always-available stand-in
    for "how many events does this split have relative to train":

        scaled = round(base_nbins * split_fraction / train_fraction)

    i.e. LINEAR in expected event count - this keeps the average number
    of sideband events per bin roughly the same as whatever fit
    base_nbins was originally tuned for on the train split, rather than
    keeping nbins fixed while events-per-bin drops (emptier bins -> a
    noisier, less-constrained exponential fit - exactly what motivated
    this in the first place).

    split_name: "train" (returned unchanged - it's the reference split),
    "val", or "test" (test_fraction is derived as
    1 - train_fraction - val_fraction, same convention as
    training_config.yaml's data_prep.val_fraction comment).

    min_bins: hard floor (default 5). An exponential sideband fit has 2
    free parameters (normalization + slope - see compute_nonres_reweight_
    factors in end_to_end_helpers.py), so below ~5 bins there's too little
    independent shape information left to constrain both, no matter how
    few events remain - this floor is about the fit being well-posed, not
    about event statistics.

    Lives here (not end_to_end_helpers.py, despite being used mostly for
    THAT module's compute_nonres_reweight_factors) because end_to_end_
    helpers.py already imports from training.py at module level (see its
    own top-level import) - importing the other way round would be
    circular. Same reason nonres_reweight_fn is passed into
    train_complete_model as a callback instead of imported directly.
    """
    base_nbins = int(base_nbins)
    if split_name == "train":
        return base_nbins

    data_prep_cfg = data_prep_cfg or {}
    train_fraction = float(data_prep_cfg.get("train_fraction", 0.7))
    val_fraction = float(data_prep_cfg.get("val_fraction", 0.15))
    if train_fraction <= 0:
        raise ValueError(f"data_prep.train_fraction must be > 0 to scale nbins, got {train_fraction}")

    if split_name == "val":
        split_fraction = val_fraction
    elif split_name == "test":
        split_fraction = max(0.0, 1.0 - train_fraction - val_fraction)
    else:
        raise ValueError(f"split_name must be 'train', 'val', or 'test', got {split_name!r}")

    scaled = int(round(base_nbins * split_fraction / train_fraction))
    return max(min_bins, scaled)


def resolve_nonres_nbins(nonres_cfg, split_name, data_prep_cfg, min_bins=3):
    """
    Thin convenience wrapper around scaled_nbins_for_split: pulls the base
    bin count out of a nonres_reweight-style config dict (nonres_cfg -
    either training_config's own nonres_reweight section, or the
    nonres_reweight_cfg passed around end_to_end_helpers.py's
    compute_nonres_reweight_factors call sites) with the same
    default (40) used at every other nbins=... call site in this file,
    then scales it for split_name. See scaled_nbins_for_split's docstring
    for the scaling itself.

    Exists so end_to_end_helpers.py (which needs this at several call
    sites) doesn't have to duplicate the `.get("nbins", 40)` lookup - and
    lives here rather than there for the same import-direction reason
    given in scaled_nbins_for_split's docstring.
    """
    nonres_cfg = nonres_cfg or {}
    return scaled_nbins_for_split(nonres_cfg.get("nbins", 40), split_name, data_prep_cfg, min_bins=min_bins)


def save_with_retry(write_fn, path, retries=5, initial_backoff_s=5.0, max_backoff_s=60.0,
                     label=None, ensure_dir=None):
    """
    Calls write_fn(path) (e.g. `lambda p: classifier.save_weights(p)`, or
    `lambda p: shutil.copy(src, p)`), retrying with exponential backoff on
    OSError - covers FileNotFoundError, which is what a transient
    autofs/NFS remount hiccup on a network-mounted out-dir surfaces as.

    Written for exactly this failure: a PERIODIC checkpoint write (e.g.
    pretrain's checkpoint_every snapshot) crashed the whole run with
    FileNotFoundError on `classifier.save_weights(ckpt_path)`, even though
    the same checkpoint_dir had been created once at startup and several
    earlier periodic writes to that SAME directory had already succeeded
    - i.e. nothing in this codebase ever deletes/moves that directory
    mid-run. The much more likely explanation is the network filesystem
    itself: automounters (autofs) commonly unmount an NFS path after a
    period of no activity, and if the gap between periodic writes to an
    otherwise-idle out-dir (checkpoint_every epochs' worth of GPU-bound
    compute, reading training data from a DIFFERENT mount the whole time)
    exceeds that idle timeout, the out-dir's mount can be dropped and not
    yet cleanly remounted the instant h5py tries to create a new file
    there - surfacing as ENOENT even though the directory conceptually
    still exists server-side. Heavy concurrent I/O elsewhere on the same
    filesystem (large chunked reads, several diagnostics CSVs being
    appended to at once - see gmm_contour_diagnostics/discriminant_
    density_diagnostics/nonres_reweight's periodic writes) makes this kind
    of race more likely to be hit, not less, since there's more
    contention around exactly when the automounter decides the path is
    "idle".

    Before each attempt, re-runs os.makedirs(exist_ok=True) on either
    ensure_dir (if given - use this for a DIRECTORY-target write, e.g.
    gato.save(checkpoint_dir), where `path` itself is the directory to
    recreate) or path's own parent directory (the default, for a
    FILE-target write like classifier.save_weights). Cheap, idempotent,
    and is what actually works around an autofs-unmounted-mid-run
    scenario: touching the path forces a fresh automount lookup, which a
    raw h5py file-create (or gatohep's own directory write) call does not
    reliably trigger on every autofs configuration.

    Raises the LAST error if every retry is exhausted, so a genuinely
    persistent problem (disk actually full, permissions, wrong path) still
    surfaces loudly instead of being silently swallowed.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            os.makedirs(ensure_dir if ensure_dir is not None else (os.path.dirname(path) or "."), exist_ok=True)
            write_fn(path)
            if attempt > 0:
                print(f"[INFO] {label or path}: write succeeded on retry {attempt}.")
            return
        except OSError as e:
            last_err = e
            if attempt >= retries:
                break
            backoff = min(max_backoff_s, initial_backoff_s * (2 ** attempt))
            print(
                f"[WARNING] {label or path}: write failed on attempt {attempt + 1}/{retries + 1} "
                f"({type(e).__name__}: {e}). Retrying in {backoff:.0f}s - this usually means a "
                f"transient network-filesystem hiccup (e.g. an autofs mount idle-unmounted "
                f"mid-run), not a real/persistent problem with the path or disk. "
                f"See save_with_retry's docstring for the full reasoning."
            )
            time.sleep(backoff)
    raise last_err

try:
    import resource
except ImportError:  # non-POSIX
    resource = None


# ---------------------------------------------------------------------------
# Wall-clock / memory instrumentation
# ---------------------------------------------------------------------------
# Purpose: make it possible to answer "did that change actually help, and by
# how much?" quantitatively rather than by feel. Everything here is inert
# unless training.profiling.enabled is true in the config, and the numbers are
# written to a JSON file next to the other run outputs so two runs can be
# diffed directly (see compare_profiles.py).

def _bytes_to_mib(x):
    return None if x is None else x / (1024.0 ** 2)


def _host_rss_bytes():
    """Current resident set size of this process, in bytes."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _host_peak_rss_bytes():
    """High-water-mark RSS since process start, in bytes."""
    if resource is None:
        return None
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS/BSD report bytes.
        return ru * 1024 if sys.platform.startswith("linux") else ru
    except Exception:
        return None


def _gpu_memory_bytes(device="GPU:0"):
    """(current, peak) device memory in bytes, or (None, None) without a GPU."""
    try:
        info = tf.config.experimental.get_memory_info(device)
        return float(info["current"]), float(info["peak"])
    except (ValueError, KeyError, AttributeError):
        return None, None


def _reset_gpu_peak(device="GPU:0"):
    """Resets the device peak counter to the current allocation."""
    try:
        tf.config.experimental.reset_memory_stats(device)
        return True
    except (ValueError, KeyError, AttributeError):
        return False


_SYNC_PROBE = None


def _device_sync():
    """
    Blocks until work already queued on the device is finished.

    Needed because TF enqueues GPU kernels asynchronously: without a barrier,
    a timer around a forward pass measures how long it took to *launch* the
    kernels, not to run them, and the cost silently lands in whatever phase
    happens to read a tensor next. Enqueuing a trivial op and copying it back
    to the host drains the stream (device streams are FIFO), which is enough
    to make the per-phase attribution honest.
    """
    global _SYNC_PROBE
    try:
        if _SYNC_PROBE is None:
            _SYNC_PROBE = tf.Variable(0.0, dtype=tf.float32)
        _SYNC_PROBE.assign_add(0.0).numpy()
    except Exception:
        pass


class PhaseProfiler:
    """
    Accumulates wall-clock time (and, for the device, memory peaks) per named
    phase of an epoch.

    Two levels of phase:
      * top-level  - disjoint, and together they account for the whole epoch
                     ('train_steps', 'validation', 'diagnostics').
      * sub-phase  - a breakdown INSIDE train_steps ('chunk_io',
                     'pass1_forward', 'pass2_fwd_bwd', 'optimizer_apply').
                     These are reported separately because they are a
                     partition of train_steps, not additional to it.

    Disabled profilers hand out a shared no-op context manager, so the
    instrumented code paths cost essentially nothing when profiling is off.
    """

    def __init__(self, enabled=True, sync=True, device="GPU:0", label=""):
        self.enabled = bool(enabled)
        self.sync = bool(sync)
        self.device = device
        self.label = label
        self.totals = {}      # phase -> {"seconds": float, "calls": int}
        self.sub_phases = set()
        self.epochs = []      # one record per epoch
        self.bytes_read = 0
        self._epoch = None
        self._epoch_t0 = None
        self._chunk_peak_bytes = None
        self.has_gpu = _gpu_memory_bytes(device)[0] is not None

    # -- phase timing -------------------------------------------------------
    @contextlib.contextmanager
    def _noop(self):
        yield

    def phase(self, name, sub=False):
        if not self.enabled:
            return self._noop()
        return self._timed(name, sub)

    @contextlib.contextmanager
    def _timed(self, name, sub):
        if self.sync:
            _device_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                _device_sync()
            dt = time.perf_counter() - t0
            rec = self.totals.setdefault(name, {"seconds": 0.0, "calls": 0})
            rec["seconds"] += dt
            rec["calls"] += 1
            if sub:
                self.sub_phases.add(name)
            if self._epoch is not None:
                self._epoch.setdefault(name, 0.0)
                self._epoch[name] += dt

    def add_bytes(self, n):
        """Records bytes pulled off the feature array, for a MiB/s estimate."""
        if self.enabled and n:
            self.bytes_read += int(n)
            if self._epoch is not None:
                self._epoch["_bytes"] = self._epoch.get("_bytes", 0) + int(n)

    # -- device-memory probes ----------------------------------------------
    def mark_chunk_step_start(self):
        """Reset the device peak so the next read isolates the chunked step."""
        if self.enabled and self.has_gpu:
            _device_sync()
            _reset_gpu_peak(self.device)

    def mark_chunk_step_end(self):
        """
        Peak device memory reached during the chunked forward+backward - the
        number to look at when tuning training.chunk_size, and the one the
        persistent-vs-single tape change moves.
        """
        if self.enabled and self.has_gpu:
            _device_sync()
            _, peak = _gpu_memory_bytes(self.device)
            if peak is not None:
                self._chunk_peak_bytes = max(self._chunk_peak_bytes or 0.0, peak)

    # -- epoch bookkeeping --------------------------------------------------
    def start_epoch(self, ep):
        if not self.enabled:
            return
        self._epoch = {"epoch": int(ep)}
        self._chunk_peak_bytes = None
        if self.has_gpu:
            _reset_gpu_peak(self.device)
        self._epoch_t0 = time.perf_counter()

    def end_epoch(self):
        if not self.enabled or self._epoch is None:
            return None
        if self.sync:
            _device_sync()
        rec = self._epoch
        rec["wall_s"] = time.perf_counter() - self._epoch_t0
        gpu_cur, gpu_peak = _gpu_memory_bytes(self.device)
        rec["gpu_current_MiB"] = _bytes_to_mib(gpu_cur)
        rec["gpu_peak_MiB"] = _bytes_to_mib(gpu_peak)
        rec["gpu_peak_chunk_step_MiB"] = _bytes_to_mib(self._chunk_peak_bytes)
        rec["host_rss_MiB"] = _bytes_to_mib(_host_rss_bytes())
        rec["host_peak_rss_MiB"] = _bytes_to_mib(_host_peak_rss_bytes())
        self.epochs.append(rec)
        self._epoch = None
        return rec

    def epoch_line(self):
        """Compact per-epoch string to append to the existing Stage-B log."""
        if not self.enabled or not self.epochs:
            return ""
        r = self.epochs[-1]
        out = f" | {r['wall_s']:.2f}s"
        for name in ("train_steps", "validation", "diagnostics"):
            if name in r:
                out += f" ({name.split('_')[0]} {r[name]:.2f}s)"
        if r.get("gpu_peak_MiB") is not None:
            out += f" gpu_peak={r['gpu_peak_MiB']:.0f}MiB"
        if r.get("host_rss_MiB") is not None:
            out += f" rss={r['host_rss_MiB']:.0f}MiB"
        return out

    # -- reporting ----------------------------------------------------------
    def _epoch_times(self):
        return [r["wall_s"] for r in self.epochs]

    def to_dict(self, extra=None):
        times = self._epoch_times()
        gpu_peaks = [r["gpu_peak_MiB"] for r in self.epochs if r.get("gpu_peak_MiB") is not None]
        chunk_peaks = [r["gpu_peak_chunk_step_MiB"] for r in self.epochs
                       if r.get("gpu_peak_chunk_step_MiB") is not None]
        rss = [r["host_rss_MiB"] for r in self.epochs if r.get("host_rss_MiB") is not None]
        d = {
            "label": self.label,
            "n_epochs": len(self.epochs),
            "total_wall_s": float(sum(times)) if times else 0.0,
            "mean_epoch_s": float(np.mean(times)) if times else None,
            "median_epoch_s": float(np.median(times)) if times else None,
            "min_epoch_s": float(np.min(times)) if times else None,
            "max_epoch_s": float(np.max(times)) if times else None,
            "phase_totals_s": {k: v["seconds"] for k, v in self.totals.items()
                               if k not in self.sub_phases},
            "sub_phase_totals_s": {k: v["seconds"] for k, v in self.totals.items()
                                   if k in self.sub_phases},
            "phase_calls": {k: v["calls"] for k, v in self.totals.items()},
            "bytes_read": int(self.bytes_read),
            "gib_read": self.bytes_read / (1024.0 ** 3),
            "gpu_peak_MiB_max": float(max(gpu_peaks)) if gpu_peaks else None,
            "gpu_peak_chunk_step_MiB_max": float(max(chunk_peaks)) if chunk_peaks else None,
            "host_rss_MiB_max": float(max(rss)) if rss else None,
            "host_peak_rss_MiB": _bytes_to_mib(_host_peak_rss_bytes()),
            "epochs": self.epochs,
        }
        if extra:
            d.update(extra)
        return d

    def report(self, n_rows=None):
        """Human-readable summary table."""
        if not self.enabled or not self.epochs:
            return
        d = self.to_dict()
        w = 62
        print("\n" + "=" * w)
        print(f" PROFILE{(' - ' + self.label) if self.label else ''}")
        print("=" * w)
        print(f" epochs run                  : {d['n_epochs']}")
        print(f" total wall clock            : {d['total_wall_s']:.1f} s")
        print(f" per epoch  mean / median    : {d['mean_epoch_s']:.2f} s / {d['median_epoch_s']:.2f} s")
        print(f"            min / max        : {d['min_epoch_s']:.2f} s / {d['max_epoch_s']:.2f} s")
        if n_rows:
            rate = n_rows / d["mean_epoch_s"] if d["mean_epoch_s"] else 0.0
            print(f" throughput                  : {rate:,.0f} train rows/s (per epoch)")

        total = d["total_wall_s"] or 1.0
        if d["phase_totals_s"]:
            print("-" * w)
            print(" phase breakdown (disjoint, sums to the epoch)")
            for k, v in sorted(d["phase_totals_s"].items(), key=lambda kv: -kv[1]):
                print(f"   {k:<24} {v:8.1f} s   {100.0 * v / total:5.1f} %")
        if d["sub_phase_totals_s"]:
            sub_total = sum(d["sub_phase_totals_s"].values()) or 1.0
            print("-" * w)
            print(" inside train_steps (a partition of the row above)")
            for k, v in sorted(d["sub_phase_totals_s"].items(), key=lambda kv: -kv[1]):
                print(f"   {k:<24} {v:8.1f} s   {100.0 * v / sub_total:5.1f} %")
            io_s = d["sub_phase_totals_s"].get("chunk_io", 0.0)
            if io_s > 0 and d["bytes_read"]:
                print(f"   -> feature data read     : {d['gib_read']:.2f} GiB "
                      f"at {d['bytes_read'] / io_s / 1024**2:,.0f} MiB/s")
        print("-" * w)
        print(" memory")
        if d["gpu_peak_MiB_max"] is not None:
            print(f"   GPU peak (any phase)     : {d['gpu_peak_MiB_max']:8.0f} MiB")
            if d["gpu_peak_chunk_step_MiB_max"] is not None:
                print(f"   GPU peak (chunked step)  : {d['gpu_peak_chunk_step_MiB_max']:8.0f} MiB"
                      f"   <- scales with training.chunk_size")
        else:
            print("   GPU                      :   not active / not tracked")
        if d["host_rss_MiB_max"] is not None:
            print(f"   host RSS max             : {d['host_rss_MiB_max']:8.0f} MiB")
        if d["host_peak_rss_MiB"] is not None:
            print(f"   host RSS high-water mark : {d['host_peak_rss_MiB']:8.0f} MiB")
        print("=" * w + "\n")

    def save(self, path, extra=None):
        if not self.enabled or not path:
            return None
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.to_dict(extra=extra), f, indent=2)
            print(f"[PROFILE] Wrote timing/memory profile to {path}")
            return path
        except OSError as e:
            print(f"[PROFILE][WARNING] Could not write profile to {path}: {e}")
            return None



def soft_mass_window_weight(mass, low, high, softness):
    """
    Differentiable "soft" replacement for a hard mass-window cut
    ``(mass >= low) & (mass <= high)``.

    Returns, for each event, a weight in (0, 1):
        sigmoid((mass - low) / softness) * sigmoid((high - mass) / softness)
    which is close to 1 well inside [low, high], close to 0 well outside,
    and transitions smoothly (width ~softness) at each edge instead of
    jumping discontinuously. As softness -> 0 this converges to the hard
    cut.

    `mass` may be a numpy array or a tf.Tensor; the computation is done
    with tf ops throughout so the result is a genuine part of the graph
    (as opposed to a numpy boolean mask computed outside of it).

    Caveat: `softness` is a per-edge scale, not the window width. If it is
    not small compared to (high-low)/2, the two sigmoids never both
    saturate near 1 anywhere inside the window, so the returned weight
    peaks well below 1 even at the window center. Keep softness well below
    ~1/3 of the window half-width to get a proper near-1 plateau inside.

    PRECISION: mass/low/high/softness live at O(100) GeV, so float32's ULP
    there is ~1e-5 GeV (see the discussion this was written for). For a
    `softness` anywhere near or below that scale, `mass - low` (a
    subtraction of two nearly-equal float32 numbers once mass_low/mass_high
    - which are TRAINABLE and move during optimization - pass close to a
    real event's mass) suffers catastrophic cancellation: the result is
    dominated by float32 rounding noise rather than the true distance from
    the boundary, and dividing that noise by a tiny `softness` can produce
    an arbitrarily large, spurious gradient contribution from a single
    event. Computing the subtraction and sigmoid in float64 (~1e-14 GeV
    ULP at this magnitude - 9 orders of magnitude better) removes this
    failure mode entirely; the result is cast back to float32 before
    returning so nothing downstream (or upstream - `tf.cast` is
    differentiable, so gradients flow back through it automatically) needs
    to know float64 was ever involved. This costs a handful of extra bytes
    per event for a few length-`chunk_size` intermediate vectors - utterly
    negligible next to the classifier's own per-chunk activation memory.
    """
    mass64 = tf.cast(tf.convert_to_tensor(mass), tf.float64)
    low64 = tf.cast(low, tf.float64)
    high64 = tf.cast(high, tf.float64)
    softness64 = tf.cast(softness, tf.float64)
    lower_edge = tf.sigmoid((mass64 - low64) / softness64)
    upper_edge = tf.sigmoid((high64 - mass64) / softness64)
    return tf.cast(lower_edge * upper_edge, tf.float32)


# ---------------------------------------------------------------------------
# Cross-entropy variants for the Stage-B classifier regularizer
# ---------------------------------------------------------------------------
def grouped_binary_cross_entropy(
    y, logits, signal_class, n_classes, normalize_background=False
):
    """
    Per-sample cross-entropy for the *collapsed* signal-vs-background problem,
    computed from the n_classes-way classifier logits.

    Motivation: the classifier is pretrained with a genuine n-class CE, but
    the Asimov significance downstream only ever distinguishes ONE signal
    class from "everything else". Keeping a 4-class CE as the Stage-B
    regularizer therefore penalizes the classifier for re-arranging
    probability mass *within* the background block, even though such a
    re-arrangement is completely invisible to Z. Collapsing the CE to the
    same 2-class view the significance uses removes that spurious pull.

    DEFAULT (normalize_background=False). With one-hot truth labels p and
    softmax predictions q, and (for the signal_class=3, n_classes=4 case)
    the background block {0, 1, 2}:

        S = (p_0 + p_1 + p_2) * log(q_0 + q_1 + q_2) + p_3 * log(q_3)
        loss = -S

    generalized to arbitrary `signal_class` / `n_classes`: the background
    block is every class index != signal_class.

    Collapsing by SUMMING is what keeps the 2-class problem properly
    normalized. Both collapsed vectors remain probability distributions:

        p_bkg = sum_{i in bkg} p_i = 1 for a background event, 0 for signal
        Q_bkg = sum_{i in bkg} q_i = 1 - q_signal

    so p_bkg + p_sig = 1 and Q_bkg + Q_sig = 1, and this reduces to the
    ordinary binary cross-entropy on (Q_bkg, Q_sig). Per event:

        background event:  loss = -log(Q_bkg)
        signal event:      loss = -log(q_signal)

    OPTIONAL VARIANT (normalize_background=True) implements instead

        S = (p_0+p_1+p_2)/3 * log((q_0+q_1+q_2)/3) + p_3 * log(q_3)

    i.e. it AVERAGES over the n_bkg background classes rather than summing.
    Kept only for reproducing that exact formula; it is not recommended,
    because averaging breaks the normalization that summing preserves -
    neither (1/n_bkg, 0) nor (Q_bkg/n_bkg, q_s) sums to 1. Concretely it
    does two things:

      * the "/n_bkg" INSIDE the log is an additive constant, -log(n_bkg)/n_bkg.
        Zero gradient, so it cannot affect training - but it puts a floor on
        the reported loss (a perfectly-classified background event is charged
        +log(3)/3 = 0.366 instead of 0), which makes the logged CE
        incomparable to Stage A's.
      * the "/n_bkg" OUTSIDE the log scales the background LABEL from 1 to
        1/n_bkg while signal keeps label 1. This one does have a gradient: it
        multiplies the learned signal-to-background odds by exactly n_bkg
        everywhere, so a region that is genuinely 25% signal gets scored 0.50
        (for n_bkg=3). If background really should count less, express that
        in class_weight_scheme where it is explicit and tunable, not in the
        loss algebra where it is invisible.

    Numerics: everything is computed in log-space from the raw logits via
    reduce_logsumexp, so no softmax is ever materialized and no epsilon
    clipping is needed - log(Q_bkg) and log(q_s) stay finite even when the
    corresponding probability underflows float32.
    """
    logits = tf.cast(logits, tf.float32)
    y = tf.cast(tf.reshape(y, [-1]), tf.int32)

    signal_class = int(signal_class)
    n_classes = int(n_classes)
    bkg_indices = [i for i in range(n_classes) if i != signal_class]
    n_bkg = len(bkg_indices)
    if n_bkg < 1:
        raise ValueError(
            f"grouped_binary_cross_entropy needs at least one background class "
            f"(got n_classes={n_classes}, signal_class={signal_class})."
        )

    # log(sum_i exp(l_i)) -- the softmax normalizer, computed stably.
    log_norm = tf.reduce_logsumexp(logits, axis=1)

    # log q_s and log(sum_{i in bkg} q_i), both still in log-space.
    log_q_sig = logits[:, signal_class] - log_norm
    bkg_logits = tf.gather(logits, tf.constant(bkg_indices, dtype=tf.int32), axis=1)
    log_Q_bkg = tf.reduce_logsumexp(bkg_logits, axis=1) - log_norm

    is_sig = tf.cast(tf.equal(y, signal_class), tf.float32)
    is_bkg = 1.0 - is_sig

    if normalize_background:
        # (sum p_i)/n_bkg * log((sum q_i)/n_bkg)
        coeff_bkg = is_bkg / tf.cast(n_bkg, tf.float32)
        log_arg_bkg = log_Q_bkg - tf.math.log(tf.cast(n_bkg, tf.float32))
    else:
        coeff_bkg = is_bkg
        log_arg_bkg = log_Q_bkg

    S = coeff_bkg * log_arg_bkg + is_sig * log_q_sig
    return -S


def make_ce_loss_fn(ce_mode, signal_class, n_classes):
    """
    Factory for the per-sample cross-entropy tracked during Stage B for
    MONITORING only (ce_history/val_ce_history) - it never enters any
    Stage-B loss or gradient (the old `lambda_ce` regularizer term has been
    removed as outdated). Returns a callable
    `fn(y, logits) -> per-sample loss of shape (batch,)`.

    ce_mode:
      "binary_plain"   - DEFAULT. The collapsed 2-class CE obtained by
                         SUMMING the background probabilities:
                             S = (sum_{i in bkg} p_i) * log(sum_{i in bkg} q_i)
                                 + p_s * log(q_s),   CE = -S
                         Keeps the collapsed problem normalized, so it is
                         exactly the binary CE on (Q_bkg, Q_sig) and its
                         value is directly comparable to Stage A's.
      "multiclass"     - the original n-class sparse CE (what Stage A uses).
                         Keeps pulling the classifier towards its full
                         n-way solution, including distinctions the
                         significance cannot see.
      "binary_grouped" - the same collapse but AVERAGING over the n_bkg
                         background classes (an extra 1/n_bkg inside and
                         outside the log). Not recommended - see
                         grouped_binary_cross_entropy's docstring; the outer
                         1/n_bkg tilts the learned odds by a factor n_bkg.
    """
    ce_mode = (ce_mode or "binary_plain").lower()

    if ce_mode == "multiclass":
        return tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

    if ce_mode in ("binary_grouped", "binary_plain"):
        normalize_background = ce_mode == "binary_grouped"

        def _fn(y, logits):
            return grouped_binary_cross_entropy(
                y, logits, signal_class, n_classes,
                normalize_background=normalize_background,
            )

        return _fn

    raise ValueError(
        f"training.ce_mode must be one of "
        f"('multiclass', 'binary_grouped', 'binary_plain'), got '{ce_mode}'."
    )


def _default_shuffle_buffer(n, batch_size, shuffle_buffer):
    if shuffle_buffer is not None:
        return shuffle_buffer
    return min(n, max(batch_size * 50, 200_000))


# Define custom dataset
def make_dataset(X, y, sample_weights, no_absolute_weights=None, batch_size=1024, shuffle=False, seed=None, shuffle_buffer=None):
    """
    TF equivalent of the CustomDataset + DataLoader combo. Returns a
    tf.data.Dataset that yields the same tuple structure as the original
    PyTorch DataLoader:
        (X_batch, y_batch, weights_batch, weights_batch_no)  if no_absolute_weights is given
        (X_batch, y_batch, weights_batch)                    otherwise

    shuffle_buffer: bounded shuffle buffer for large datasets (defaults to
    min(n, max(batch_size * 50, 200_000)) instead of the full dataset).
    """
    if no_absolute_weights is not None:
        ds = tf.data.Dataset.from_tensor_slices((X, y, sample_weights, no_absolute_weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y, sample_weights))

    if shuffle:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        buf = _default_shuffle_buffer(n, batch_size, shuffle_buffer)
        ds = ds.shuffle(buffer_size=buf, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_memmap_dataset(
    X, y, sample_weights, no_absolute_weights=None,
    batch_size=1024, shuffle=False, seed=None, shuffle_buffer=None,
):
    """
    Build a tf.data.Dataset from numpy/memmap arrays without copying the
    full feature matrix into TensorFlow. Only the current batch is read
    from disk and cast to float32.
    """
    n = int(X.shape[0])
    feature_dim = int(X.shape[1])
    buf = _default_shuffle_buffer(n, batch_size, shuffle_buffer)
    has_no_abs = no_absolute_weights is not None

    def _batch_from_indices(indices):
        idx = indices.numpy()
        Xb = np.asarray(X[idx], dtype=np.float32)
        yb = np.asarray(y[idx], dtype=np.int64).reshape(-1)
        wb = np.asarray(sample_weights[idx], dtype=np.float32).reshape(-1)
        # print(f"wb: {wb.shape}")
        if has_no_abs:
            wb2 = np.asarray(no_absolute_weights[idx], dtype=np.float32).reshape(-1)
            return Xb, yb, wb, wb2
        return Xb, yb, wb

    if has_no_abs:
        out_types = (tf.float32, tf.int64, tf.float32, tf.float32)
        out_shapes = (
            tf.TensorShape([None, feature_dim]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
        )
    else:
        out_types = (tf.float32, tf.int64, tf.float32)
        out_shapes = (
            tf.TensorShape([None, feature_dim]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
        )

    ds = tf.data.Dataset.range(n)
    if shuffle:
        ds = ds.shuffle(buffer_size=buf, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)

    def _fetch_and_set_shape(indices):
        batch = tf.py_function(_batch_from_indices, [indices], out_types)
        return tuple(tf.ensure_shape(t, s) for t, s in zip(batch, out_shapes))

    ds = ds.map(
        _fetch_and_set_shape,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# A minimal re-implementation of torch's ReduceLROnPlateau for use inside a
# manual training loop (Keras's callback of the same name only works with
# model.fit).
class ReduceLROnPlateau:
    """
    Re-implementation of torch.optim.lr_scheduler.ReduceLROnPlateau, matched
    to PyTorch's DEFAULTS (mode='min', threshold=1e-4, cooldown=0, eps=1e-8),
    but with a SIGN-ROBUST relative-improvement test instead of PyTorch's own
    `threshold_mode='rel'` formula:

      - is_better: an epoch only counts as an "improvement" (and resets the
        bad-epoch counter) if
            metric < best - threshold * abs(best)
        i.e. it must beat the running best by at least `threshold` as a
        RELATIVE fraction of the running best's MAGNITUDE, not just be
        infinitesimally smaller.

        For a non-negative `best` (e.g. Stage A's CE loss), abs(best) ==
        best, so this reduces to `metric < best * (1 - threshold)` -
        PyTorch's own `threshold_mode='rel'` formula, byte-for-byte -
        so pretraining's scheduling behavior is unaffected by this change.

        For a NEGATIVE `best` (e.g. Stage B's val_scalar_loss, dominated by
        -Z), PyTorch's multiplicative formula `best * (1 - threshold)`
        shifts the wrong way (it makes "improvement" easier to claim, not
        harder, since multiplying a negative number by a value < 1 makes it
        LESS negative). Subtracting a magnitude-scaled margin instead always
        tightens the bar in the correct direction regardless of best's sign
        - this is what actually makes the scheduler safe to reuse for both
        Stage A (loss >= 0) and Stage B (loss can be < 0) without silently
        rewarding a step that made Z worse.

        Special case: on the very first call, `best` is +inf, so
        `best - threshold*abs(best)` would be `inf - inf = nan` -
        `metric < nan` is always False in Python/numpy, which would make the
        very first epoch's "improvement" fail to register. Guarded
        explicitly below (`self.best == np.inf` -> any metric is better).

      - the LR is only actually reduced if the drop exceeds `eps`
        (old_lr - new_lr > eps), matching PyTorch's own negligible-change
        guard in `_reduce_lr`.
      - `cooldown` (epochs to wait, ignoring bad epochs, right after a
        reduction) is not implemented, matching PyTorch's own default of 0 -
        add it explicitly if a non-zero cooldown is ever needed.
    """
    def __init__(self, optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6,
                 threshold=1e-4, eps=1e-8):
        assert mode == 'min'
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.eps = eps
        self.best = np.inf
        self.num_bad_epochs = 0

    def step(self, metric):
        is_better = (self.best == np.inf) or (metric < self.best - self.threshold * abs(self.best))
        if is_better:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            current_lr = float(self.optimizer.learning_rate.numpy())
            new_lr = max(current_lr * self.factor, self.min_lr)
            if current_lr - new_lr > self.eps:
                self.optimizer.learning_rate.assign(new_lr)
            self.num_bad_epochs = 0

    def get_config(self):
        return {
            'factor': self.factor,
            'patience': self.patience,
            'min_lr': self.min_lr,
            'threshold': self.threshold,
            'eps': self.eps,
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
        }

    def load_config(self, config):
        self.factor = config['factor']
        self.patience = config['patience']
        self.min_lr = config['min_lr']
        self.threshold = config.get('threshold', 1e-4)
        self.eps = config.get('eps', 1e-8)
        self.best = config['best']
        self.num_bad_epochs = config['num_bad_epochs']


# Training and evaluation functions
@tf.function
def _train_step(model, optimizer, X_batch, y_batch, weights_batch, weights_batch_no, loss_fn, weight_decay=0.0):
    """
    Single, @tf.function-compiled training step (graph mode instead of
    eager). Defined at module level so it is traced once per (model,
    optimizer, loss_fn, weight_decay) combination instead of being
    retraced (or run eagerly) on every train_one_epoch() call.

    `weight_decay` reproduces PyTorch's `torch.optim.Adam(weight_decay=...)`
    behaviour exactly: grad <- grad + weight_decay * param, applied BEFORE
    the Adam moment updates ("coupled" L2). This is deliberately NOT done
    via `tf.keras.optimizers.Adam(weight_decay=...)` - since TF 2.11 that
    argument implements *decoupled* (AdamW-style) weight decay, which is a
    different regularizer even for the same numeric value. `optimizer`
    passed in here must therefore be a plain `Adam(learning_rate=...)`
    with no weight_decay kwarg set on it.
    """
    with tf.GradientTape() as tape:
        y_pred = model(X_batch, training=True)
        loss = loss_fn(y_batch, y_pred)  # per-sample loss, shape (batch,)
        weighted_loss = tf.reduce_sum(loss * weights_batch) / tf.reduce_sum(weights_batch)

    grads = tape.gradient(weighted_loss, model.trainable_variables) # 12 Variables to train

    if weight_decay:
        grads = [
            g + weight_decay * w if g is not None else g
            for g, w in zip(grads, model.trainable_variables)
        ]

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    weighted_loss_no_abs = tf.reduce_sum(loss * weights_batch_no) / tf.reduce_sum(weights_batch_no)

    # compute weighted accuracy
    correct = tf.cast(tf.argmax(y_pred, axis=1, output_type=y_batch.dtype) == y_batch, tf.float32) # casts bool to float
    weighted_acc = tf.reduce_sum(correct * weights_batch) / tf.reduce_sum(weights_batch)

    return weighted_loss, weighted_acc, weighted_loss_no_abs


def train_one_epoch(model, optimizer, dataset, loss_fn, epoch, weight_decay=0.0):
    # tf.keras.metrics.Mean accumulates on-device -> no per-batch .numpy()
    # call (and therefore no per-batch GPU->CPU sync) is needed anymore.
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.Mean()
    loss_no_abs_metric = tf.keras.metrics.Mean()

    progress_bar = tqdm(dataset, desc=f"Epoch {epoch} [Training]", leave=False)
    for step, (X_batch, y_batch, weights_batch, weights_batch_no) in enumerate(progress_bar):
        weighted_loss, weighted_acc, weighted_loss_no_abs = _train_step(
            model, optimizer, X_batch, y_batch, weights_batch, weights_batch_no, loss_fn,
            weight_decay=weight_decay,
        )

        loss_metric.update_state(weighted_loss)
        acc_metric.update_state(weighted_acc)
        loss_no_abs_metric.update_state(weighted_loss_no_abs)

        # Only refresh the progress bar (and therefore only sync to CPU)
        # every 50 steps instead of on every single batch.
        if step % 50 == 0:
            progress_bar.set_postfix({
                'Loss': f'{loss_metric.result().numpy():.4f}',
                'Acc': f'{acc_metric.result().numpy():.4f}',
                'Loss_no_abs': f'{loss_no_abs_metric.result().numpy():.4f}',
            })

    return (
        float(loss_metric.result().numpy()),
        float(acc_metric.result().numpy()),
        float(loss_no_abs_metric.result().numpy()),
    )


def evaluate(model, dataset, loss_fn, epoch):
    val_losses = []
    val_accs = []
    progress_bar = tqdm(dataset, desc=f"Epoch {epoch} [Validation]", leave=False)
    for X_batch, y_batch, weights_batch in progress_bar:
        y_pred = model(X_batch, training=False)
        loss = loss_fn(y_batch, y_pred)
        weighted_loss = tf.reduce_sum(loss * weights_batch) / tf.reduce_sum(weights_batch)

        # compute weighted accuracy
        correct = tf.cast(tf.argmax(y_pred, axis=1, output_type=y_batch.dtype) == y_batch, tf.float32)
        weighted_acc = tf.reduce_sum(correct * weights_batch) / tf.reduce_sum(weights_batch)

        val_losses.append(float(weighted_loss.numpy()))
        val_accs.append(float(weighted_acc.numpy()))

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{float(weighted_loss.numpy()):.4f}',
            'Acc': f'{float(weighted_acc.numpy()):.4f}'
        })

    return np.mean(val_losses), np.mean(val_accs)


def compute_confusion_counts_from_dataset(model, dataset, n_classes):
    """
    Weighted + raw (unweighted) confusion-matrix counts, accumulated over
    an ENTIRE tf.data.Dataset of (X, y, weights[, ...anything else]) batches
    - the SAME batches train_one_epoch()/evaluate() already iterate over,
    so this can be called directly on train_dataset/val_dataset (Stage A)
    or an equivalent batched dataset (Stage B) with no separate data
    pipeline. Only the first three elements of each batch tuple are used
    (X, y, weights), so it works whether or not a 4th column (e.g.
    cw_train_no_abs) is present.

    Returns (cm_weighted, cm_raw), each shape (n_classes, n_classes), where
    cm[i, j] = the (weighted, or raw event) count of events whose TRUE
    class is i and whose PREDICTED (arg-max logit) class is j. This is
    enough to reconstruct, for any class c treated one-vs-rest:
        TP = cm[c, c]
        FP = sum(cm[:, c]) - TP    (predicted c, but wasn't)
        FN = sum(cm[c, :]) - TP    (was c, but predicted something else)
        TN = total - TP - FP - FN
    and therefore accuracy/precision/recall/etc. - see
    append_confusion_counts_row, which does exactly that.
    """
    cm_weighted = np.zeros((n_classes, n_classes), dtype=np.float64)
    cm_raw = np.zeros((n_classes, n_classes), dtype=np.int64)
    for batch in dataset:
        X_batch, y_batch, w_batch = batch[0], batch[1], batch[2]
        logits = model(X_batch, training=False)
        preds = tf.argmax(logits, axis=1, output_type=y_batch.dtype).numpy().astype(np.int64)
        y_np = y_batch.numpy().astype(np.int64)
        w_np = w_batch.numpy().astype(np.float64)
        flat_idx = y_np * n_classes + preds
        cm_weighted += np.bincount(flat_idx, weights=w_np, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        cm_raw += np.bincount(flat_idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(np.int64)
    return cm_weighted, cm_raw


def _confusion_counts_columns(class_names):
    """Column names for append_confusion_counts_row's wide, one-row-per-
    epoch format: 10 columns per class (TP/FP/TN/FN, both weighted and
    raw, plus precision/recall), then 2 overall-accuracy columns."""
    cols = ["epoch"]
    for c in class_names:
        cols += [
            f"{c}_TP", f"{c}_FP", f"{c}_TN", f"{c}_FN",
            f"{c}_TP_raw", f"{c}_FP_raw", f"{c}_TN_raw", f"{c}_FN_raw",
            f"{c}_precision", f"{c}_recall",
        ]
    cols += ["overall_accuracy_weighted", "overall_accuracy_raw"]
    return cols


def append_confusion_counts_row(path, epoch, cm_weighted, cm_raw, class_names=None):
    """
    Appends ONE ROW (for `epoch`) to a single, run-long CSV, instead of
    writing a brand-new file every time this is called - one-vs-rest
    TP/FP/TN/FN (both weighted - using the same per-event weights the
    accuracy metric already prints every epoch - and raw unweighted event
    counts), precision/recall, and overall accuracy, all flattened into
    that one row's columns (see _confusion_counts_columns). Writing a
    separate file per checkpoint epoch (the original version of this
    function did that) is wasteful for something this cheap and small; one
    growing table is both fewer files on disk and directly loadable as a
    single pandas.read_csv() per split, indexed by epoch, to regenerate
    accuracy/precision/recall/efficiency-vs-epoch plots later.

    cm[i, j] = count of true-class-i events predicted as class j (see
    compute_confusion_counts_from_dataset/_from_arrays). The header is
    written once, the first time `path` doesn't yet exist; every
    subsequent call (any epoch, any run reusing the same out_dir) appends
    a new row underneath it.
    """
    n = cm_weighted.shape[0]
    class_names = list(class_names) if class_names is not None else [str(i) for i in range(n)]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    total_w = float(cm_weighted.sum())
    total_raw = int(cm_raw.sum())

    row = [epoch]
    for i in range(n):
        tp = float(cm_weighted[i, i])
        fp = float(cm_weighted[:, i].sum() - tp)
        fn = float(cm_weighted[i, :].sum() - tp)
        tn = total_w - tp - fp - fn
        tp_r = int(cm_raw[i, i])
        fp_r = int(cm_raw[:, i].sum() - tp_r)
        fn_r = int(cm_raw[i, :].sum() - tp_r)
        tn_r = total_raw - tp_r - fp_r - fn_r
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        row += [tp, fp, tn, fn, tp_r, fp_r, tn_r, fn_r, precision, recall]
    row += [
        np.trace(cm_weighted) / total_w if total_w > 0 else float("nan"),
        np.trace(cm_raw) / total_raw if total_raw > 0 else float("nan"),
    ]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_confusion_counts_columns(class_names))
        writer.writerow(row)
    print(f"[INFO] Appended epoch {epoch} confusion-matrix counts to: {path}")


STEP_HISTORY_COLUMNS = [
    "epoch", "step_idx", "n_steps_in_epoch", "updated_classifier", "updated_gato",
    "sig_loss", "penalty", "ce", "ce_no_abs", "Z", "total_loss",
]


def append_step_history_row(path, epoch, step_idx, n_steps_in_epoch, updated_classifier, updated_gato,
                             sig_loss, penalty, ce, ce_no_abs, Z, total_loss):
    """Appends ONE ROW per training step (i.e. per entry of `step_plan` in
    train_complete_model - see that function's docstring) to a single,
    run-long CSV under out_dir - same append-only, header-written-once
    pattern as append_confusion_counts_row. Under the default 1:1
    schedule this is exactly one row per epoch (step_idx=0,
    n_steps_in_epoch=1, updated_classifier=updated_gato=True), so nothing
    is lost or duplicated relative to loss_history/penalty_history/
    ce_history for that common case - it only adds resolution when a
    non-1:1 schedule is in use, since epoch-granular plots can always be
    recovered later by keeping only the last step_idx of each epoch.

    ce_no_abs: the same per-step CE, reduced with the TRUE (signed,
    non-abs) train weights instead of the abs-valued ones - see
    finetune_train_step_chunked's cw_all_no_abs docstring. NaN if no
    cw_train_no_abs was passed to train_complete_model() for this run."""
    write_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(STEP_HISTORY_COLUMNS)
        writer.writerow([
            epoch, step_idx, n_steps_in_epoch, bool(updated_classifier), bool(updated_gato),
            float(sig_loss), float(penalty), float(ce), float(ce_no_abs), float(Z), float(total_loss),
        ])


def pretrain(classifier, raw_data, class_weights, training_config, profile_out_path=None, out_dir=None,
             classes=None):
    print("\n[STAGE A] Pretraining classifier with weighted cross-entropy\n")

    # load config (explicit keys, not positional .values(), so extra keys
    # like checkpoint_every can be added without breaking this unpacking):
    pretrain_cfg = training_config["pretrain"]
    n_epochs = pretrain_cfg["epochs"]
    batch_size = pretrain_cfg["batch_size"]
    early_stopping_patience = pretrain_cfg["early_stopping_patience"]
    scheduler = pretrain_cfg["scheduler"]
    pretrain_lr,pretrain_wd = training_config["pretrain_optimizer"].values()

    shuffle_buffer = training_config.get("pretrain", {}).get("shuffle_buffer", None)

    # Periodic checkpointing: additionally save weights every
    # `checkpoint_every` epochs (on top of the best-val-loss weights this
    # function already saves at the end). Purely for diagnostics, e.g. to
    # study how the pretraining stage affects finetuning results.
    # Set to 0/omit to disable (default).
    #
    # THREE things are written at that same cadence now, into three
    # separate subfolders of out_dir:
    #   pretrain_checkpoints/          - UNCHANGED: the current epoch's
    #       weights, blindly, exactly as before.
    #   pretrain_checkpoints_best/     - NEW: whatever weights are
    #       currently the best-val-loss weights SEEN SO FAR (i.e. a copy of
    #       best_weights_path at that point), not the current epoch's own
    #       weights - so this is a "best-known-so-far" checkpoint trail,
    #       distinct from the blind one above (which can be from a worse
    #       epoch than the running best). Skipped for any checkpoint epoch
    #       before the first val-loss improvement (best_weights_path is
    #       still None then).
    #   pretrain_checkpoints/confusion_counts_{train,val}.csv - NEW: one
    #       growing table per split, ONE ROW PER checkpoint epoch (not one
    #       file per epoch - see append_confusion_counts_row), holding
    #       one-vs-rest TP/FP/TN/FN (+precision/recall) per class plus
    #       overall accuracy, for the CURRENT epoch's weights - see
    #       compute_confusion_counts_from_dataset/append_confusion_counts_row
    #       above. Each row's "epoch" column lines up with the blind
    #       checkpoint of the same epoch, so plots like accuracy-vs-epoch
    #       can be regenerated later (e.g. via pandas.read_csv) without
    #       rerunning the model.
    checkpoint_every = pretrain_cfg.get("checkpoint_every", 0) or 0
    checkpoint_dir = None
    best_checkpoint_dir = None
    if checkpoint_every and out_dir is not None:
        checkpoint_dir = os.path.join(out_dir, "pretrain_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_checkpoint_dir = os.path.join(out_dir, "pretrain_checkpoints_best")
        os.makedirs(best_checkpoint_dir, exist_ok=True)

    n_classes = len(classes) if classes is not None else int(
        np.unique(np.asarray(raw_data["y_train"][:])).size
    )
    class_names = list(classes) if classes is not None else None

    # Read batches directly from memmap/numpy on CPU; never materialize the
    # full feature matrix as a single TF tensor on GPU or RAM.
    # seeds.shuffle if present (separate-seeds scheme - see pretrain_config.yaml/
    # combined_optimization_config.yaml), else the older flat random_seed key,
    # so this still works unchanged against a pre-refactor training_config.yaml.
    shuffle_seed = training_config.get("seeds", {}).get("shuffle", training_config.get("random_seed"))
    train_dataset = make_memmap_dataset(
        raw_data["X_train"], raw_data["y_train"],
        class_weights["cw_train"], class_weights["cw_train_no_abs"],
        batch_size=batch_size, shuffle=True, seed=shuffle_seed,
        shuffle_buffer=shuffle_buffer,
    )
    val_dataset = make_memmap_dataset(
        raw_data["X_val"], raw_data["y_val"], class_weights["cw_val"],
        batch_size=batch_size, shuffle=False,
    )


    loss_fn_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    # NOTE: weight_decay is intentionally NOT passed to Adam() here - see
    # the docstring on _train_step(). It is applied manually, coupled into
    # the gradient, to match torch.optim.Adam(weight_decay=...) exactly.
    pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=pretrain_lr, epsilon=1e-8)
    scheduler = ReduceLROnPlateau(pretrain_optimizer, mode=scheduler["mode"], factor=scheduler["factor"], patience=scheduler["patience"], min_lr=scheduler["min_lr"])

    # Collect Histories:
    train_loss_history = []
    val_loss_history = []
    # NEW: train_one_epoch() has always computed weighted_loss_no_abs (the
    # SAME per-sample CE, just weighted by cw_train_no_abs - i.e.
    # true_class_weights, the signed rel_w, INSTEAD of cw_train - i.e.
    # whatever class_weight_scheme selected, class_weights_abs or
    # class_weights_only_positive) - it just used to be silently discarded
    # (`_`) below. val_loss above is already computed on the signed
    # class_weights_val (dataset_prep never abs's the val/test split's
    # weights - see build_training_dataset), so train_loss_no_abs is the
    # APPLES-TO-APPLES train-side counterpart to val_loss: both use signed
    # weights, unlike the primary train_loss/val_loss pair, which mixes an
    # abs (or only-positive)-weighted train curve against a signed-weighted
    # val curve. See pretrain.py's second loss plot.
    train_loss_no_abs_history = []


    prof_cfg = training_config.get("training", {}).get("profiling", {}) or {}
    profiler = PhaseProfiler(
        enabled=bool(prof_cfg.get("enabled", True)),
        sync=bool(prof_cfg.get("device_sync", True)),
        label="Stage A (pretraining)",
    )

    best_loss = np.inf
    best_weights_path = None
    counter = 0

    # Best-checkpoint tracking restricted to the last 20%/40% of epochs, in
    # addition to the plain best-overall tracking above - same rationale/
    # convention as train_complete_model's identical block (Stage B), and
    # gato_categorization_merged.py's update_best() (its
    # `epoch_idx >= args.epochs * 0.8` guard): a classifier whose lowest
    # val_loss happened to land early doesn't hide a more representative
    # "best of the converged tail" checkpoint from evaluation. Windows are
    # measured against the CONFIGURED n_epochs (fixed here, before the loop
    # starts), not however many epochs the run actually completes, since
    # early_stopping_patience can cut it short. Written permanently under
    # out_dir/pretrain_best_checkpoints/ whenever out_dir is given; silently
    # skipped (no behavior change) otherwise.
    last20_start_epoch = n_epochs * 0.8
    last40_start_epoch = n_epochs * 0.6
    best_loss_last20 = np.inf
    best_loss_last40 = np.inf
    best_checkpoints_dir = os.path.join(out_dir, "pretrain_best_checkpoints") if out_dir is not None else None
    if best_checkpoints_dir is not None:
        os.makedirs(best_checkpoints_dir, exist_ok=True)

    for epoch in range(n_epochs):
        profiler.start_epoch(epoch)
        with profiler.phase("train_steps"):
            train_loss, train_acc, train_loss_no_abs = train_one_epoch(
                classifier, pretrain_optimizer, train_dataset, loss_fn_ce, epoch,
                weight_decay=pretrain_wd,
            )
        with profiler.phase("validation"):
            val_loss, val_acc = evaluate(classifier, val_dataset, loss_fn_ce, epoch)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_loss_no_abs_history.append(train_loss_no_abs)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            if best_weights_path is None:
                fd, best_weights_path = tempfile.mkstemp(suffix=".weights.h5")
                os.close(fd)
            classifier.save_weights(best_weights_path)
            counter = 0
            if best_checkpoints_dir is not None:
                save_with_retry(
                    classifier.save_weights,
                    os.path.join(best_checkpoints_dir, "classifier_pretrain_best_overall.weights.h5"),
                    label="Stage A best-overall checkpoint",
                )
        else:
            counter += 1

        # Windowed "best of the last 20%/40% of epochs" checkpoints -
        # independent of (and in addition to) the best-overall tracking
        # above. The last-20% window is a subset of the last-40% window, so
        # both can update in the same epoch - expected, not a bug.
        if best_checkpoints_dir is not None:
            if epoch >= last40_start_epoch and val_loss < best_loss_last40:
                best_loss_last40 = val_loss
                save_with_retry(
                    classifier.save_weights,
                    os.path.join(best_checkpoints_dir, "classifier_pretrain_best_last40pct.weights.h5"),
                    label="Stage A best-last-40%-of-epochs checkpoint",
                )
            if epoch >= last20_start_epoch and val_loss < best_loss_last20:
                best_loss_last20 = val_loss
                save_with_retry(
                    classifier.save_weights,
                    os.path.join(best_checkpoints_dir, "classifier_pretrain_best_last20pct.weights.h5"),
                    label="Stage A best-last-20%-of-epochs checkpoint",
                )

        profiler.end_epoch()
        print(f"[Stage A][{epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_loss_no_abs={train_loss_no_abs:.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}" + profiler.epoch_line())

        # Periodic snapshot, independent of early stopping / best-loss tracking.
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"classifier_pretrain_epoch{epoch:03d}.weights.h5")
            save_with_retry(classifier.save_weights, ckpt_path, label=f"Stage A periodic checkpoint (epoch {epoch:03d})")
            print(f"[Stage A] Saved periodic checkpoint: {ckpt_path}")

            # NEW: best-so-far weights (a copy of the running best_weights_path,
            # NOT the current epoch's own weights - see docstring above).
            if best_weights_path is not None:
                best_ckpt_path = os.path.join(
                    best_checkpoint_dir, f"classifier_pretrain_best_upto_epoch{epoch:03d}.weights.h5"
                )
                save_with_retry(
                    lambda p: shutil.copy(best_weights_path, p), best_ckpt_path,
                    label=f"Stage A best-so-far checkpoint (epoch {epoch:03d})",
                )
                print(f"[Stage A] Saved best-so-far checkpoint: {best_ckpt_path}")
            else:
                print(
                    f"[Stage A] No best-so-far checkpoint saved at epoch {epoch:03d}: "
                    f"val loss has not improved yet."
                )

            # Confusion-matrix counts (TP/FP/TN/FN etc.) for the CURRENT
            # epoch's weights, on both train and val - one growing table
            # per split (out_dir/pretrain_checkpoints/confusion_counts_
            # {train,val}.csv), one row appended per checkpoint epoch,
            # instead of a new file every time - see
            # append_confusion_counts_row's docstring.
            cm_w_train, cm_raw_train = compute_confusion_counts_from_dataset(classifier, train_dataset, n_classes)
            append_confusion_counts_row(
                os.path.join(checkpoint_dir, "confusion_counts_train.csv"),
                epoch, cm_w_train, cm_raw_train, class_names=class_names,
            )
            cm_w_val, cm_raw_val = compute_confusion_counts_from_dataset(classifier, val_dataset, n_classes)
            append_confusion_counts_row(
                os.path.join(checkpoint_dir, "confusion_counts_val.csv"),
                epoch, cm_w_val, cm_raw_val, class_names=class_names,
            )

        if counter >= early_stopping_patience:
            print(f"[Stage A] Early stopping at epoch {epoch}")
            break

    profiler.report(n_rows=int(raw_data["X_train"].shape[0]))
    profiler.save(profile_out_path, extra={
        "stage": "A",
        "n_train_rows": int(raw_data["X_train"].shape[0]),
        "n_val_rows": int(raw_data["X_val"].shape[0]),
        "n_features": int(raw_data["X_train"].shape[1]),
        "feature_dtype": str(getattr(raw_data["X_train"], "dtype", "unknown")),
        "batch_size": int(batch_size),
    })

    if best_weights_path is not None:
        classifier.load_weights(best_weights_path)
        os.remove(best_weights_path)

    return best_loss, train_loss_history, val_loss_history, train_loss_no_abs_history


def _chunk_bounds(n, chunk_size):
    """[(start, end), ...] slices covering range(n) in steps of chunk_size."""
    return [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]


def _read_chunk(X_all, y_all, w_all, a, b, prof=None):
    """
    Pulls rows [a:b] of the Stage-B arrays.

    When profiling is on the feature block is materialized here. Slicing a
    memmap (or a PermutedView hitting its contiguous fast path) can hand back
    a lazy view, in which case the actual disk read happens later, inside the
    forward pass, and its cost would be misattributed to compute instead of
    I/O. Forcing it here MOVES the read, it does not add one: TF performs
    exactly the same copy when it converts the array to a tensor. With
    profiling off, nothing is forced and the behaviour is unchanged.
    """
    Xc, yc, wc = X_all[a:b], y_all[a:b], w_all[a:b]
    if prof is not None and prof.enabled:
        Xc = np.ascontiguousarray(Xc)
        prof.add_bytes(Xc.nbytes)
    return Xc, yc, wc


@tf.function(reduce_retracing=True)
def covariance_damping_penalty(scale_tril):
    """Additive, lambda-weighted regularizer pushing each GMM category's
    covariance toward diagonal (axis-aligned) - a project-local addition,
    NOT part of gatohep.losses (unlike low_bkg_penalty/
    high_bkg_uncertainty_penalty above), since gatohep only exposes a
    fixed, non-trainable-weight damping baked into gato_gmm_model's own
    get_scale_tril() (models.py's cov_offdiag_damping constructor
    argument - a hard multiplicative constant applied every step
    regardless of any lambda, and not even exposed by this project's
    GMMFromScores.__init__, which never passes it through). This penalty
    is independent of and additional to that: it softly encourages
    whatever off-diagonal correlation survives the fixed damping toward
    exactly zero, at a strength `lam_cov` genuinely controls (0.0 == off),
    the same disable-by-zero convention as the other two regularizers.

    IMPORTANT - this is fundamentally different in character from
    low_bkg_penalty/high_bkg_uncertainty_penalty: those are nonlinear
    functions of AGGREGATED, chunk-summed yields (S_total/B_total/
    B_w2_total), so they need the gradient-caching trick documented in
    finetune_train_step_chunked's docstring. This penalty depends ONLY on
    the model's own parameters (scale_tril, derived from
    model.gato.unconstrained_L) - there is no per-event/per-chunk data
    dependence at all, so it must be computed and differentiated exactly
    ONCE per training step (like an ordinary weight-decay term), never
    folded into the per-chunk pseudo-loss - see its call site in
    finetune_train_step_chunked, added once after the chunk loop rather
    than accumulated chunk-by-chunk.

    scale_tril: (n_cats, dim, dim) lower-triangular Cholesky factor, as
    returned by gato_gmm_model.get_scale_tril().
    """
    diag_part = tf.linalg.diag(tf.linalg.diag_part(scale_tril))
    off_diag = scale_tril - diag_part
    return tf.reduce_sum(tf.square(off_diag))


def _chunk_forward(model, X, y, w, mass, training, rewt=None):
    """
    Thin, tf.function-wrapped call to model.chunk_yields, factored out so it
    can be reused, unchanged, both for the "forward-only" pass and the
    "forward+backward" pass of the chunked Stage-B step below.
    `reduce_retracing=True` avoids a graph retrace every time the *last*
    chunk of an epoch (which is typically smaller than `chunk_size`) is
    passed in.

    `mass`, if not None, is the per-event mass column for this chunk - only
    used (inside model.chunk_yields -> model.build_data_dict) when the
    model has soft mass-window weighting enabled. Pass None to disable it
    entirely (numerically identical to the pre-mass-window behavior).

    `rewt`, if not None, is the length-n_cats non-resonant-background
    reweight tensor (see end_to_end_helpers.compute_nonres_reweight_factors);
    forwarded unchanged to model.chunk_yields for every chunk within an
    epoch - it is recomputed only every nonres_reweight.recompute_every
    epochs in train_complete_model, so it is a fixed constant within any
    single call to finetune_train_step_chunked / evaluate_complete_model_chunked.
    """
    return model.chunk_yields(X, y, w, mass=mass, training=training, rewt=rewt)


def build_data_dict_chunked(model, X_all, y_all, w_all, chunk_size, mass_all=None):
    """
    Memory-safe replacement for `model.build_data_dict(X_all, y_all, w_all)`
    run on the *entire* dataset at once.

    Calling build_data_dict directly on the full dataset forces a single
    classifier forward pass over all N events simultaneously, which means
    all hidden-layer activations for N rows exist in GPU memory at the same
    time - this is exactly the kind of spike `chunk_size` in the training
    step does NOT protect against (it only chunks the Stage-B train step,
    not this periodic diagnostic call).

    Here we forward each chunk separately (no gradient tape needed - this
    is inference only), immediately pull the *small* per-event score
    vectors (dim = gato_dim, typically 2-4 floats/event, much smaller than
    the raw features or hidden activations) down to numpy/CPU, and
    concatenate. Peak GPU memory is therefore O(chunk_size) instead of
    O(len(X_all)), while the returned data_dict is numerically identical to
    the unchunked version.
    """
    n = X_all.shape[0]
    bounds = _chunk_bounds(n, chunk_size)
    # Bucket keys are whatever model.build_data_dict() produces ("signal" +
    # "background", OR "signal" + "background_nonres" + "background_other"
    # when nonres reweighting is enabled) - collected dynamically instead of
    # hardcoded so this stays correct either way.
    nn_out_chunks = {}
    weight_chunks = {}

    for (a, b) in bounds:
        mc = mass_all[a:b] if mass_all is not None else None
        dd, _, _ = model.build_data_dict(X_all[a:b], y_all[a:b], w_all[a:b], mass=mc, training=False)
        for proc, t in dd.items():
            # In mass_window position="after_gmm" the window factor is kept
            # separate from "weight" (it is applied to the per-event yields
            # instead); fold it in here so this diagnostic dict describes the
            # same effective per-event weight either way.
            wgt = t["weight"]
            if t.get("mass_weight", None) is not None:
                wgt = wgt * t["mass_weight"]
            nn_out_chunks.setdefault(proc, []).append(t["NN_output"].numpy())
            weight_chunks.setdefault(proc, []).append(wgt.numpy())

    with tf.device("/CPU:0"):
        data_dict = {
            proc: {
                "NN_output": tf.constant(np.concatenate(nn_out_chunks[proc], axis=0), dtype=tf.float32),
                "weight": tf.constant(np.concatenate(weight_chunks[proc], axis=0), dtype=tf.float32),
            }
            for proc in nn_out_chunks
        }
    return data_dict


def finetune_train_step_chunked(
    model, X_all, y_all, w_all, mass_all, chunk_size,
    classifier_optimizer, gato_optimizer,
    ce_loss_fn, lam_penalty, penalty_thr, dropout_in_finetune,
    cw_all=None, ce_weight_sum=None,
    cw_all_no_abs=None, ce_weight_sum_no_abs=None,
    penalty_in_gato_loss=True, penalty_in_classifier_loss=True,
    significance_in_classifier_loss=True,
    lam_unc=0.0, bkg_unc_threshold=0.2,
    lam_cov=0.0,
    update_classifier=True, update_gato=True, profiler=None,
    rewt=None,
):
    """
    Memory-efficient replacement for a single full-batch
        with tf.GradientTape():
            sig_loss, S, B, Z, B_w2, scores, logits = model.call(X_all, y_all, w_all)
            ...
            grads = tape.gradient(total_loss, trainable_vars)
    that gives *numerically the same gradients*, while never holding the
    forward-pass activations of the *entire* dataset in memory at once.

    -----------------------------------------------------------------------
    WHY NAIVE MINI-BATCHING DOES NOT WORK HERE
    -----------------------------------------------------------------------
    For an ordinary per-sample loss (e.g. cross-entropy), the total loss is
    a *sum* over samples, so splitting the data into mini-batches and
    averaging/accumulating per-batch gradients gives exactly (up to
    floating point) the same gradient as one full-batch pass. That is the
    standard justification for mini-batch SGD.

    The Asimov significance Z = asymptotic_significance(S, B) is different:
    it is a *nonlinear* function of the two *aggregated* yield vectors S, B
    (summed signal/background weights per category), not a per-sample
    quantity. If you were to chunk the data, compute Z_chunk on each
    chunk's own S_chunk/B_chunk, and backprop those independently, you
    would get the gradient of the WRONG loss - the gradient of
    "average significance of many small chunks" instead of "significance
    of the whole dataset". Since Z is concave-ish and yields grow with
    dataset size, Z_chunk-based gradients are systematically biased and
    would silently give you a mis-trained categorizer.

    -----------------------------------------------------------------------
    HOW THIS FUNCTION GETS AN EXACT GRADIENT ANYWAY
    -----------------------------------------------------------------------
    The key fact (see model.chunk_yields' docstring) is that S and B
    *themselves* decompose additively over events:
        S_total(theta) = sum_c S_c(theta),   B_total(theta) = sum_c B_c(theta)
    where `c` indexes chunks and theta = model parameters. Only the final
    step, Z = f(S_total, B_total), is nonlinear - and it only ever touches
    two small vectors of length n_cats, never the full per-event data.

    By the chain rule:
        dL/dtheta = (dL/dS_total) . dS_total/dtheta + (dL/dB_total) . dB_total/dtheta
                  = (dL/dS_total) . sum_c dS_c/dtheta + (dL/dB_total) . sum_c dB_c/dtheta
                  = sum_c [ (dL/dS_total).dS_c/dtheta + (dL/dB_total).dB_c/dtheta ]

    So if we treat dS = dL/dS_total and dB = dL/dB_total as FIXED constants
    (numbers, not functions of theta) - which is valid because they are
    evaluated at the true, fully-aggregated S_total/B_total - then for each
    chunk c we can define a per-chunk pseudo-loss
        chunk_loss_c = dS . S_c(theta) + dB . B_c(theta)
    whose gradient w.r.t. theta, summed over all chunks, is EXACTLY
    dL/dtheta. This is the standard "gradient caching" trick used for
    large-batch contrastive losses (e.g. CLIP/SimCLR-style training) and it
    applies here for the same structural reason: a nonlinear function of a
    linearly-decomposable statistic.

    Concretely, this function does 3 passes over the chunks each epoch:
      1. Forward-only pass (no gradient tape): accumulate S_total, B_total.
      2. A tiny GradientTape over just the 2 length-n_cats vectors
         S_total, B_total to get dS, dB analytically (cheap - no large
         activations involved).
      3. Forward+backward pass, one chunk at a time: recompute S_c, B_c
         (and the CE logits) under a GradientTape, form chunk_loss_c, get
         its gradient, and accumulate the gradient (not the loss!) across
         chunks. Only ONE chunk's activations are ever alive at a time.

    Step 3 needs to see IDENTICAL classifier outputs to step 1 for the
    accumulated gradient to be exactly right (not just approximately
    right). The only source of randomness in the classifier's forward pass
    is Dropout, so `dropout_in_finetune` should normally be False during
    Stage B - see the config file comment for why that's also a sensible
    choice on its own merits.

    -----------------------------------------------------------------------
    COST
    -----------------------------------------------------------------------
    Peak memory is O(chunk_size) instead of O(len(X_all)) - it no longer
    depends on the size of the full dataset at all, so `chunk_size` can be
    tuned purely to fit whatever GPU/CPU memory is available.
    Runtime cost: a plain full-batch step is roughly 1 forward + 1 backward
    pass (backward is typically ~2x the cost of forward, so ~3 "forward
    units" total). This function does 2 forward passes over the full
    dataset (steps 1 and the forward part of step 3) plus 1 backward pass
    worth of work spread across chunks in step 3 (~1 + 1 + 2 = 4 "forward
    units"). So expect roughly 30-40% more wall-clock time per epoch than
    an (unattainable, out-of-memory) single full-batch step - a small and
    predictable price for being able to train at all on the full dataset.

    -----------------------------------------------------------------------
    WEIGHTED CROSS-ENTROPY (cw_all, ce_weight_sum)
    -----------------------------------------------------------------------
    `cw_all`, if given, are the same per-event class weights used for the
    CE term in Stage A (pretrain's `cw_train`/`cw_val`, respecting
    class_weight_scheme), so the CE regularizer here is computed the same
    way (weighted mean) as in Stage A instead of silently degrading into
    an unweighted mean over individual events. `ce_weight_sum` is the
    (theta-independent, precomputed-once) sum of cw_all over the WHOLE
    dataset - used as a fixed normalizer instead of `n`.

    IMPORTANT: CE here is MONITORING-ONLY (ce_history/val_ce_history) - it
    is computed every call (via `tf.stop_recording()`, see Pass 2 below, so
    it costs nothing extra on the tape) but never enters cls_chunk_loss or
    any gradient. There used to be an optional `lambda_ce * CE` regularizer
    term on the classifier's loss; that has been removed as outdated - CE
    is tracked purely so you can watch the same quantity Stage A optimizes
    (typically training.ce_mode: "multiclass", the full n-class CE) without
    it influencing Stage B's parameter updates at all.

    `cw_all_no_abs`/`ce_weight_sum_no_abs`: same idea as `cw_all`/
    `ce_weight_sum` above, but weighted by the TRUE (signed, non-abs)
    per-event weights - the train-side counterpart of Stage A's
    cw_train_no_abs (training.py's train_one_epoch already computes both
    weighted_loss and weighted_loss_no_abs side-by-side; this mirrors that
    for Stage B). The per-sample CE values (`ce_vals`) are identical either
    way - only the weighting used to reduce them to a scalar differs - so
    this is computed from the SAME `ce_vals` already produced for the abs-
    weighted CE, at essentially no extra cost. Purely diagnostic, like the
    abs-weighted CE: never enters any gradient.

    -----------------------------------------------------------------------
    ADDITIONAL REGULARIZERS: high_bkg_uncertainty_penalty, covariance damping
    -----------------------------------------------------------------------
    `lam_unc` * high_bkg_uncertainty_penalty(B_w2_total, B_total,
    rel_threshold=bkg_unc_threshold) is added unconditionally to BOTH
    gato_scalar_loss and cls_scalar_loss (whenever the classifier has an
    objective at all) - unlike low_bkg_penalty, it has no separate
    penalty_in_gato_loss/penalty_in_classifier_loss-style placement
    toggle, so it never affects `fuse_objectives`' correctness condition
    (penalty_in_gato_loss == penalty_in_classifier_loss remains the sole
    thing that needs to match). It is a nonlinear function of the
    AGGREGATED B_w2_total/B_total, so it goes through the exact same
    gradient-caching trick as Z and low_bkg_penalty - just extended from a
    2-way (S, B) aggregate to a 3-way (S, B, B_w2) one; B_w2_total is
    accumulated additively across chunks in Pass 1 exactly like S_total/
    B_total (see model.chunk_yields' updated docstring for why this sum-
    of-squares quantity is just as additive as a sum). `lam_unc=0.0`
    (default) disables it exactly like `lam_penalty=0.0` disables
    low_bkg_penalty - B_w2 is still computed (cheap - it falls out of the
    same forward pass already computing S/B) but contributes nothing.

    `lam_cov` * covariance_damping_penalty(model.gato.get_scale_tril()) is
    a completely different case: it depends ONLY on the GMM's own
    parameters, never on any event/chunk data, so it is computed and
    differentiated exactly ONCE per step (a single small tape over
    gato_vars alone, after the chunk loop), not folded into the per-chunk
    pseudo-loss and not part of the gradient-caching trick at all - see
    covariance_damping_penalty's own docstring for why. It only ever
    affects gato_vars (the GMM has no analogue of a "classifier
    objective" for this term - covariance is purely a GMM parameter), so
    it has no placement toggle either. `lam_cov=0.0` (default) disables
    it - the tiny tape is skipped entirely in that case, not just zeroed.

    -----------------------------------------------------------------------
    SPLIT LOSSES: GMM gets Z, CLASSIFIER gets Z (or nothing)
    -----------------------------------------------------------------------
    Two DIFFERENT objectives are built and each parameter group is
    differentiated against its own:

        L_gato = -Z  [ + lam_penalty * low_bkg_penalty(B) ]
        L_cls  = -Z  [ + lam_penalty * low_bkg_penalty(B) ]        (only if significance_in_classifier_loss)

    (the bracketed penalty terms are toggled by `penalty_in_gato_loss` /
    `penalty_in_classifier_loss`.) `significance_in_classifier_loss=False`
    means the classifier has NO Stage-B objective at all - not "an objective
    that happens to evaluate to zero", but genuinely no loss is built and no
    gradient is requested for classifier_vars (see "NO-GRADIENT FAST PATH"
    below). This is the intended way to run Stage B with the classifier
    frozen (e.g. GMM-only categorization optimization on top of an already-
    trained classifier); set significance_in_classifier_loss=True whenever
    you DO want the classifier to keep moving with -Z during Stage B.

    Correctness of the split under chunking (when significance_in_classifier_loss
    is True). The gradient-caching trick above is applied SEPARATELY to
    each objective: one tiny tape gives (dS_g, dB_g) = (dL_gato/dS_total,
    dL_gato/dB_total) and a second gives (dS_c, dB_c) for L_cls. Per chunk
    we then form two pseudo-losses

        gato_chunk_loss = dS_g . S_c + dB_g . B_c
        cls_chunk_loss  = dS_c . S_c + dB_c . B_c

    and take tape.gradient(gato_chunk_loss, gato_vars) and
    tape.gradient(cls_chunk_loss, classifier_vars) off the SAME persistent
    tape (one shared forward pass, so no extra forward cost) whenever the
    two objectives actually differ (see "fuse_objectives" below for the
    cheaper case where they don't). Summed over chunks these are exactly
    the full-batch gradients of L_gato w.r.t. the GMM parameters and of
    L_cls w.r.t. the classifier parameters.

    -----------------------------------------------------------------------
    NO-GRADIENT FAST PATH (significance_in_classifier_loss=False)
    -----------------------------------------------------------------------
    When the classifier has no Stage-B objective, cls_chunk_loss is never
    built and classifier_vars are never passed to tape.gradient() at all -
    not computed-then-discarded, genuinely not requested. Since only ONE
    tensor (gato_chunk_loss) ever needs differentiating in this case, a
    plain NON-persistent GradientTape with a single tape.gradient() call is
    used - cheaper than even the "fused" path below, since there's no
    second objective to keep the tape alive for. The classifier's forward
    pass (to produce NN_output for the GMM, and logits for the monitored
    CE) still runs every chunk either way - only its OWN parameter gradient
    is skipped.

    Where the mass-window edges get their gradient. model.mass_low /
    model.mass_high enter only through the per-event significance weights,
    so they are always differentiated against `gato_chunk_loss`, regardless
    of whether the classifier has its own objective this run.

    -----------------------------------------------------------------------
    ASYMMETRIC UPDATE SCHEDULE (update_classifier / update_gato)
    -----------------------------------------------------------------------
    These two flags gate which parameter group actually receives an update
    from this call. They let the caller run k classifier steps per GMM step
    (see train_complete_model's `update_schedule` config), the same device
    GAN training uses to keep one player from outrunning the other. Note
    this is orthogonal to significance_in_classifier_loss: `update_classifier`
    only matters (and only has an effect) when significance_in_classifier_loss
    is True - with it False, the classifier already never receives a
    gradient regardless of `update_classifier`'s value.

    Why this is safe: every call recomputes S_total/B_total from scratch in
    Pass 1, so each step is a complete, exact full-batch gradient step for
    whichever group it updates - evaluated at the CURRENT parameters. There
    is no stale-gradient approximation: after a classifier-only step the
    yields have moved, and the next call simply measures them again.

    Skipping a group also skips its backward extraction (`tape.gradient`),
    so a classifier-only step is genuinely cheaper than a joint one, not
    just masked at the apply step.

    The mass-window edges follow the GMM (`update_gato`): they share the
    significance objective with it, and letting the analysis window drift
    on every classifier sub-step would defeat the point of holding the
    categorization still while the classifier catches up.
    """
    n = X_all.shape[0]
    bounds = _chunk_bounds(n, chunk_size)
    n_cats = model.gato.n_cats
    prof = profiler if profiler is not None else PhaseProfiler(enabled=False)

    # ---- Pass 1: forward-only, accumulate the true S_total / B_total ----
    S_total = tf.zeros(n_cats, dtype=tf.float32)
    B_total = tf.zeros(n_cats, dtype=tf.float32)
    B_w2_total = tf.zeros(n_cats, dtype=tf.float32)
    for (a, b) in bounds:
        with prof.phase("chunk_io", sub=True):
            Xc, yc, wc = _read_chunk(X_all, y_all, w_all, a, b, prof)
            mc = mass_all[a:b] if mass_all is not None else None
        with prof.phase("pass1_forward", sub=True):
            Sc, Bc, B_w2c, _ = _chunk_forward(model, Xc, yc, wc, mc, dropout_in_finetune, rewt)
            S_total = S_total + Sc
            B_total = B_total + Bc
            B_w2_total = B_w2_total + B_w2c

    # ---- Tiny tapes: analytic dL/dS_total, dL/dB_total ----
    # Cheap: this tape only ever touches two length-n_cats vectors, never any
    # per-event activations.
    S_total = tf.identity(S_total)
    B_total = tf.identity(B_total)
    B_w2_total = tf.identity(B_w2_total)
    with tf.GradientTape(persistent=significance_in_classifier_loss) as scalar_tape:
        scalar_tape.watch(S_total)
        scalar_tape.watch(B_total)
        scalar_tape.watch(B_w2_total)
        z_per_cat = asymptotic_significance(S_total, B_total)
        Z = tf.sqrt(tf.reduce_sum(z_per_cat ** 2))
        sig_loss = -Z
        penalty = low_bkg_penalty(B_total, threshold=penalty_thr)
        unc_penalty = high_bkg_uncertainty_penalty(B_w2_total, B_total, rel_threshold=bkg_unc_threshold)

        # Objective seen by the GMM ("gato just gets Z").
        gato_scalar_loss = sig_loss
        if penalty_in_gato_loss:
            gato_scalar_loss = gato_scalar_loss + lam_penalty * penalty
        gato_scalar_loss = gato_scalar_loss + lam_unc * unc_penalty

        # Objective seen by the classifier - ONLY built when the classifier
        # actually has a Stage-B objective (see this function's "NO-GRADIENT
        # FAST PATH" docstring section); otherwise dS_c/dB_c are never
        # needed at all, so skip computing them.
        if significance_in_classifier_loss:
            cls_scalar_loss = sig_loss
            if penalty_in_classifier_loss:
                cls_scalar_loss = cls_scalar_loss + lam_penalty * penalty
            cls_scalar_loss = cls_scalar_loss + lam_unc * unc_penalty

        # Monitoring value only (never differentiated): the full physics
        # objective, used for the scheduler / early stopping / history.
        scalar_loss = sig_loss + lam_penalty * penalty + lam_unc * unc_penalty

    dS_g, dB_g, dBw2_g = scalar_tape.gradient(gato_scalar_loss, [S_total, B_total, B_w2_total])
    if significance_in_classifier_loss:
        dS_c, dB_c, dBw2_c = scalar_tape.gradient(cls_scalar_loss, [S_total, B_total, B_w2_total])
    del scalar_tape

    # ---- Pass 2: one chunk at a time, forward+backward, accumulate grads ----
    # list(...) rather than the raw attribute: tf.Module returns a tuple while
    # keras.Model returns a list, and the fused path below concatenates these.
    classifier_vars = list(model.classifier.trainable_variables)
    gato_vars = list(model.gato.trainable_variables)
    # The mass-window edges follow the GMM's update cadence (see docstring).
    classifier_grad_acc = [tf.zeros_like(v) for v in classifier_vars]
    gato_grad_acc = [tf.zeros_like(v) for v in gato_vars]
    ce_total = tf.zeros([], dtype=tf.float32)
    ce_total_no_abs = tf.zeros([], dtype=tf.float32)

    # ---- Can the two objectives share a single, NON-persistent tape? ----
    # A persistent GradientTape has to retain every intermediate activation
    # of the chunk until it is explicitly deleted, because it has to be able
    # to serve an arbitrary number of later .gradient() calls. For a 5-layer
    # x 1024-node classifier that is on the order of 40 kB of retained
    # activations per event, i.e. ~800 MB at chunk_size=20000 - and it is the
    # single largest contributor to Stage-B peak GPU memory.
    #
    # significance_in_classifier_loss=False is the CHEAPEST case of all: the
    # classifier has no objective whatsoever, so only gato_chunk_loss is
    # ever differentiated - a single tensor, single tape.gradient() call,
    # plain non-persistent tape (handled directly in the loop below, not
    # via this flag).
    #
    # When significance_in_classifier_loss=True, fuse_objectives is only
    # True if additionally
    #     penalty_in_gato_loss == penalty_in_classifier_loss
    # in which case the two are built from literally the same terms
    # upstream:
    #     gato_scalar_loss = sig_loss [+ lam_penalty * penalty]
    #     cls_scalar_loss  = sig_loss [+ lam_penalty * penalty]
    # so (dS_g, dB_g) == (dS_c, dB_c) exactly, and therefore
    #     cls_chunk_loss == gato_chunk_loss.
    # Differentiating that SINGLE tensor against classifier_vars + gato_vars
    # in one call returns exactly the same numbers two separate calls
    # would - the GMM and mass-window gradients are unchanged bit-for-bit,
    # not approximated. One .gradient() call means the tape can be
    # non-persistent and can free each chunk's activations as it walks back
    # through them.
    #
    # The persistent path is kept for the case where both objectives are
    # real but differ (significance_in_classifier_loss=True with mismatched
    # penalty settings).
    fuse_objectives = bool(
        significance_in_classifier_loss
        and (penalty_in_gato_loss == penalty_in_classifier_loss)
    )

    prof.mark_chunk_step_start()
    for (a, b) in bounds:
        with prof.phase("chunk_io", sub=True):
            Xc, yc, wc = _read_chunk(X_all, y_all, w_all, a, b, prof)
            mc = mass_all[a:b] if mass_all is not None else None
            cwc = cw_all[a:b] if cw_all is not None else None
            cwc_no_abs = cw_all_no_abs[a:b] if cw_all_no_abs is not None else None
        ce_c = None
        ce_c_no_abs = None
        # persistent only needed for the "two genuinely different objectives"
        # case; both other cases (classifier disabled, or fused) only ever
        # need ONE tape.gradient() call.
        with prof.phase("pass2_fwd_bwd", sub=True):
            with tf.GradientTape(
                persistent=(significance_in_classifier_loss and not fuse_objectives)
            ) as tape:
                # Note: no explicit tape.watch(classifier_vars)/tape.watch(gato_vars)
                # here - trainable tf.Variables are watched automatically by
                # GradientTape, and passing a *list* of Variables into watch()
                # is not supported by all TF versions (it expects a single
                # Tensor/Variable or raises a ValueError like the one you hit).
                Sc, Bc, B_w2c, logits_c = _chunk_forward(model, Xc, yc, wc, mc, dropout_in_finetune, rewt)

                # Pseudo-loss for the GMM: pure significance objective plus
                # the (always-included, see docstring) uncertainty-penalty
                # term - same gradient-caching trick, just a 3-way
                # aggregate (S, B, B_w2) instead of 2-way (S, B).
                gato_chunk_loss = (
                    tf.reduce_sum(dS_g * Sc) + tf.reduce_sum(dB_g * Bc) + tf.reduce_sum(dBw2_g * B_w2c)
                )

                # Pseudo-loss for the classifier - ONLY built when the
                # classifier has a Stage-B objective at all (see this
                # function's "NO-GRADIENT FAST PATH" docstring section).
                # When the two objectives coincide we reuse the very same
                # tensor instead of rebuilding an identical one.
                if significance_in_classifier_loss:
                    if fuse_objectives:
                        cls_chunk_loss = gato_chunk_loss
                    else:
                        cls_chunk_loss = (
                            tf.reduce_sum(dS_c * Sc) + tf.reduce_sum(dB_c * Bc) + tf.reduce_sum(dBw2_c * B_w2c)
                        )

                # CE is MONITORING-ONLY (never enters any loss/gradient - see
                # docstring). Computed inside `tape.stop_recording()` so
                # these ops are never tracked by the tape at all, instead of
                # being tracked-then-ignored.
                with tape.stop_recording():
                    ce_vals = ce_loss_fn(yc, logits_c)  # per-sample CE, shape (chunk,)
                    if cwc is not None:
                        ce_c = tf.reduce_sum(ce_vals * tf.cast(cwc, tf.float32))
                    else:
                        ce_c = tf.reduce_sum(ce_vals)
                    if cwc_no_abs is not None:
                        ce_c_no_abs = tf.reduce_sum(ce_vals * tf.cast(cwc_no_abs, tf.float32))

            if not significance_in_classifier_loss:
                # No classifier objective this run: only gato_chunk_loss is
                # ever differentiated, so a single tape.gradient() call on
                # the (already non-persistent) tape above suffices. Cheaper
                # than "fused" - there's no second objective to keep the
                # tape alive for at all.
                g_classifier = []
                g_gato = tape.gradient(gato_chunk_loss, gato_vars) if update_gato else []
                del tape
            elif fuse_objectives:
                # ONE backward extraction for all parameter groups (see above).
                # The variables are concatenated in a fixed order and the
                # result is sliced back apart; groups that are not being
                # updated this step are left out of the request entirely, so
                # their backward work is still never computed.
                wanted = []
                if update_classifier:
                    wanted = wanted + classifier_vars
                if update_gato:
                    wanted = wanted + gato_vars

                g_all = tape.gradient(cls_chunk_loss, wanted) if wanted else []
                del tape

                cursor = 0
                if update_classifier:
                    g_classifier = g_all[cursor:cursor + len(classifier_vars)]
                    cursor += len(classifier_vars)
                else:
                    g_classifier = []
                if update_gato:
                    g_gato = g_all[cursor:cursor + len(gato_vars)]
                    cursor += len(gato_vars)
                else:
                    g_gato = []
                g_mass = g_all[cursor:cursor]
            else:
                # One shared forward pass, separate backward extractions - each
                # parameter group is differentiated against ITS OWN objective.
                # A group that is not being updated this step is skipped
                # entirely, so its backward pass is never computed.
                g_classifier = tape.gradient(cls_chunk_loss, classifier_vars) if update_classifier else []
                g_gato = tape.gradient(gato_chunk_loss, gato_vars) if update_gato else []
                del tape

        # Accumulated OUTSIDE the tape context so the running CE total never
        # keeps a chunk's graph alive across loop iterations.
        if ce_c is not None:
            ce_total = ce_total + ce_c
        if ce_c_no_abs is not None:
            ce_total_no_abs = ce_total_no_abs + ce_c_no_abs

        if update_classifier and significance_in_classifier_loss:
            classifier_grad_acc = [
                acc if g is None else acc + g for acc, g in zip(classifier_grad_acc, g_classifier)
            ]
        if update_gato:
            gato_grad_acc = [
                acc if g is None else acc + g for acc, g in zip(gato_grad_acc, g_gato)
            ]

    prof.mark_chunk_step_end()

    # ---- Covariance damping: ONE tiny tape, once per step, gato only ----
    # Deliberately NOT inside the chunk loop above and NOT part of the
    # gradient-caching trick - see covariance_damping_penalty's docstring
    # for why: it depends only on model.gato's own parameters (scale_tril),
    # never on any event/chunk data, so differentiating it once and adding
    # its gradient directly is exact - folding it into every chunk's
    # pseudo-loss would instead differentiate it n_chunks times.
    cov_penalty = tf.constant(0.0, dtype=tf.float32)
    if lam_cov != 0.0 and update_gato:
        with tf.GradientTape() as cov_tape:
            cov_penalty = covariance_damping_penalty(model.gato.get_scale_tril())
            cov_loss = lam_cov * cov_penalty
        cov_grads = cov_tape.gradient(cov_loss, gato_vars)
        gato_grad_acc = [
            acc if g is None else acc + g for acc, g in zip(gato_grad_acc, cov_grads)
        ]

    with prof.phase("optimizer_apply", sub=True):
        if update_classifier and significance_in_classifier_loss:
            classifier_optimizer.apply_gradients(zip(classifier_grad_acc, classifier_vars))
        if update_gato:
            gato_optimizer.apply_gradients(zip(gato_grad_acc, gato_vars))

    # Mean CE (monitoring-only, never contributed to any gradient - see
    # docstring).
    ce_denom = ce_weight_sum if ce_weight_sum is not None else tf.cast(n, tf.float32)
    ce_mean = ce_total / ce_denom

    # Same, but weighted by the TRUE (signed, non-abs) weights - see
    # cw_all_no_abs's docstring above. Falls back to NaN (not the abs-
    # weighted value, and not 0) when no cw_all_no_abs was passed, so it's
    # obvious in ce_no_abs_history that this run didn't track it, rather
    # than silently looking like a (misleading) valid number.
    if cw_all_no_abs is not None:
        ce_denom_no_abs = (
            ce_weight_sum_no_abs if ce_weight_sum_no_abs is not None else tf.cast(n, tf.float32)
        )
        ce_mean_no_abs = ce_total_no_abs / ce_denom_no_abs
    else:
        ce_mean_no_abs = tf.constant(float("nan"), dtype=tf.float32)

    return (
        sig_loss, penalty, scalar_loss, S_total, B_total, Z, ce_mean, ce_mean_no_abs,
        unc_penalty, cov_penalty,
    )


def evaluate_complete_model_chunked(
    model, X, y, w, mass, chunk_size, lam_penalty, penalty_thr,
    ce_loss_fn=None, cw=None, rewt=None,
    lam_unc=0.0, bkg_unc_threshold=0.2,
):
    """
    Forward-only (no GradientTape, no parameter update) evaluation of the
    Asimov-significance loss on a held-out set (e.g. the Stage-B validation
    split), memory-safe for large datasets.

    Reuses the same additive-yields trick as Pass 1 of
    `finetune_train_step_chunked`: S_total/B_total (and now B_w2_total) are
    accumulated chunk by chunk (each chunk's activations are freed again
    immediately), and Z/the uncertainty penalty are only ever computed once
    from the fully aggregated totals - so this returns numerically the same
    sig_loss/penalty/unc_penalty/Z as a full-batch forward pass over the
    whole set, just with O(chunk_size) peak memory. No gradient is ever
    needed here (forward-only), so unlike the chunked TRAIN step, there is
    no gradient-caching trick to apply - B_w2_total is just summed like any
    other monitoring-only aggregate.

    `mass`, if not None, is used the same way as in finetune_train_step_chunked
    (soft mass-window weighting with the model's *current* mass_low/mass_high
    - no gradient needed here since this is forward-only).
    """
    n = int(X.shape[0])
    bounds = _chunk_bounds(n, chunk_size if (chunk_size and chunk_size < n) else n)
    n_cats = model.gato.n_cats

    S_total = tf.zeros(n_cats, dtype=tf.float32)
    B_total = tf.zeros(n_cats, dtype=tf.float32)
    B_w2_total = tf.zeros(n_cats, dtype=tf.float32)
    ce_total = tf.zeros([], dtype=tf.float32)
    ce_denom = 0.0
    want_ce = ce_loss_fn is not None

    for (a, b) in bounds:
        mc = mass[a:b] if mass is not None else None
        Sc, Bc, B_w2c, logits_c = _chunk_forward(model, X[a:b], y[a:b], w[a:b], mc, False, rewt)
        S_total = S_total + Sc
        B_total = B_total + Bc
        B_w2_total = B_w2_total + B_w2c
        if want_ce:
            ce_vals = ce_loss_fn(y[a:b], logits_c)
            if cw is not None:
                cwc = tf.cast(np.asarray(cw[a:b], dtype=np.float32), tf.float32)
                ce_total = ce_total + tf.reduce_sum(ce_vals * cwc)
                ce_denom += float(tf.reduce_sum(cwc).numpy())
            else:
                ce_total = ce_total + tf.reduce_sum(ce_vals)
                ce_denom += float(b - a)

    z_per_cat = asymptotic_significance(S_total, B_total)
    Z = tf.sqrt(tf.reduce_sum(z_per_cat ** 2))
    sig_loss = -Z
    penalty = low_bkg_penalty(B_total, threshold=penalty_thr)
    unc_penalty = high_bkg_uncertainty_penalty(B_w2_total, B_total, rel_threshold=bkg_unc_threshold)
    scalar_loss = sig_loss + lam_penalty * penalty + lam_unc * unc_penalty
    ce_mean = float(ce_total.numpy() / ce_denom) if (want_ce and ce_denom > 0.0) else float("nan")
    return (
        float(sig_loss.numpy()), float(penalty.numpy()),
        float(scalar_loss.numpy()), float(Z.numpy()), ce_mean,
        float(unc_penalty.numpy()),
    )


def compute_confusion_counts_from_arrays(classifier, X, y, weights, n_classes, chunk_size=None):
    """
    Array-based counterpart of compute_confusion_counts_from_dataset, for
    Stage B: X/y/weights are the plain in-memory tensors/arrays
    finetune_train_step_chunked etc. already work with (e.g. X_all_t/
    y_all_t/w_all_t for train, X_val/y_val/w_val for val), chunked here the
    same way build_data_dict_chunked chunks them, to bound memory on large
    datasets. Returns (cm_weighted, cm_raw) with the exact same shape/
    semantics as compute_confusion_counts_from_dataset - see that
    function's docstring.
    """
    n = int(X.shape[0])
    step = chunk_size or n
    cm_weighted = np.zeros((n_classes, n_classes), dtype=np.float64)
    cm_raw = np.zeros((n_classes, n_classes), dtype=np.int64)
    y_np_all = np.asarray(y)
    w_np_all = np.asarray(weights, dtype=np.float64)
    for a in range(0, n, step):
        b = min(a + step, n)
        Xc = tf.constant(np.asarray(X[a:b], dtype=np.float32))
        logits = classifier(Xc, training=False)
        preds = tf.argmax(logits, axis=1).numpy().astype(np.int64)
        y_np = y_np_all[a:b].astype(np.int64)
        w_np = w_np_all[a:b]
        flat_idx = y_np * n_classes + preds
        cm_weighted += np.bincount(flat_idx, weights=w_np, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        cm_raw += np.bincount(flat_idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(np.int64)
    return cm_weighted, cm_raw


def train_complete_model(model, X_train, y_train, w_train, training_config,
                          X_val=None, y_val=None, w_val=None,
                          mass_train=None, mass_val=None,
                          cw_train=None, cw_val=None, cw_train_no_abs=None, n_classes=None,
                          profile_out_path=None,
                          nonres_reweight_fn=None, out_dir=None, classes=None,
                          finetune_checkpoint_every=0):
    """
    Stage B (fine-tuning).

    nonres_reweight_fn: callable, e.g.
    end_to_end_helpers.compute_nonres_reweight_factors, matching its
    signature. Required (non-None) whenever
    training_config["nonres_reweight"]["enabled"] is true; passed in as a
    callback rather than imported directly to avoid a circular import
    (end_to_end_helpers.py already imports from this module). Every
    nonres_reweight.recompute_every epochs it is called with the model's
    CURRENT weights on the TRAIN split only (never validation, to avoid
    leaking val-split information into training - same as the reference's
    identical comment), and the resulting per-category factors are used as
    `rewt` for every training/eval step until the next recompute.

    If a validation split is given AND nonres_reweight["track_val_fit"] is
    true (default), the SAME fit is ALSO run on the val split at the same
    cadence, purely for side-by-side monitoring of how the two splits'
    sideband fits compare over training - its returned factors are only
    logged/written to CSV (see out_dir below), never assigned to `rewt`,
    so this does NOT reintroduce the val-leakage the train-only design
    above exists to avoid. Uses its own, smaller nbins (see
    scaled_nbins_for_split - val has far fewer sideband events than
    train).

    out_dir: if given, every one of these periodic in-training refits also
    writes a per-category diagnostic CSV (n_raw, nbins, branch, chi2/ndof,
    fit A/B, factor - see compute_nonres_reweight_factors' docstring) to
    out_dir/nonres_fit_diagnostics_training/nonres_fit_diagnostics_epoch
    {ep:04d}.csv (train split) and, if the val-split tracking above is
    active, out_dir/nonres_fit_diagnostics_validation/nonres_fit_
    diagnostics_epoch{ep:04d}.csv (val split) - this used to not be saved
    anywhere at all, only printed to stdout as one line. Controlled by
    training_config["nonres_reweight"]["save_training_fit_diagnostics_csv"]
    (default True) and, for the accompanying per-category histogram+fit+
    pull PDF plots, ["save_training_fit_plots"] (default False, since a
    long run can trigger this many times - e.g. n_epochs=450,
    recompute_every=50 is 9 recomputes x n_cats plots each) - both flags
    apply to train AND val output identically (they're written to
    separate directories so they never collide). If out_dir is None (or
    nonres_reweight is disabled), nothing is written and behavior is
    unchanged from before.

    The GMM ("gato") and the classifier are optimized against DIFFERENT
    objectives (see finetune_train_step_chunked's docstring for the exact
    chunked-gradient bookkeeping):

        GMM:        L_gato = -Z [+ lam_penalty * low_bkg_penalty(B)]
        classifier: L_cls  = -Z [+ lam_penalty * low_bkg_penalty(B)]   (only if significance_in_classifier_loss)
                    or no Stage-B objective at all (significance_in_classifier_loss=False -
                    e.g. to run Stage B as GMM-only categorization optimization on top
                    of an already-trained, frozen classifier)

    `n_classes` is needed to build the (monitoring-only) collapsed/multiclass
    CE tracked alongside this (training.ce_mode, see make_ce_loss_fn); if
    None it is inferred from the classifier's output layer.

    cw_train: optional, per-event class weights for the MONITORED CE
    (ce_history/val_ce_history - it never enters any Stage-B gradient, see
    finetune_train_step_chunked's docstring), row-aligned with
    X_train/y_train/w_train. Should be the SAME array (respecting
    class_weight_scheme's abs/only-positive handling) used for cw_train in
    Stage A's pretrain(), so the tracked CE here is a weighted mean exactly
    like Stage A's CE, instead of an unweighted mean over individual events
    (which is not directly comparable to Stage A's CE value). If None, CE
    monitoring falls back to an unweighted mean and prints a warning.

    cw_train_no_abs: optional, the TRUE (signed, non-abs) per-event
    weights on the train split - the Stage-B counterpart of Stage A's
    cw_train_no_abs (train_one_epoch already tracks both weighted_loss and
    weighted_loss_no_abs side-by-side; see its docstring). Row-aligned with
    X_train/y_train/w_train/cw_train. Produces a second, separately
    tracked history (ce_no_abs_history) alongside ce_history, using the
    exact same per-sample CE values, just reduced with the true weights
    instead of the abs-valued ones. Purely diagnostic - like cw_train, it
    never enters any Stage-B gradient. If None, ce_no_abs_history is filled
    with NaN for every epoch rather than silently omitted, so its absence
    is visible instead of just missing from a results dict.

    Note this has no val-side counterpart: cw_val is already the TRUE
    (signed, non-abs) weights (see load_class_weights - only the TRAIN
    split needs a separate abs-valued variant, because it enters a
    gradient/weighted-mean loss that needs stable, non-negative weights;
    cw_val is monitoring-only in both Stage A and Stage B and was never
    abs-valued to begin with).

    X_train/y_train/w_train are used for the actual gradient updates.
    X_val/y_val/w_val, if given (non-None and non-empty), are used purely
    for monitoring: a forward-only significance loss is computed on them
    once per epoch, and is used to (a) drive the same kind of
    ReduceLROnPlateau scheduling and early stopping that Stage A already
    uses, and (b) pick the best-validation classifier+GMM snapshot to
    return, instead of always returning whatever the last epoch produced.
    If no validation set is given, behavior falls back to the previous
    "always run n_epochs on X_train" behavior.

    finetune_checkpoint_every: if > 0 (and out_dir is given), saves a
    PAIRED classifier + gato checkpoint every `finetune_checkpoint_every`
    epochs, under out_dir/finetune_checkpoints/ - mirroring pretrain()'s
    checkpoint_every, but for Stage B, and always saving BOTH halves of
    the model together at the same epoch (not just the classifier).
    Deliberately paired: a classifier snapshot from epoch N is only
    evaluable together with the GMM categorizer's state AT THAT SAME
    epoch N - the existing "best" checkpoint further down already saves
    both for exactly this reason, this just does it on a fixed cadence
    too, independent of validation performance, so intermediate epochs
    are recoverable even if they were never the running best. 0 (default)
    disables this - unaffected otherwise. Exposed as a CLI flag
    (--finetune-checkpoint-every) rather than a config key in
    end_to_end_training.py, per the user's request.
    """

    # ---- load config (key-based, robust to key order / new keys) ----
    train_cfg = training_config["training"]
    n_epochs = train_cfg["n_epochs"]
    early_stopping_patience = train_cfg["early_stopping_patience"]
    # NOTE: training.lambda_ce (the old CE-regularizer weight) is no longer
    # read here - CE is monitoring-only now (see finetune_train_step_chunked's
    # docstring); the key can be left in an existing config file harmlessly,
    # it's simply ignored.
    #
    # All three Stage-B regularizers now live under their own dedicated
    # `regularization:` top-level config section (moved out of `training:`,
    # which previously held lam_penalty/penalty_threshold directly) - see
    # combined_optimization_config.yaml. Each sub-block's `lam` is the
    # additive weight added to the loss; low_bkg_penalty has no `enabled`
    # flag (it predates this section and was always on - lam=0.0 disables
    # it the same way it always could). The other two are newer and
    # default to disabled: `enabled: false` (or omitting the block
    # entirely) forces lam to 0.0 regardless of whatever `lam` value is
    # written in the config, so a sweep can leave a tuned lam in place
    # while still toggling the term off/on with one flag.
    reg_cfg = training_config["regularization"]
    lam_penalty = reg_cfg["low_bkg_penalty"]["lam"]
    penalty_thr = reg_cfg["low_bkg_penalty"]["threshold"]

    unc_cfg = reg_cfg.get("high_bkg_uncertainty_penalty", {}) or {}
    lam_unc = float(unc_cfg.get("lam", 0.0)) if unc_cfg.get("enabled", False) else 0.0
    bkg_unc_threshold = float(unc_cfg.get("rel_threshold", 0.2))

    cov_cfg = reg_cfg.get("covariance_damping", {}) or {}
    lam_cov = float(cov_cfg.get("lam", 0.0)) if cov_cfg.get("enabled", False) else 0.0

    scheduler_cfg = train_cfg.get("scheduler", {}) or {}

    # chunk_size: how many rows are processed at a time in the memory-
    # efficient chunked training step (see finetune_train_step_chunked's
    # docstring). 0/None/missing -> fall back to the original, single-pass
    # full-batch step (only viable for datasets that fit in memory whole).
    chunk_size = train_cfg.get("chunk_size", None)
    dropout_in_finetune = bool(train_cfg.get("dropout_in_finetune", False))

    # How often to run the per-epoch DIAGNOSTIC block (GMM bias + temperature
    # history + the log line). That block calls build_data_dict_chunked, i.e.
    # a complete extra forward pass over the training set plus a host-side
    # copy of every event's NN_output/weight - roughly one more full pass per
    # epoch on top of the actual optimization. It contributes nothing to the
    # gradients, so running it every epoch is pure overhead; 10 keeps the
    # history plots perfectly readable at ~10% of the cost. Set to 1 to
    # restore the old (accidental) every-epoch behaviour.
    diagnostics_every = max(1, int(train_cfg.get("diagnostics_every", 10)))

    # ---- wall-clock / memory instrumentation ----
    prof_cfg = train_cfg.get("profiling", {}) or {}
    profiler = PhaseProfiler(
        enabled=bool(prof_cfg.get("enabled", True)),
        sync=bool(prof_cfg.get("device_sync", True)),
        label="Stage B (fine-tuning)",
    )

    # ---- non-resonant continuum-background reweighting (optional) ----
    nonres_cfg = training_config.get("nonres_reweight", {}) or {}
    nonres_enabled = bool(nonres_cfg.get("enabled", False))
    if nonres_enabled:
        if nonres_reweight_fn is None:
            raise ValueError(
                "training_config['nonres_reweight']['enabled'] is true but "
                "no nonres_reweight_fn was passed to train_complete_model() "
                "(pass end_to_end_helpers.compute_nonres_reweight_factors "
                "from end_to_end_training.py)."
            )
        if mass_train is None:
            raise ValueError("nonres_reweight.enabled requires mass_train.")
    nonres_class = int(nonres_cfg.get("nonres_bkg_class", -1))
    nonres_recompute_every = max(1, int(nonres_cfg.get("recompute_every", 50)))
    nonres_default_factor = float(nonres_cfg.get("default_factor", 0.062))
    n_cats_gmm = model.gato.n_cats
    rewt_tf = (
        tf.constant(np.full(n_cats_gmm, nonres_default_factor, dtype=np.float32))
        if nonres_enabled else None
    )
    # Diagnostics for the periodic refit below (see out_dir in this
    # function's docstring) - previously not saved anywhere.
    nonres_diag_dir = (
        os.path.join(out_dir, "nonres_fit_diagnostics_training")
        if (nonres_enabled and out_dir is not None) else None
    )
    nonres_save_train_csv = bool(nonres_cfg.get("save_training_fit_diagnostics_csv", True))
    nonres_save_train_plots = bool(nonres_cfg.get("save_training_fit_plots", False))
    nonres_plot_dtype = training_config.get("plot_dtype", "png")

    # Same periodic sideband fit, but on the VAL split, purely for
    # side-by-side monitoring - NOT used to update rewt_tf/the training
    # loss (see the epoch loop below: this call's `factors` return value
    # is only logged/written to CSV, never assigned to rewt_tf), so it
    # does NOT reintroduce the val-info-leakage the train-only design
    # above was written to avoid. Written to its own directory so it
    # never collides with the train-side CSVs/plots. Reuses the same
    # save_training_fit_diagnostics_csv/save_training_fit_plots flags
    # (they now gate BOTH splits' periodic-refit output) plus its own
    # track_val_fit toggle, since this doubles the periodic sideband-fit
    # cost (one extra forward pass + fit per recompute) and some users
    # may not want that.
    nonres_track_val_fit = bool(nonres_cfg.get("track_val_fit", True))
    nonres_val_diag_dir = (
        os.path.join(out_dir, "nonres_fit_diagnostics_validation")
        if (nonres_enabled and nonres_track_val_fit and out_dir is not None) else None
    )

    # Confusion-matrix (TP/FP/TN/FN etc.) counts, saved every
    # diagnostics_every epochs alongside the bias/temperature diagnostics
    # already computed there (see the main loop below) - reuses the same
    # forward pass's cost bracket rather than adding a new cadence. One
    # growing CSV per split (confusion_counts_train.csv / _val.csv) under
    # out_dir/stageB_confusion_counts/, one row appended per diagnostics
    # epoch - see append_confusion_counts_row. Requires out_dir; silently
    # skipped (no behavior change) if not given.
    confusion_dir = os.path.join(out_dir, "stageB_confusion_counts") if out_dir is not None else None

    # Periodic checkpoint pair at the discriminant-density cadence.
    # PREVIOUSLY this block computed+saved live 2D discriminant-density
    # histograms (compute_class_dim_histograms/append_class_dim_histograms -
    # per-true-class 1/2/3-sigma-equivalent density contours over the
    # classifier's full softmax output). That's been replaced with a much
    # cheaper action at the SAME cadence/config section
    # (discriminant_density_diagnostics.enabled/.recompute_every): just save
    # a paired classifier+gato checkpoint, under
    # out_dir/discriminant_checkpoints/. The discriminant-density evolution
    # can still be reconstructed AFTER the fact from these checkpoints (run
    # compute_class_dim_histograms/plot_discriminant_density.py against each
    # saved pair offline) without paying the histogram compute+storage cost
    # live, every disc_recompute_every epochs, during the run itself.
    # Requires out_dir; silently skipped (no behavior change) if not given
    # or not enabled.
    disc_cfg = training_config.get("discriminant_density_diagnostics", {}) or {}
    disc_enabled = bool(disc_cfg.get("enabled", False)) and out_dir is not None
    disc_recompute_every = max(1, int(disc_cfg.get("recompute_every", 10)))
    disc_checkpoints_dir = os.path.join(out_dir, "discriminant_checkpoints") if disc_enabled else None
    if disc_checkpoints_dir is not None:
        os.makedirs(disc_checkpoints_dir, exist_ok=True)
    if confusion_dir is not None:
        os.makedirs(confusion_dir, exist_ok=True)
    confusion_class_names = list(classes) if classes is not None else None

    # Periodic PAIRED classifier+gato checkpoints (see finetune_checkpoint_every
    # in this function's docstring). Own directory, parallel to pretrain()'s
    # pretrain_checkpoints/, so Stage A and Stage B periodic snapshots never
    # collide even when continue_training points a later run at the same
    # out_dir.
    finetune_checkpoint_every = int(finetune_checkpoint_every or 0)
    finetune_checkpoint_dir = (
        os.path.join(out_dir, "finetune_checkpoints")
        if (finetune_checkpoint_every > 0 and out_dir is not None) else None
    )
    if finetune_checkpoint_dir is not None:
        os.makedirs(finetune_checkpoint_dir, exist_ok=True)

    # Step-level (sub-epoch) history: ONE row per entry of `step_plan`
    # (built below), i.e. one row per epoch under the default 1:1
    # schedule (identical information to loss_history/penalty_history/
    # ce_history, just also on disk), or several rows per epoch under an
    # asymmetric schedule (classifier_steps_per_epoch/gato_steps_per_epoch
    # != 1:1). Previously, under an asymmetric schedule, ONLY the last
    # (always-fused) step's metrics were kept - every classifier-only or
    # GMM-only step's sig_loss/penalty/ce/Z was computed and then
    # discarded, so there was no way to see what those intermediate steps
    # were actually doing. This CSV keeps all of them, tagged with which
    # half of the model that step updated, so a schedule's within-epoch
    # behavior can be plotted/debugged later - see append_step_history_row.
    step_history_csv_path = os.path.join(out_dir, "stageB_step_history.csv") if out_dir is not None else None

    # ---- split-loss configuration (GMM objective vs. classifier objective) ----
    loss_split_cfg = train_cfg.get("loss_split", {}) or {}
    penalty_in_gato_loss = bool(loss_split_cfg.get("penalty_in_gato_loss", True))
    penalty_in_classifier_loss = bool(loss_split_cfg.get("penalty_in_classifier_loss", True))
    significance_in_classifier_loss = bool(
        loss_split_cfg.get("significance_in_classifier_loss", True)
    )

    # ---- IMPORTANT: make the "no classifier gradient" path ACTUALLY skip work ----
    # finetune_train_step_chunked already avoids *requesting* the
    # classifier's gradient when significance_in_classifier_loss is False
    # (tape.gradient(gato_chunk_loss, gato_vars) only). That is NOT enough
    # on its own: _chunk_forward is wrapped in @tf.function, and TF's
    # autodiff builds ONE general backward function for that whole traced
    # call, covering every variable that was WATCHED during tracing - not
    # just whichever subset of `sources` a caller later asks for. If the
    # classifier's variables are still `trainable=True` (hence
    # auto-watched by GradientTape), that backward function still computes
    # and allocates the full per-event backward signal through every
    # classifier layer, even though the result is discarded - this is what
    # causes an OOM on a [chunk_size, num_nodes] activation-gradient tensor
    # inside the classifier's hidden layers, even at epoch 0, even though
    # the classifier's gradient is never used.
    #
    # The actual fix is to change what GradientTape WATCHES, not just what
    # it's asked to return: GradientTape only auto-watches `trainable`
    # variables, so marking the classifier genuinely non-trainable for the
    # duration of Stage B means the backward function generated for
    # _chunk_forward never builds that unnecessary path through the
    # classifier's layers in the first place. This does NOT change the
    # classifier's forward computation (Dropout etc. are controlled by the
    # `training` argument passed to call(), not by `.trainable`) - only
    # what gets differentiated. Restored to its original value at the end
    # of this function so it has no surprising side effects on the model
    # object for any caller (e.g. a later call with
    # significance_in_classifier_loss=True).
    classifier_was_trainable = model.classifier.trainable
    if not significance_in_classifier_loss and classifier_was_trainable:
        model.classifier.trainable = False
        print(
            "[Stage B] significance_in_classifier_loss=False: setting "
            "model.classifier.trainable=False for this call. Without this, "
            "the classifier's gradient is never *requested* but still gets "
            "*computed* inside _chunk_forward's tf.function boundary, "
            "wasting exactly the memory this setting is meant to save - "
            "see the comment above this print for the full explanation."
        )

    # ---- asymmetric update schedule (classifier_steps_per_epoch / gato_steps_per_epoch) ----
    # NOTE: these keys used to be read as sched_cfg["class_steps"]/["gato_steps"],
    # which never matched training_config.yaml's actual keys
    # (update_schedule.classifier_steps_per_epoch / .gato_steps_per_epoch) -
    # since .get() silently falls back to its default on a missing key,
    # setting either yaml key to anything other than 1 had NO effect at
    # all (this always silently ran the 1:1 schedule instead). Fixed here
    # to read the yaml's actual key names.
    sched_cfg = train_cfg.get("update_schedule", {}) or {}

    class_steps = int(sched_cfg.get("classifier_steps_per_epoch", 1))
    gato_steps = int(sched_cfg.get("gato_steps_per_epoch", 1))

    if class_steps < 1 or gato_steps < 1:
        raise ValueError(
            "training.update_schedule.classifier_steps_per_epoch and gato_steps_per_epoch "
            "must both be >= 1."
        )
    if class_steps != 1 and gato_steps != 1:
        raise ValueError(
            "training.update_schedule requires exactly one of classifier_steps_per_epoch "
            "or gato_steps_per_epoch to be 1."
        )

    # Build the per-epoch update plan.
    # The last step is always joint.
    if class_steps > gato_steps:
        step_plan = [(True, False)] * (class_steps - 1) + [(True, True)]
    elif gato_steps > class_steps:
        step_plan = [(False, True)] * (gato_steps - 1) + [(True, True)]
    else:
        step_plan = [(True, True)]

    passes_per_epoch = len(step_plan)

    # ---- which cross-entropy is tracked for monitoring (never a gradient term) ----
    ce_mode = train_cfg.get("ce_mode", "binary_plain")
    signal_class = int(training_config["gato_model"]["signal_class"])
    if n_classes is None:
        # Fall back to the classifier's output dimensionality.
        n_classes = int(model.classifier.model_layers[-1].units)

    # Class weights for the MONITORED CE - see cw_train's docstring above.
    # ce_weight_sum is a FIXED constant (sum over the whole train set,
    # independent of theta), precomputed once so both the chunked and
    # full-batch paths can divide by it as a normalizer, matching
    # evaluate_complete_model_chunked()'s own weighted-mean convention
    # (tf.reduce_sum(cwc)) so train- and val-side CE stay directly
    # comparable to each other and to Stage A's CE.
    cw_all_t = cw_train  # kept as-is (memmap/ndarray) for chunked slicing below
    ce_weight_sum = None
    if cw_train is not None:
        ce_weight_sum = tf.cast(
            tf.reduce_sum(tf.cast(np.asarray(cw_train), tf.float32)), tf.float32
        )
    else:
        print(
            "[Stage B][INFO] No cw_train was passed to train_complete_model() - "
            "the monitored cross-entropy (ce_history/val_ce_history) will fall "
            "back to an UNWEIGHTED mean, which is NOT directly comparable to "
            "Stage A's weighted CE. This has no effect on any gradient - CE is "
            "monitoring-only in Stage B."
        )

    # Same, but for the TRUE (signed, non-abs) weights - see
    # cw_train_no_abs's docstring above. Independent of cw_train: it's fine
    # (and expected) for cw_train to be given while cw_train_no_abs is not,
    # e.g. for older call sites that haven't been updated yet.
    cw_all_no_abs_t = cw_train_no_abs
    ce_weight_sum_no_abs = None
    if cw_train_no_abs is not None:
        ce_weight_sum_no_abs = tf.cast(
            tf.reduce_sum(tf.cast(np.asarray(cw_train_no_abs), tf.float32)), tf.float32
        )
    else:
        print(
            "[Stage B][INFO] No cw_train_no_abs was passed to train_complete_model() - "
            "ce_no_abs_history will be filled with NaN. This has no effect on "
            "any gradient or on ce_history - it only disables the additional, "
            "true-weighted CE diagnostic."
        )

    use_validation = X_val is not None and y_val is not None and w_val is not None and int(X_val.shape[0]) > 0

    temperature_init = training_config["gato_model"]["temperature_init"]
    temperature_final = training_config["gato_model"]["temperature_final"]
    class_lr = training_config["train_optimizer"]["class_learning_rate"]
    gato_lr = training_config["train_optimizer"]["gato_learning_rate"]

    # Keep numpy/memmap views on CPU; chunked training slices one chunk at a
    # time so only O(chunk_size) is transferred to the GPU per step.
    n_total = int(X_train.shape[0])
    X_all_t, y_all_t, w_all_t = X_train, y_train, w_train
    mass_all_t = mass_train  # kept as-is (memmap/ndarray) for chunked slicing below

    # separate optimizers (and separate LRs) for classifier vs. GMM: the
    # classifier typically needs a much smaller LR at this stage than the GMM.
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=class_lr, epsilon=1e-8)
    gato_optimizer = tf.keras.optimizers.RMSprop(learning_rate=make_schedule(gato_lr, steps=n_epochs // 7))


    mass_window_enabled = bool(getattr(model, "mass_window_enabled", False))
    if mass_window_enabled:
        print(
            f"[Stage B] Soft mass-window weighting enabled"
            f"window=[{float(model.mass_low.numpy()):.3g}, "
            f"{float(model.mass_high.numpy()):.3g}], softness={float(model.mass_softness):.3g}.\n"
        )

    ce_loss_fn = make_ce_loss_fn(ce_mode, signal_class, n_classes)
    print(
        f"[Stage B] Cross-entropy mode='{ce_mode}' (signal_class={signal_class}, "
        f"n_classes={n_classes}) is computed and tracked (ce_history/val_ce_history) "
        f"for monitoring only - it never enters any Stage-B loss or gradient."
    )
    print(
        f"[Stage B] Loss split -> GMM: -Z"
        f"{' + lam*penalty' if penalty_in_gato_loss else ''} | "
        f"classifier: {'-Z' if significance_in_classifier_loss else '(no Stage-B objective - frozen)'}"
        f"{' + lam*penalty' if (significance_in_classifier_loss and penalty_in_classifier_loss) else ''}\n"
    )
    if not significance_in_classifier_loss:
        print(
            "[Stage B] significance_in_classifier_loss=False: the classifier has NO "
            "Stage-B objective at all, so its gradient is never computed (not "
            "computed-then-discarded) - cheapest possible chunked step, a single "
            "non-persistent GradientTape differentiating only the GMM's objective.\n"
        )
    elif penalty_in_gato_loss == penalty_in_classifier_loss:
        print(
            "[Stage B] The GMM and classifier objectives are the same function of "
            "(S, B), so the chunked step uses ONE non-persistent GradientTape per "
            "chunk instead of a persistent one. Gradients are unchanged; peak GPU "
            "memory per chunk drops substantially because activations are freed "
            "during the single backward pass.\n"
        )
    else:
        print(
            "[Stage B] The GMM and classifier objectives differ (loss_split), so the "
            "chunked step keeps a persistent GradientTape per chunk. Expect higher "
            "peak GPU memory; lower training.chunk_size if it does not fit.\n"
        )
    if class_steps == 1 and gato_steps == 1:
        print(
            "[Stage B] Update schedule: class_steps=1, gato_steps=1 "
            "(single joint update per epoch).\n"
        )
    elif class_steps > gato_steps:
        print(
            f"[Stage B] Update schedule: class_steps={class_steps}, gato_steps={gato_steps} "
            f"(classifier-only steps first, then one final joint step).\n"
            f"           One epoch = {passes_per_epoch} full pass(es) over the training set.\n"
        )
    else:
        print(
            f"[Stage B] Update schedule: class_steps={class_steps}, gato_steps={gato_steps} "
            f"(GMM-only steps first, then one final joint step).\n"
            f"           One epoch = {passes_per_epoch} full pass(es) over the training set.\n"
        )

    temperature_scheduler = TemperatureScheduler(
        model.gato, t_initial=temperature_init, t_final=temperature_final,
        total_epochs=n_epochs, mode="cosine",
    )

    use_chunked = bool(chunk_size) and chunk_size < n_total
    if use_chunked:
        print(
            f"[Stage B] Using chunked training (chunk_size={chunk_size}, "
            f"n_total={n_total}) - gradients are exact, see training.py "
            f"docstring of finetune_train_step_chunked for details.\n"
        )
    else:
        print(
            "[Stage B] chunk_size not set (or >= dataset size): using a "
            "single full-batch forward/backward pass, as before.\n"
        )
        X_all_t = tf.constant(np.asarray(X_train, dtype=np.float32))
        y_all_t = tf.constant(np.asarray(y_train, dtype=np.int64))
        w_all_t = tf.constant(np.asarray(w_train, dtype=np.float32))
        mass_all_t = (
            tf.constant(np.asarray(mass_train, dtype=np.float32)) if mass_train is not None else None
        )
        cw_all_t = (
            tf.constant(np.asarray(cw_train, dtype=np.float32)) if cw_train is not None else None
        )
        cw_all_no_abs_t = (
            tf.constant(np.asarray(cw_train_no_abs, dtype=np.float32))
            if cw_train_no_abs is not None else None
        )

        @tf.function
        def _finetune_grads_full_batch(X, y, w, mass, cw, cw_no_abs, update_classifier, update_gato, rewt=None):
            with tf.GradientTape(persistent=True) as tape:
                sig_loss, S, B, Z, B_w2, scores, logits = model.call(X, y, w, mass=mass, training=dropout_in_finetune, rewt=rewt)
                penalty = low_bkg_penalty(B, threshold=penalty_thr)
                unc_penalty = high_bkg_uncertainty_penalty(B_w2, B, rel_threshold=bkg_unc_threshold)
                # Covariance damping: pure model-parameter regularizer, no
                # data dependence - fine to just add it straight into the
                # same full-batch tape here (unlike the chunked path, there
                # is no per-chunk double-differentiation risk to guard
                # against when there's only one "chunk").
                cov_penalty = (
                    covariance_damping_penalty(model.gato.get_scale_tril())
                    if lam_cov != 0.0 else tf.constant(0.0, dtype=tf.float32)
                )

                # --- objective seen by the GMM: "gato just gets Z" ---
                gato_loss = sig_loss
                if penalty_in_gato_loss:
                    gato_loss = gato_loss + lam_penalty * penalty
                gato_loss = gato_loss + lam_unc * unc_penalty + lam_cov * cov_penalty

                # --- objective seen by the classifier: Z, or none at all ---
                # Only built when the classifier actually has a Stage-B
                # objective; with significance_in_classifier_loss=False the
                # classifier is intentionally frozen for this run (its
                # gradient is simply never requested below).
                if significance_in_classifier_loss:
                    cls_loss = sig_loss
                    if penalty_in_classifier_loss:
                        cls_loss = cls_loss + lam_penalty * penalty
                    cls_loss = cls_loss + lam_unc * unc_penalty
                    # Covariance damping deliberately excluded from cls_loss:
                    # it only ever depends on model.gato's parameters, so it
                    # has no classifier gradient to contribute regardless.
                else:
                    cls_loss = None

                # CE is MONITORING-ONLY (ce_history/val_ce_history) - never
                # enters any loss/gradient. Computed inside
                # `tape.stop_recording()` so it's never tracked at all.
                with tape.stop_recording():
                    ce_vals = ce_loss_fn(y, logits)  # per-sample CE
                    if cw is not None:
                        ce = tf.reduce_sum(ce_vals * cw) / tf.reduce_sum(cw)
                    else:
                        ce = tf.reduce_mean(ce_vals)
                    if cw_no_abs is not None:
                        ce_no_abs = tf.reduce_sum(ce_vals * cw_no_abs) / tf.reduce_sum(cw_no_abs)
                    else:
                        ce_no_abs = tf.constant(float("nan"), dtype=tf.float32)

                # Monitoring only (never differentiated).
                total_loss = sig_loss + lam_penalty * penalty + lam_unc * unc_penalty + lam_cov * cov_penalty

            classifier_grads = (
                tape.gradient(cls_loss, model.classifier.trainable_variables)
                if (update_classifier and cls_loss is not None) else []
            )
            gato_grads = (
                tape.gradient(gato_loss, model.gato.trainable_variables)
                if update_gato else []
            )
            del tape

            return (sig_loss, penalty, total_loss, S, B, Z, ce, ce_no_abs, unc_penalty, cov_penalty,
                    classifier_grads, gato_grads)

        def finetune_train_step_full_batch(X, y, w, mass, cw, cw_no_abs,
                                           update_classifier=True, update_gato=True, rewt=None):
            """
            Eager wrapper around the traced gradient computation above.

            IMPORTANT: apply_gradients() must stay OUT of the @tf.function.
            Passing `update_classifier`/`update_gato` as Python bools makes
            tf.function retrace once per combination, and a Keras optimizer
            creates its slot variables on its first apply_gradients call - so
            leaving the apply inside the graph makes the second trace try to
            create those variables again, which raises
            "tf.function only supports singleton tf.Variables created on the
            first call". Applying out here sidesteps that entirely; the cost
            is negligible because the heavy forward/backward work is still
            compiled.
            """
            (sig_loss, penalty, total_loss, S, B, Z, ce, ce_no_abs, unc_penalty, cov_penalty,
             classifier_grads, gato_grads) = _finetune_grads_full_batch(
                X, y, w, mass, cw, cw_no_abs, update_classifier, update_gato, rewt
            )

            if update_classifier:
                classifier_optimizer.apply_gradients(
                    zip(classifier_grads, model.classifier.trainable_variables)
                )
            if update_gato:
                gato_optimizer.apply_gradients(
                    zip(gato_grads, model.gato.trainable_variables)
                )
            return sig_loss, penalty, total_loss, S, B, Z, ce, ce_no_abs, unc_penalty, cov_penalty

    # ---- validation-driven scheduler / early stopping (mirrors Stage A) ----
    # Classifier only: gato intentionally has NO ReduceLROnPlateau. Its LR is
    # governed entirely by the fixed ExponentialDecay baked into gato_optimizer
    # at construction (see make_schedule(gato_lr, steps=n_epochs // 7) above,
    # keyed to gato_optimizer.iterations - one tick per epoch under the
    # default 1:1 update_schedule). A plateau scheduler on top of that fixed
    # decay would fight it on the same optimizer with no well-defined combined
    # behavior, so this was removed rather than left dead/commented-out - do
    # not re-add a lr_scheduler_gato without first removing the schedule=...
    # passed into gato_optimizer's construction.
    lr_scheduler_classifier = ReduceLROnPlateau(
        classifier_optimizer,
        mode=scheduler_cfg.get("mode", "min"),
        factor=scheduler_cfg.get("factor", 0.5),
        patience=scheduler_cfg.get("patience", 15),
        min_lr=scheduler_cfg.get("min_lr", 1.0e-6),
    )
    if use_validation and chunk_size and int(X_val.shape[0]) > chunk_size:
        val_chunk_size = chunk_size
    else:
        val_chunk_size = None

    best_val_loss = np.inf
    best_classifier_weights_path = None
    best_gato_checkpoint_dir = None
    early_stop_counter = 0

    # Best-checkpoint tracking restricted to the last 20%/40% of epochs, in
    # addition to the plain best-overall tracking above - mirrors
    # gato_categorization_merged.py's update_best() convention (its
    # `epoch_idx >= args.epochs * 0.8` guard), so a run whose lowest
    # val_scalar_loss happened to land early (before the categorization has
    # settled) doesn't hide a more representative "best of the converged
    # tail" checkpoint from evaluation. The 20%/40% windows are measured
    # against the CONFIGURED n_epochs (fixed here, before the loop starts),
    # not however many epochs the run actually completes - early stopping
    # can cut a run short, so anchoring to the configured total keeps the
    # windows well-defined regardless of if/when that happens. Paired
    # classifier+gato checkpoints (same reasoning as finetune_checkpoint_every
    # above) are written permanently under out_dir/finetune_best_checkpoints/
    # every time either window's running best improves; silently skipped
    # (no behavior change) if out_dir is None.
    last20_start_epoch = n_epochs * 0.8
    last40_start_epoch = n_epochs * 0.6
    best_val_loss_last20 = np.inf
    best_val_loss_last40 = np.inf
    best_checkpoints_dir = os.path.join(out_dir, "finetune_best_checkpoints") if out_dir is not None else None
    if best_checkpoints_dir is not None:
        os.makedirs(best_checkpoints_dir, exist_ok=True)

    def _save_paired_best_checkpoint(tag):
        """Writes model.classifier + model.gato under best_checkpoints_dir,
        tagged (e.g. 'overall'/'last20pct'/'last40pct') - the paired-save
        counterpart of the periodic finetune_checkpoint_every snapshots
        above, but selected by validation performance instead of a fixed
        cadence."""
        save_with_retry(
            model.classifier.save_weights,
            os.path.join(best_checkpoints_dir, f"classifier_best_{tag}.weights.h5"),
            label=f"Stage B best-{tag} classifier checkpoint",
        )
        gato_dir = os.path.join(best_checkpoints_dir, f"gato_best_{tag}")
        save_with_retry(
            model.gato.save, gato_dir, ensure_dir=gato_dir,
            label=f"Stage B best-{tag} GMM checkpoint",
        )

    loss_history = []
    val_loss_history = []
    ce_history = []
    ce_no_abs_history = []
    val_ce_history = []
    mean_bias_history = []
    bias_epochs = []
    temperature_history = []
    penalty_history = []
    unc_penalty_history = []
    val_unc_penalty_history = []
    cov_penalty_history = []

    if use_validation:
        print(
            f"[Stage B] Using a held-out validation split for monitoring "
            f"(n_train={n_total}, n_val={int(X_val.shape[0])}); scheduler and "
            f"early stopping (patience={early_stopping_patience}) are driven "
            f"by the validation significance loss.\n"
        )

    for ep in range(n_epochs):
        profiler.start_epoch(ep)
        temperature_scheduler.update(ep)

        if nonres_enabled and ep % nonres_recompute_every == 0:
            # Recomputed from the TRAIN split only, to avoid leaking
            # validation-split information into training (mirrors
            # gato_categorization_merged2.py's identical comment).
            with profiler.phase("nonres_reweight"):
                factors = nonres_reweight_fn(
                    model, X_train, y_train, w_train, mass_train,
                    nonres_class=nonres_class, n_cats=n_cats_gmm,
                    mass_sb_low=nonres_cfg.get("mass_sb_low", 100.0),
                    mass_sb_high=nonres_cfg.get("mass_sb_high", 180.0),
                    mass_sig_low=nonres_cfg.get("mass_sig_low", 120.0),
                    mass_sig_high=nonres_cfg.get("mass_sig_high", 130.0),
                    nbins=scaled_nbins_for_split(nonres_cfg.get("nbins", 40), "train", training_config.get("data_prep")),
                    default_factor=nonres_default_factor,
                    chunk_size=chunk_size,
                    diagnostics_csv_path=(
                        os.path.join(nonres_diag_dir, f"nonres_fit_diagnostics_epoch{ep:04d}.csv")
                        if (nonres_diag_dir is not None and nonres_save_train_csv) else None
                    ),
                    save_fit_plots=(nonres_diag_dir is not None and nonres_save_train_plots),
                    plot_dir=(nonres_diag_dir if (nonres_diag_dir is not None and nonres_save_train_plots) else None),
                    plot_dtype=nonres_plot_dtype,
                    call_label=f"train_epoch{ep:04d}",
                )
                rewt_tf = tf.constant(factors, dtype=tf.float32)
            print(f"[Stage B][{ep:03d}] nonres reweight factors: {factors}")

            if use_validation and nonres_track_val_fit:
                # Diagnostic-only twin of the train-split fit above: same
                # cadence/window, but on X_val/y_val/w_val/mass_val, its
                # own val-scaled nbins (see scaled_nbins_for_split - val
                # has far fewer sideband events than train), and its
                # result is only logged/written to CSV below - it never
                # touches rewt_tf, so training itself is unaffected.
                with profiler.phase("nonres_reweight_val"):
                    factors_val = nonres_reweight_fn(
                        model, X_val, y_val, w_val, mass_val,
                        nonres_class=nonres_class, n_cats=n_cats_gmm,
                        mass_sb_low=nonres_cfg.get("mass_sb_low", 100.0),
                        mass_sb_high=nonres_cfg.get("mass_sb_high", 180.0),
                        mass_sig_low=nonres_cfg.get("mass_sig_low", 120.0),
                        mass_sig_high=nonres_cfg.get("mass_sig_high", 130.0),
                        nbins=scaled_nbins_for_split(nonres_cfg.get("nbins", 40), "val", training_config.get("data_prep")),
                        default_factor=nonres_default_factor,
                        chunk_size=chunk_size,
                        diagnostics_csv_path=(
                            os.path.join(nonres_val_diag_dir, f"nonres_fit_diagnostics_epoch{ep:04d}.csv")
                            if (nonres_val_diag_dir is not None and nonres_save_train_csv) else None
                        ),
                        save_fit_plots=(nonres_val_diag_dir is not None and nonres_save_train_plots),
                        plot_dir=(nonres_val_diag_dir if (nonres_val_diag_dir is not None and nonres_save_train_plots) else None),
                        plot_dtype=nonres_plot_dtype,
                        call_label=f"val_epoch{ep:04d}",
                    )
                print(f"[Stage B][{ep:03d}] nonres reweight factors (VAL, diagnostic only): {factors_val}")

        with profiler.phase("train_steps"):
            for step_idx, (upd_cls, upd_gato) in enumerate(step_plan):
                if use_chunked:
                    sig_loss, penY, total_loss, S, B, Z, ce, ce_no_abs, unc_pen, cov_pen = finetune_train_step_chunked(
                        model, X_all_t, y_all_t, w_all_t, mass_all_t, chunk_size,
                        classifier_optimizer, gato_optimizer,
                        ce_loss_fn, lam_penalty, penalty_thr, dropout_in_finetune,
                        cw_all=cw_all_t, ce_weight_sum=ce_weight_sum,
                        cw_all_no_abs=cw_all_no_abs_t, ce_weight_sum_no_abs=ce_weight_sum_no_abs,
                        penalty_in_gato_loss=penalty_in_gato_loss,
                        penalty_in_classifier_loss=penalty_in_classifier_loss,
                        significance_in_classifier_loss=significance_in_classifier_loss,
                        lam_unc=lam_unc, bkg_unc_threshold=bkg_unc_threshold,
                        lam_cov=lam_cov,
                        update_classifier=upd_cls, update_gato=upd_gato,
                        profiler=profiler,
                        rewt=rewt_tf,
                    )
                else:
                    sig_loss, penY, total_loss, S, B, Z, ce, ce_no_abs, unc_pen, cov_pen = finetune_train_step_full_batch(
                        X_all_t, y_all_t, w_all_t, mass_all_t, cw_all_t, cw_all_no_abs_t,
                        upd_cls, upd_gato, rewt=rewt_tf
                    )
                if step_history_csv_path is not None:
                    append_step_history_row(
                        step_history_csv_path, ep, step_idx, passes_per_epoch, upd_cls, upd_gato,
                        sig_loss.numpy(), penY.numpy(), ce.numpy(), ce_no_abs.numpy(), Z.numpy(), total_loss.numpy(),
                    )
        # The reported metrics are those of the LAST inner step, i.e. measured
        # at the most recent parameters (still, as before, *before* that
        # step's own update is applied).

        loss_history.append(float(sig_loss.numpy()))
        penalty_history.append(float(penY.numpy()))
        ce_history.append(float(ce.numpy()))
        ce_no_abs_history.append(float(ce_no_abs.numpy()))
        unc_penalty_history.append(float(unc_pen.numpy()))
        cov_penalty_history.append(float(cov_pen.numpy()))

        # Periodic PAIRED classifier+gato checkpoint (see
        # finetune_checkpoint_every in this function's docstring) - saved
        # on a fixed cadence, independent of validation performance, so
        # intermediate epochs stay recoverable even when they were never
        # the running best. Both halves are always written together so a
        # later reload evaluates the classifier against the SAME epoch's
        # GMM state, not a mismatched pairing.
        if finetune_checkpoint_dir is not None and (ep + 1) % finetune_checkpoint_every == 0:
            ckpt_classifier_path = os.path.join(
                finetune_checkpoint_dir, f"classifier_finetune_epoch{ep:04d}.weights.h5"
            )
            save_with_retry(
                model.classifier.save_weights, ckpt_classifier_path,
                label=f"Stage B periodic classifier checkpoint (epoch {ep:03d})",
            )
            ckpt_gato_dir = os.path.join(finetune_checkpoint_dir, f"gato_finetune_epoch{ep:04d}")
            save_with_retry(
                model.gato.save, ckpt_gato_dir, ensure_dir=ckpt_gato_dir,
                label=f"Stage B periodic GMM checkpoint (epoch {ep:03d})",
            )
            print(f"[Stage B] Saved periodic checkpoint pair (classifier+gato) at epoch {ep:03d}: "
                  f"{finetune_checkpoint_dir}")

        log_line = (
            f"[Stage B][{ep:03d}] Z={Z.numpy():.4f} sig_loss={sig_loss.numpy():.4f} "
            f"penalty={penY.numpy():.4f} ce={ce.numpy():.4f}"
        )

        if use_validation:
            with profiler.phase("validation"):
                val_sig_loss, val_penalty, val_scalar_loss, val_Z, val_ce, val_unc_penalty = evaluate_complete_model_chunked(
                    model, X_val, y_val, w_val, mass_val, val_chunk_size, lam_penalty, penalty_thr,
                    # CE is always computed here too, purely for monitoring -
                    # it never enters val_scalar_loss (used for the scheduler /
                    # early stopping / best-checkpoint selection).
                    ce_loss_fn=ce_loss_fn, cw=cw_val, rewt=rewt_tf,
                    lam_unc=lam_unc, bkg_unc_threshold=bkg_unc_threshold,
                )
            val_loss_history.append(val_sig_loss)
            val_ce_history.append(val_ce)
            val_unc_penalty_history.append(val_unc_penalty)
            log_line += (
                f" | val_Z={val_Z:.4f} val_sig_loss={val_sig_loss:.4f} "
                f"val_penalty={val_penalty:.4f} val_ce={val_ce:.4f} val_unc_penalty={val_unc_penalty:.4f}"
            )

            lr_scheduler_classifier.step(val_scalar_loss)

            if val_scalar_loss < best_val_loss:
                best_val_loss = val_scalar_loss
                if best_classifier_weights_path is None:
                    fd, best_classifier_weights_path = tempfile.mkstemp(suffix=".weights.h5")
                    os.close(fd)
                if best_gato_checkpoint_dir is None:
                    best_gato_checkpoint_dir = tempfile.mkdtemp(prefix="gato_best_ckpt_")
                model.classifier.save_weights(best_classifier_weights_path)
                model.gato.save(best_gato_checkpoint_dir)
                if mass_window_enabled:
                    best_mass_low = float(model.mass_low.numpy())
                    best_mass_high = float(model.mass_high.numpy())
                early_stop_counter = 0
                if best_checkpoints_dir is not None:
                    _save_paired_best_checkpoint("overall")
            else:
                early_stop_counter += 1

            # Windowed "best of the last 20%/40% of epochs" checkpoints -
            # independent of (and in addition to) the best-overall tracking
            # above; see this function's docstring / the setup comment near
            # best_val_loss_last20/last40 for the rationale. The last-20%
            # window is a subset of the last-40% window, so both can update
            # in the same epoch - that's expected, not a bug.
            if best_checkpoints_dir is not None:
                if ep >= last40_start_epoch and val_scalar_loss < best_val_loss_last40:
                    best_val_loss_last40 = val_scalar_loss
                    _save_paired_best_checkpoint("last40pct")
                if ep >= last20_start_epoch and val_scalar_loss < best_val_loss_last20:
                    best_val_loss_last20 = val_scalar_loss
                    _save_paired_best_checkpoint("last20pct")

        # NOTE: this used to read `if ep % 1 == 0`, which is true for EVERY
        # epoch - so the purely diagnostic block below (a full extra forward
        # pass over the whole training set, plus a host-side copy of every
        # event's scores and weights) ran every single epoch despite the
        # comment further down saying "only every 10th". `diagnostics_every`
        # restores the intended behaviour and makes the interval tunable.
        #
        # NOTE: the per-epoch console print used to live INSIDE this gated
        # block too, so raising diagnostics_every (e.g. to 10 or 20) used to
        # also silence per-epoch progress printing as a side effect. It's
        # now unconditional (see `print(log_line)` after this if/else) -
        # only the expensive bias/temperature/confusion computation itself
        # stays on the reduced cadence.
        if ep % diagnostics_every == 0 or ep == n_epochs - 1:
            with profiler.phase("diagnostics"):
                if use_chunked:
                    data_dict = build_data_dict_chunked(
                        model, X_all_t, y_all_t, w_all_t, chunk_size, mass_all=mass_all_t
                    )
                else:
                    data_dict, _, _ = model.build_data_dict(
                        X_all_t, y_all_t, w_all_t, mass=mass_all_t, training=False
                    )
                bias_vec = model.gato.get_bias(data_dict)
                mean_bias_history.append(float(np.mean(np.abs(bias_vec))))
                bias_epochs.append(ep)
                temp_val = float(model.gato.temperature.numpy()) if hasattr(model.gato.temperature, "numpy") else float(model.gato.temperature)
                temperature_history.append(temp_val)

                # Confusion-matrix counts (TP/FP/TN/FN, precision/recall) -
                # same cadence as the bias/temperature diagnostics above,
                # reusing this same profiler.phase("diagnostics") bracket
                # rather than adding a new one. The classifier's own
                # forward pass here is cheap relative to the GMM work
                # already happening in this block (see docstring). One
                # growing table per split (confusion_counts_{train,val}.csv),
                # one row appended per diagnostics epoch, not one file per
                # epoch - see append_confusion_counts_row.
                if confusion_dir is not None:
                    cm_w_train, cm_raw_train = compute_confusion_counts_from_arrays(
                        model.classifier, X_all_t, y_all_t, w_all_t, n_classes, chunk_size=chunk_size,
                    )
                    append_confusion_counts_row(
                        os.path.join(confusion_dir, "confusion_counts_train.csv"),
                        ep, cm_w_train, cm_raw_train, class_names=confusion_class_names,
                    )
                    if use_validation:
                        cm_w_val, cm_raw_val = compute_confusion_counts_from_arrays(
                            model.classifier, X_val, y_val, w_val, n_classes, chunk_size=val_chunk_size,
                        )
                        append_confusion_counts_row(
                            os.path.join(confusion_dir, "confusion_counts_val.csv"),
                            ep, cm_w_val, cm_raw_val, class_names=confusion_class_names,
                        )
            log_line += f" temp={temp_val:.3f}"

        profiler.end_epoch()
        log_line += profiler.epoch_line()
        print(log_line)

        # Periodic checkpoint pair at the discriminant-density cadence - see
        # this function's setup comment near disc_checkpoints_dir for why
        # this replaced live histogram computation/saving.
        if disc_enabled and (ep % disc_recompute_every == 0 or ep == n_epochs - 1):
            with profiler.phase("discriminant_checkpoints"):
                save_with_retry(
                    model.classifier.save_weights,
                    os.path.join(disc_checkpoints_dir, f"classifier_epoch{ep:04d}.weights.h5"),
                    label=f"Discriminant-density-cadence classifier checkpoint (epoch {ep:03d})",
                )
                gato_ckpt_dir = os.path.join(disc_checkpoints_dir, f"gato_epoch{ep:04d}")
                save_with_retry(
                    model.gato.save, gato_ckpt_dir, ensure_dir=gato_ckpt_dir,
                    label=f"Discriminant-density-cadence GMM checkpoint (epoch {ep:03d})",
                )

        if use_validation and early_stop_counter >= early_stopping_patience:
            print(f"[Stage B] Early stopping at epoch {ep} (best val loss = {best_val_loss:.4f})")
            break

    if use_validation and best_classifier_weights_path is not None:
        model.classifier.load_weights(best_classifier_weights_path)
        model.gato.restore(best_gato_checkpoint_dir)
        os.remove(best_classifier_weights_path)
        shutil.rmtree(best_gato_checkpoint_dir, ignore_errors=True)

    profiler.report(n_rows=n_total)
    profiler.save(profile_out_path, extra={
        "stage": "B",
        "n_train_rows": int(n_total),
        "n_val_rows": int(X_val.shape[0]) if use_validation else 0,
        "n_features": int(X_train.shape[1]),
        "feature_dtype": str(getattr(X_train, "dtype", "unknown")),
        "chunk_size": int(chunk_size) if chunk_size else None,
        "use_chunked": bool(use_chunked),
        "classifier_steps_per_epoch": int(class_steps),
        "gato_steps_per_epoch": int(gato_steps),
        "passes_per_epoch": int(passes_per_epoch),
        "diagnostics_every": int(diagnostics_every),
        "finetune_checkpoint_every": int(finetune_checkpoint_every),
        "classifier_has_stage_b_objective": bool(significance_in_classifier_loss),
        "single_tape": bool(
            (not significance_in_classifier_loss)
            or (penalty_in_gato_loss == penalty_in_classifier_loss)
        ),
    })

    # Restore whatever model.classifier.trainable was before this call (see
    # the setup comment near classifier_was_trainable above) - this
    # function should have no surprising side effects on the model object
    # for any caller.
    model.classifier.trainable = classifier_was_trainable

    return (
        loss_history, penalty_history, mean_bias_history, bias_epochs,
        temperature_history, val_loss_history,
        ce_history, val_ce_history,
        ce_no_abs_history,
        unc_penalty_history, val_unc_penalty_history, cov_penalty_history,
    )