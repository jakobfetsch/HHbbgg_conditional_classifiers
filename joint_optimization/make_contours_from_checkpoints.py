

#!/usr/bin/env python3
"""
make_contours_from_checkpoints.py

Vereinfachte Version von evaluate_model_detailed_contours.py.

Erzeugt NUR die Discriminant-Density-Contour-Plots (Teil B/C des
Originalskripts) aus bereits gespeicherten Classifier-Checkpoints - fuer
alle drei Split-Sets (train/val/test), inklusive der vollstaendigen
zweistufigen Preselection (Stage-A Klassifikator-Preselection + Stage-B
statische Cuts + score-basierter Cut), genau wie im Original.

Alles andere aus dem Originalskript (CSV-History-Plots, Test-Set-
Evaluation, mgg-Cut-Sweep, Mass-Histogramme, ...) wurde entfernt.

--------------------------------------------------------------------------
Benutzung
--------------------------------------------------------------------------
Unten in CHECKPOINT_DIRS die Liste der Trainings-out-dirs eintragen, fuer
die Contour-Plots erzeugt werden sollen (jedes Verzeichnis muss
training_info.txt sowie classifier_finetuned.weights.h5 enthalten - also
genau das, worauf im Originalskript --out-dir gezeigt hat).

ANNAHME: alle Verzeichnisse in CHECKPOINT_DIRS gehoeren zu Runs mit dem
GLEICHEN Datensatz (gleiches base_dir / data_config / fraction / eras /
seeds / preselection-Setup). Die Trainingsdaten werden deshalb NUR EINMAL
geladen (Konfiguration wird aus CHECKPOINT_DIRS[0]/training_info.txt
gelesen) und fuer alle Checkpoints wiederverwendet - das ist der Teil, der
den Prozess beschleunigt. Nur die score-basierte Preselection (die vom
jeweils finalen Klassifikator eines Runs abhaengt) wird pro
Checkpoint-Verzeichnis neu angewendet. Falls eure Checkpoints tatsaechlich
zu unterschiedlichen Datensaetzen gehoeren, dieses Skript pro Gruppe
getrennt ausfuehren (das Skript warnt, falls es einen Unterschied in
base_dir/data_config_path erkennt).

Fuer jedes Verzeichnis in CHECKPOINT_DIRS:
  - werden, falls vorhanden, die periodischen
    finetune_checkpoints/classifier_finetune_epoch*.weights.h5 verwendet
    (bis zu MAX_CHECKPOINT_EPOCHS gleichmaessig verteilte Epochen)
  - sonst die finalen/besten Gewichte (classifier_finetuned.weights.h5)
  - die Plots landen jeweils unter <checkpoint_dir>/contours_checkpoint/

Ausfuehren mit:
    python make_contours_from_checkpoints.py
(keine Kommandozeilenargumente noetig - alles wird oben im Skript
konfiguriert)

Hinweis: fuer die Contour-Histogramme wird ausschliesslich der Classifier
gebraucht (nicht das GMM/gato) - siehe compute_class_dim_histograms im
Original, das nur wrapper.classifier anfasst. gato wird hier deshalb gar
nicht erst geladen.

Es wird KEINE Zwischen-CSV geschrieben/gelesen: pro Checkpoint wird
compute_class_dim_histograms() direkt aufgerufen und das Ergebnis sofort
mit plot_contours_from_histograms() geplottet (beides 1:1 aus dem
Original uebernommen - genau der Mechanismus, den das Original fuer den
mgg-Cut-Sweep verwendet, um den CSV-Roundtrip zu vermeiden).
"""

import glob
import json
import os
import re
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

try:
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    _HAVE_MPLHEP = True
except ImportError:
    _HAVE_MPLHEP = False

from mlp_tf import MLP

from end_to_end_helpers import (
    GMMFromScores,
    static_preselection_columns,
    build_static_preselection_cuts,
    build_score_preselection_cut,
    compute_preselection_cutflow,
    compute_class_scores,
)

from dataset_prep import (
    DEFAULT_ERAS, load_config as load_data_config, build_training_dataset,
    classifier_preselection_kwargs,
)

from discriminant_density_diagnostics import (
    compute_class_dim_histograms,
    styled_class_contours,
    _merge_bins as _dd_merge_bins,
    _grid_shape as _dd_grid_shape,
    DEFAULT_CHUNK_SIZE as DISC_DENSITY_DEFAULT_CHUNK_SIZE,
)


def _draw_cms_label(ax, main_text, right_label, fontsize=14):
    """1:1 aus plot_history.py uebernommen, fuer einheitliches Styling
    ueber alle Plots dieses Projekts hinweg - siehe dort fuer die volle
    Begruendung der Fallback-Kette."""
    if _HAVE_MPLHEP:
        for kwargs in (
            {"text": main_text, "data": True, "com": None, "rlabel": right_label},
            {"label": main_text, "data": True, "com": None, "rlabel": right_label},
            {"text": main_text, "data": True, "rlabel": right_label},
            {"label": main_text, "data": True, "rlabel": right_label},
        ):
            try:
                hep.cms.label(ax=ax, fontsize=fontsize, **kwargs)
                return
            except TypeError:
                continue
        try:
            hep.cms.label(ax, main_text)
            return
        except Exception:
            pass
    ax.text(0.0, 1.02, "CMS", transform=ax.transAxes, fontsize=fontsize,
             fontweight="bold", va="bottom", ha="left")
    ax.text(0.13, 1.02, main_text, transform=ax.transAxes, fontsize=fontsize - 3,
             fontstyle="italic", va="bottom", ha="left")
    if right_label:
        ax.text(1.0, 1.02, right_label, transform=ax.transAxes, fontsize=fontsize - 3,
                 va="bottom", ha="right")


# =============================================================================
# KONFIGURATION - hier anpassen
# =============================================================================

# Liste der Trainings-out-dirs, deren Checkpoints geplottet werden sollen.
CHECKPOINT_DIRS = [
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_05",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_10",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_20",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_40",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_80",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_200_epochs_10_bins_soft_class_loss_140",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_150_epochs_10_bins_soft_class_loss",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_250_epochs_15_bins_soft_class_loss1",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_350_epochs_5_bins_soft_class_loss2",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_400_epochs_7_bins_soft_class_loss",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_450_epochs_5_bins_soft_class_loss11",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_100_epochs_5_bins_soft_class_loss",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_150_epochs_7_bins_soft_class_loss1",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_150_epochs_10_bins_soft_class_loss3",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_150_epochs_15_bins_soft_class_loss10",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_250_epochs_10_bins_soft_class_loss2",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_200_epochs_7_bins_soft_class_loss7",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_200_epochs_7_bins_soft_class_loss11",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_200_epochs_10_bins_soft_class_loss5",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_250_epochs_15_bins_soft_class_loss2",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_300_epochs_5_bins_soft_class_loss",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_450_epochs_5_bins_soft_class_loss13"
]

CHECKPOINT_DIRS = [
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_05"
]

# "Aufloesung" der Contour-Plots: Anzahl der Bins pro Achse fuer die
# Discriminant-Density-Histogramme, aus denen die Konturen berechnet
# werden (entspricht discriminant_density_diagnostics.bins im
# Trainings-Config, dortiger Default: 160). Hoeher = feinere/koernigere
# Aufloesung (ggf. zusammen mit CONTOUR_MERGE_BINS/CONTOUR_SMOOTH_SIGMA
# unten anpassen, falls die Konturen dann wie einzelne Punkte aussehen).
CONTOUR_BINS = 160

# Wertebereich der Klassifikator-Scores pro Achse.
CONTOUR_SCORE_RANGE = (0.0, 1.0)

# Glaettung/Merging der Contour-Plots (Defaultwerte aus dem Original
# uebernommen) - macht aus einem feinen, evtl. spaerlich besetzten Grid
# wieder glatte Konturflaechen.
CONTOUR_SMOOTH_SIGMA = 1.0
CONTOUR_MERGE_BINS = 8

# Wie viele periodische finetune_checkpoints/-Epochen pro Verzeichnis
# hoechstens geplottet werden (gleichmaessig ueber alle gespeicherten
# Epochen verteilt, falls mehr vorhanden sind).
MAX_CHECKPOINT_EPOCHS = 6

CUDA_DEVICE = 1  # -1 = CPU
PLOT_DTYPE = "pdf"
CONTOUR_SUBDIR_NAME = "contours_checkpoint"

# Zusaetzlich zum vollen Grid (alle Dimensions-Kombinationen) noch einen
# zweiten, kompakteren Plot mit nur diesen 3 (dim_i, dim_j)-Paaren
# erzeugen (Dateiname bekommt das Suffix "_3d" bzw. was auch immer ihr
# unten bei COMPACT_SUFFIX eintragt). Auf None setzen, um den kompakten
# Plot ganz wegzulassen. Die Paare muessen unter den in den Histogrammen
# vorhandenen (i, j)-Kombinationen sein (0-indiziert, i < j).
CONTOUR_DIM_PAIRS_COMPACT = [(0, 1), (0, 2), (1, 2)]
COMPACT_SUFFIX = "_3d"

# Achsenbeschriftung: "P(<class>)" ist nicht besonders treffend fuer einen
# GMM-Discriminanten - AXIS_LABEL_STYLE waehlt zwischen ein paar
# Alternativen (siehe axis_label() weiter unten fuer die exakten Strings):
#   "discriminant_class" -> "ggF HH discriminant (dim. 0)"
#   "discriminant_generic"    -> "Discriminant (dim. 0)"
#   "score_class"              -> "ggF HH classifier score (dim. 0)"
AXIS_LABEL_STYLE = "discriminant_generic"

# Anzeigenamen der Klassen (fuer Achsenbeschriftungen UND Legende).
# WICHTIG: die Keys links muessen exakt den INTERNEN Klassennamen
# entsprechen, wie sie aus dem Datensatz kommen (classes-Liste) - das
# Skript druckt diese beim Start aus ("[INFO] Klassen (intern): [...]"),
# einfach dort nachschauen und die Keys unten ggf. anpassen. Klassen, die
# hier nicht auftauchen, werden unveraendert (mit ihrem internen Namen)
# angezeigt.
CLASS_DISPLAY_NAMES = {
    "ggHH": "ggF HH",
    "VBFHH": "VBF HH",
    "nonRes": "nonres. background",
    "resBkg": "res. background",
}

# CMS-Label (wie in plot_history.py) - main_text links neben "CMS", right_label
# oben rechts (z.B. leer lassen, oder z.B. euren Run-/Datensatznamen eintragen).
CMS_MAIN_TEXT = "Private Work"
CMS_RIGHT_LABEL = ""

AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 9
CMS_LABEL_FONTSIZE = 14

# dataset_prep-Spaltenname -> logischer Name, den build_static_preselection_cuts erwartet
EXTRA_COLUMN_LOGICAL_NAME = {
    "nonResReg_vbfpair_dijet_mass_DNNreg": "dijet_mass",
    "lead_mvaID": "lead_mvaid",
    "sublead_mvaID": "sublead_mvaid",
    "is_boosted": "is_boosted",
}


#!/usr/bin/env python3
"""
make_contours_from_checkpoints.py

Vereinfachte Version von evaluate_model_detailed_contours.py.

Erzeugt NUR die Discriminant-Density-Contour-Plots (Teil B/C des
Originalskripts) aus bereits gespeicherten Classifier-Checkpoints - fuer
alle drei Split-Sets (train/val/test), inklusive der vollstaendigen
zweistufigen Preselection (Stage-A Klassifikator-Preselection + Stage-B
statische Cuts + score-basierter Cut), genau wie im Original.

Alles andere aus dem Originalskript (CSV-History-Plots, Test-Set-
Evaluation, mgg-Cut-Sweep, Mass-Histogramme, ...) wurde entfernt.

--------------------------------------------------------------------------
Benutzung
--------------------------------------------------------------------------
Unten in CHECKPOINT_DIRS die Liste der Trainings-out-dirs eintragen, fuer
die Contour-Plots erzeugt werden sollen (jedes Verzeichnis muss
training_info.txt sowie classifier_finetuned.weights.h5 enthalten - also
genau das, worauf im Originalskript --out-dir gezeigt hat).

ANNAHME: alle Verzeichnisse in CHECKPOINT_DIRS gehoeren zu Runs mit dem
GLEICHEN Datensatz (gleiches base_dir / data_config / fraction / eras /
seeds / preselection-Setup). Die Trainingsdaten werden deshalb NUR EINMAL
geladen (Konfiguration wird aus CHECKPOINT_DIRS[0]/training_info.txt
gelesen) und fuer alle Checkpoints wiederverwendet - das ist der Teil, der
den Prozess beschleunigt. Nur die score-basierte Preselection (die vom
jeweils finalen Klassifikator eines Runs abhaengt) wird pro
Checkpoint-Verzeichnis neu angewendet. Falls eure Checkpoints tatsaechlich
zu unterschiedlichen Datensaetzen gehoeren, dieses Skript pro Gruppe
getrennt ausfuehren (das Skript warnt, falls es einen Unterschied in
base_dir/data_config_path erkennt).

Fuer jedes Verzeichnis in CHECKPOINT_DIRS:
  - werden, falls vorhanden, die periodischen
    finetune_checkpoints/classifier_finetune_epoch*.weights.h5 verwendet
    (bis zu MAX_CHECKPOINT_EPOCHS gleichmaessig verteilte Epochen)
  - sonst die finalen/besten Gewichte (classifier_finetuned.weights.h5)
  - die Plots landen jeweils unter <checkpoint_dir>/contours_checkpoint/

Ausfuehren mit:
    python make_contours_from_checkpoints.py
(keine Kommandozeilenargumente noetig - alles wird oben im Skript
konfiguriert)

Hinweis: fuer die Contour-Histogramme wird ausschliesslich der Classifier
gebraucht (nicht das GMM/gato) - siehe compute_class_dim_histograms im
Original, das nur wrapper.classifier anfasst. gato wird hier deshalb gar
nicht erst geladen.

Es wird KEINE Zwischen-CSV geschrieben/gelesen: pro Checkpoint wird
compute_class_dim_histograms() direkt aufgerufen und das Ergebnis sofort
mit plot_contours_from_histograms() geplottet (beides 1:1 aus dem
Original uebernommen - genau der Mechanismus, den das Original fuer den
mgg-Cut-Sweep verwendet, um den CSV-Roundtrip zu vermeiden).
"""

import glob
import json
import os
import re
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

try:
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    _HAVE_MPLHEP = True
except ImportError:
    _HAVE_MPLHEP = False

from mlp_tf import MLP

from end_to_end_helpers import (
    GMMFromScores,
    static_preselection_columns,
    build_static_preselection_cuts,
    build_score_preselection_cut,
    compute_preselection_cutflow,
    compute_class_scores,
)

from dataset_prep import (
    DEFAULT_ERAS, load_config as load_data_config, build_training_dataset,
    classifier_preselection_kwargs,
)

from discriminant_density_diagnostics import (
    compute_class_dim_histograms,
    styled_class_contours,
    _merge_bins as _dd_merge_bins,
    _grid_shape as _dd_grid_shape,
    DEFAULT_CHUNK_SIZE as DISC_DENSITY_DEFAULT_CHUNK_SIZE,
)


def _draw_cms_label(ax, main_text, right_label, fontsize=14):
    """1:1 aus plot_history.py uebernommen, fuer einheitliches Styling
    ueber alle Plots dieses Projekts hinweg - siehe dort fuer die volle
    Begruendung der Fallback-Kette."""
    if _HAVE_MPLHEP:
        for kwargs in (
            {"text": main_text, "data": True, "com": None, "rlabel": right_label},
            {"label": main_text, "data": True, "com": None, "rlabel": right_label},
            {"text": main_text, "data": True, "rlabel": right_label},
            {"label": main_text, "data": True, "rlabel": right_label},
        ):
            try:
                hep.cms.label(ax=ax, fontsize=fontsize, **kwargs)
                return
            except TypeError:
                continue
        try:
            hep.cms.label(ax, main_text)
            return
        except Exception:
            pass
    ax.text(0.0, 1.02, "CMS", transform=ax.transAxes, fontsize=fontsize,
             fontweight="bold", va="bottom", ha="left")
    ax.text(0.13, 1.02, main_text, transform=ax.transAxes, fontsize=fontsize - 3,
             fontstyle="italic", va="bottom", ha="left")
    if right_label:
        ax.text(1.0, 1.02, right_label, transform=ax.transAxes, fontsize=fontsize - 3,
                 va="bottom", ha="right")


# =============================================================================
# KONFIGURATION - hier anpassen
# =============================================================================

# Liste der Trainings-out-dirs, deren Checkpoints geplottet werden sollen.
CHECKPOINT_DIRS = [
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_05",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_10",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_20",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_40",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_80",
    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_200_epochs_10_bins_soft_class_loss_140",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_150_epochs_10_bins_soft_class_loss",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_250_epochs_15_bins_soft_class_loss1",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_350_epochs_5_bins_soft_class_loss2",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_400_epochs_7_bins_soft_class_loss",
    "/net/data_cms3a-1/fetsch/results/reruns/hard_class_same_bins/3D_100_450_epochs_5_bins_soft_class_loss11",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_100_epochs_5_bins_soft_class_loss",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_150_epochs_7_bins_soft_class_loss1",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_150_epochs_10_bins_soft_class_loss3",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_150_epochs_15_bins_soft_class_loss10",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_1_10/3D_100_250_epochs_10_bins_soft_class_loss2",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_200_epochs_7_bins_soft_class_loss7",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_200_epochs_7_bins_soft_class_loss11",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_200_epochs_10_bins_soft_class_loss5",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_250_epochs_15_bins_soft_class_loss2",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_300_epochs_5_bins_soft_class_loss",
    "/net/data_cms3a-1/fetsch/results/reruns/soft_class_same_bins/3D_100_450_epochs_5_bins_soft_class_loss13"
]

# CHECKPOINT_DIRS = [
#    "/net/data_cms3a-1/fetsch/results/reruns/diff_pretrain/3D_100_400_epochs_10_bins_soft_class_loss_05"
#]

# "Aufloesung" der Contour-Plots: Anzahl der Bins pro Achse fuer die
# Discriminant-Density-Histogramme, aus denen die Konturen berechnet
# werden (entspricht discriminant_density_diagnostics.bins im
# Trainings-Config, dortiger Default: 160). Hoeher = feinere/koernigere
# Aufloesung (ggf. zusammen mit CONTOUR_MERGE_BINS/CONTOUR_SMOOTH_SIGMA
# unten anpassen, falls die Konturen dann wie einzelne Punkte aussehen).
CONTOUR_BINS = 160

# Wertebereich der Klassifikator-Scores pro Achse.
CONTOUR_SCORE_RANGE = (0.0, 1.0)

# Glaettung/Merging der Contour-Plots (Defaultwerte aus dem Original
# uebernommen) - macht aus einem feinen, evtl. spaerlich besetzten Grid
# wieder glatte Konturflaechen.
CONTOUR_SMOOTH_SIGMA = 1.0
CONTOUR_MERGE_BINS = 8

# Wie viele periodische finetune_checkpoints/-Epochen pro Verzeichnis
# hoechstens geplottet werden (gleichmaessig ueber alle gespeicherten
# Epochen verteilt, falls mehr vorhanden sind).
MAX_CHECKPOINT_EPOCHS = 6

CUDA_DEVICE = 1  # -1 = CPU
PLOT_DTYPE = "png"
CONTOUR_SUBDIR_NAME = "contours_checkpoint"

# Zusaetzlich zum vollen Grid (alle Dimensions-Kombinationen) noch einen
# zweiten, kompakteren Plot mit nur diesen 3 (dim_i, dim_j)-Paaren
# erzeugen (Dateiname bekommt das Suffix "_3d" bzw. was auch immer ihr
# unten bei COMPACT_SUFFIX eintragt). Auf None setzen, um den kompakten
# Plot ganz wegzulassen. Die Paare muessen unter den in den Histogrammen
# vorhandenen (i, j)-Kombinationen sein (0-indiziert, i < j).
CONTOUR_DIM_PAIRS_COMPACT = [(0, 1), (0, 2), (1, 2)]
COMPACT_SUFFIX = "_3d"

# Achsenbeschriftung: "P(<class>)" ist nicht besonders treffend fuer einen
# GMM-Discriminanten - AXIS_LABEL_STYLE waehlt zwischen ein paar
# Alternativen (siehe axis_label() weiter unten fuer die exakten Strings):
#   "discriminant_class" -> "ggF HH discriminant (dim. 0)"
#   "discriminant_generic"    -> "Discriminant (dim. 0)"
#   "score_class"              -> "ggF HH classifier score (dim. 0)"
AXIS_LABEL_STYLE = "discriminant_generic"

# Anzeigenamen der Klassen (fuer Achsenbeschriftungen UND Legende).
# WICHTIG: die Keys links muessen exakt den INTERNEN Klassennamen
# entsprechen, wie sie aus dem Datensatz kommen (classes-Liste) - das
# Skript druckt diese beim Start aus ("[INFO] Klassen (intern): [...]"),
# einfach dort nachschauen und die Keys unten ggf. anpassen. Klassen, die
# hier nicht auftauchen, werden unveraendert (mit ihrem internen Namen)
# angezeigt.
CLASS_DISPLAY_NAMES = {
    "is_ggHH_sig": "gg HH",
    "is_VBFHH_sig": "VBF HH",
    "is_nonRes_bkg": "nonres. background",
    "is_Res_bkg": "res. background",
}

# CMS-Label (wie in plot_history.py) - main_text links neben "CMS", right_label
# oben rechts (z.B. leer lassen, oder z.B. euren Run-/Datensatznamen eintragen).
CMS_MAIN_TEXT = "Private Work"
CMS_RIGHT_LABEL = ""

AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 9
CMS_LABEL_FONTSIZE = 14

# dataset_prep-Spaltenname -> logischer Name, den build_static_preselection_cuts erwartet
EXTRA_COLUMN_LOGICAL_NAME = {
    "nonResReg_vbfpair_dijet_mass_DNNreg": "dijet_mass",
    "lead_mvaID": "lead_mvaid",
    "sublead_mvaID": "sublead_mvaid",
    "is_boosted": "is_boosted",
}


# =============================================================================
# Hilfsfunktionen (aus evaluate_model_detailed_contours.py uebernommen /
# leicht vereinfacht)
# =============================================================================
def setup_device(cuda_device):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[INFO] Kein GPU sichtbar - laeuft auf CPU.")
        return
    if cuda_device is None or cuda_device < 0 or cuda_device >= len(gpus):
        print(f"[INFO] CUDA_DEVICE={cuda_device} kein gueltiger GPU-Index "
              f"({len(gpus)} verfuegbar) - laeuft auf CPU.")
        return
    try:
        selected = gpus[cuda_device]
        tf.config.set_visible_devices(selected, "GPU")
        tf.config.experimental.set_memory_growth(selected, True)
        print(f"[INFO] Verwende GPU {cuda_device}: {selected}")
    except RuntimeError as e:
        print(f"[WARNING] Konnte GPU {cuda_device} nicht auswaehlen ({e}); "
              f"laeuft mit TF-Default weiter.")


def parse_training_info(out_dir):
    """Wie im Original: findet und parst training_info.txt in out_dir."""
    candidate = os.path.join(out_dir, "training_info.txt")
    if os.path.exists(candidate):
        info_path = candidate
    else:
        matches = sorted(glob.glob(os.path.join(out_dir, "*info*.txt")))
        if not matches:
            raise FileNotFoundError(
                f"Kein training_info.txt (oder *info*.txt) unter {out_dir!r} gefunden."
            )
        info_path = matches[0]
        print(f"[WARNING] training_info.txt nicht gefunden; verwende stattdessen {info_path!r}.")

    with open(info_path, "r") as f:
        text = f.read()

    def _grab(label):
        m = re.search(rf"^{re.escape(label)}\s*(.*)$", text, flags=re.MULTILINE)
        val = m.group(1).strip() if m else ""
        return val if val and val.lower() != "none" else None

    base_dir = _grab("individual_samples base-dir:")
    data_config_path = _grab("Data config path:")

    marker = "EFFECTIVE TRAINING CONFIG (training_config.yaml)"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(
            f"{info_path!r} enthaelt nicht den Abschnitt "
            f"'EFFECTIVE TRAINING CONFIG (training_config.yaml)' - Config kann nicht "
            f"rekonstruiert werden."
        )
    rest = text[idx:].split("\n", 2)
    yaml_text = rest[2] if len(rest) > 2 else ""
    training_config = yaml.safe_load(yaml_text)
    if not isinstance(training_config, dict):
        raise ValueError(f"Konnte keinen brauchbaren training_config dict aus {info_path!r} parsen.")

    return {
        "training_config": training_config,
        "base_dir": base_dir,
        "data_config_path": data_config_path,
        "info_path": info_path,
    }


def build_classifier(training_config, input_size, output_size):
    """Baut nur den (untrainierten) Classifier - exakt wie
    build_classifier_and_gato im Original, aber ohne das GMM/gato, da das
    fuer die Contour-Histogramme nicht gebraucht wird."""
    data_prep_cfg = training_config.get("data_prep", {}) or {}
    random_search_dir = data_prep_cfg.get("random_search_dir", None)
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
        print(f"[INFO] Verwende Architektur aus {best_params_path}: {best_params}")
    else:
        num_layers, num_nodes, act_fn_name, dropout_prob = training_config["class_model"].values()

    classifier = MLP(input_size, num_layers, num_nodes, output_size, act_fn_name, dropout_prob)
    classifier(tf.zeros((1, input_size), dtype=tf.float32), training=False)  # Variablen bauen
    return classifier


def load_dataset(base_dir, data_config, fraction, seed, eras, train_fraction, val_fraction,
                  weight_scheme, class_groups, apply_classifier_preselection, extra_columns,
                  split_seed):
    """Wie load_dataset_with_sample_names im Original, aber ohne
    sample_name (fuer Contour-Plots nicht gebraucht) - liefert raw_data,
    class_weights, mass_{train,val,test}, classes, extra_arrays."""
    result = build_training_dataset(
        base_dir, data_config, fraction, seed, eras,
        train_fraction=train_fraction, val_fraction=val_fraction,
        class_groups=class_groups, verbose=True,
        apply_classifier_preselection=apply_classifier_preselection,
        extra_columns=extra_columns,
        split_seed=split_seed,
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
    }

    if weight_scheme == "weighted_abs":
        cw_train = result["class_weights_abs"]
    elif weight_scheme == "weighted_only_positive":
        cw_train = result["class_weights_only_positive"]
    else:
        raise NotImplementedError(f"weight_scheme '{weight_scheme}' ist nicht implementiert.")

    class_weights = {
        "cw_train": np.asarray(cw_train),
        "cw_val": np.asarray(result["class_weights_val"]),
        "cw_test": np.asarray(result["class_weights_test"]),
    }

    mass_train = np.asarray(result["mass_train"], dtype=np.float32)
    mass_val = np.asarray(result["mass_val"], dtype=np.float32)
    mass_test = np.asarray(result["mass_test"], dtype=np.float32)

    extra_arrays = {}
    for c in extra_columns:
        for split in ("train", "val", "test"):
            extra_arrays[f"{c}_{split}"] = np.asarray(result[f"{c}_{split}"])

    return raw_data, class_weights, mass_train, mass_val, mass_test, result["classes"], extra_arrays


def _evenly_spaced(items, k):
    items = list(items)
    if k is None or k <= 0 or len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def _dims_from_histograms(histograms):
    return sorted({(i, j) for (_, i, j) in histograms.keys()})


def display_name(cls_name):
    """Siehe CLASS_DISPLAY_NAMES oben."""
    return CLASS_DISPLAY_NAMES.get(cls_name, cls_name)


def axis_label(dim_idx, classes):
    """Siehe AXIS_LABEL_STYLE oben - waehlt zwischen ein paar Formulierungen."""
    cls_name = display_name(classes[dim_idx]) if dim_idx < len(classes) else None
    if cls_name is None:
        return f"Discriminant (dim. {dim_idx})"
    if AXIS_LABEL_STYLE == "discriminant_class":
        return f"{cls_name} discriminant (dim. {dim_idx})"
    if AXIS_LABEL_STYLE == "score_class":
        return f"{cls_name} classifier score (dim. {dim_idx})"
    return f"Discriminant (dim. {dim_idx})"  # "discriminant_generic" und Default


def plot_contours_from_histograms(histograms, classes, bins, score_range, output_path,
                                   quantiles=(0.6827, 0.9545), smooth_sigma=0.0, merge_bins=1,
                                   dim_pairs=None):
    """
    1:1 aus evaluate_model_detailed_contours.py uebernommen (plottet
    DIREKT aus einem in-memory Histogramm-Dict, kein CSV/Meta-Roundtrip),
    um zwei Dinge erweitert:
      - dim_pairs: optionale Liste von (i, j)-Paaren, auf die die Panels
        eingeschraenkt werden (z.B. fuer einen kompakteren 3-statt-6-
        Panel-Plot) - None = alle in histograms vorhandenen Paare.
      - Styling (CMS-Label, Achsenbeschriftung, Font-Groessen, quadratische
        Panels, Legende im oben-rechten Panel statt einer pro-Panel-
        Legende) an plot_history.py's Konventionen angeglichen.
    """
    edges = np.linspace(score_range[0], score_range[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pairs = dim_pairs if dim_pairs is not None else _dims_from_histograms(histograms)
    cmap = plt.get_cmap("tab10")

    nrows, ncols = _dd_grid_shape(len(pairs))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    # Legende landet im OBEN-RECHTEN Panel des Grids (erste Zeile ist
    # immer vollstaendig gefuellt, siehe _dd_grid_shape) - dort als
    # normale Achsen-Legende (loc="upper right"), nicht mehr als
    # figurweite Legende ausserhalb der Panels.
    legend_panel_idx = ncols - 1

    legend_handles, legend_labels = [], []
    seen_classes = set()
    for panel_idx, (i, j) in enumerate(pairs):
        ax = axes_flat[panel_idx]
        for c_idx, cls_name in enumerate(classes):
            key = (cls_name, i, j)
            if key not in histograms:
                continue
            color = cmap(c_idx % 10)
            H, _, _ = histograms[key]
            plot_centers = centers
            if merge_bins > 1:
                H, plot_centers = _dd_merge_bins(H, centers, merge_bins)
            if smooth_sigma > 0:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=smooth_sigma)
            if H.max() <= 0:
                continue
            pair_handles = styled_class_contours(ax, H, plot_centers, color, quantiles=quantiles)
            if cls_name not in seen_classes:
                seen_classes.add(cls_name)
                for handle, level_label in pair_handles:
                    legend_handles.append(handle)
                    legend_labels.append(f"{display_name(cls_name)} ({level_label} contour)")
        ax.set_xlabel(axis_label(i, classes), fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(axis_label(j, classes), fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlim(score_range)
        ax.set_ylim(score_range)
        ax.set_box_aspect(1)  # Panel immer quadratisch, unabhaengig vom Figure-Seitenverhaeltnis
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(alpha=0.3)

    for k in range(len(pairs), len(axes_flat)):
        axes_flat[k].axis("off")

    _draw_cms_label(axes_flat[0], CMS_MAIN_TEXT, CMS_RIGHT_LABEL, fontsize=CMS_LABEL_FONTSIZE)

    if legend_handles and legend_panel_idx < len(pairs):
        axes_flat[legend_panel_idx].legend(
            legend_handles, legend_labels, loc="upper right",
            fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.85,
        )

    fig.tight_layout(rect=[0, 0, 1, 1])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Gespeichert: {output_path}")
    return output_path



def find_classifier_checkpoints(out_dir, max_checkpoint_epochs):
    """
    Liste von (epoch_or_None, weights_path, label): entweder mehrere
    periodische finetune_checkpoints/-Epochen (epoch = echte Epochenzahl,
    label = "epochNNNN"), oder - falls keine vorhanden sind - ein
    einzelner Eintrag mit den finalen/besten Gewichten (epoch = None,
    label = "final").
    """
    finetune_ckpt_dir = os.path.join(out_dir, "finetune_checkpoints")
    ckpt_epoch_re = re.compile(r"classifier_finetune_epoch(\d+)\.weights\.h5$")
    ckpt_epochs = sorted(
        int(ckpt_epoch_re.search(os.path.basename(p)).group(1))
        for p in glob.glob(os.path.join(finetune_ckpt_dir, "classifier_finetune_epoch*.weights.h5"))
    )
    if ckpt_epochs:
        selected = _evenly_spaced(ckpt_epochs, max_checkpoint_epochs)
        return [
            (ep, os.path.join(finetune_ckpt_dir, f"classifier_finetune_epoch{ep:04d}.weights.h5"),
             f"epoch{ep:04d}")
            for ep in selected
        ]

    cls_path = os.path.join(out_dir, "classifier_finetuned.weights.h5")
    if not os.path.exists(cls_path):
        candidates = sorted(glob.glob(os.path.join(out_dir, "classifier_finetuned*.weights.h5")))
        if not candidates:
            raise FileNotFoundError(f"Keine classifier_finetuned*.weights.h5 unter {out_dir!r} gefunden.")
        cls_path = candidates[-1]
    return [(None, cls_path, "final")]



# =============================================================================
# Orchestrierung
# =============================================================================
def main():
    if not CHECKPOINT_DIRS:
        raise SystemExit("CHECKPOINT_DIRS ist leer - bitte oben im Skript mindestens ein "
                          "Trainings-out-dir eintragen.")

    setup_device(CUDA_DEVICE)

    # ---- Datenkonfiguration einmalig aus dem ERSTEN Verzeichnis lesen ----
    first_out_dir = os.path.abspath(CHECKPOINT_DIRS[0])
    info = parse_training_info(first_out_dir)
    training_config = info["training_config"]

    base_dir = info["base_dir"]
    if not base_dir:
        raise SystemExit(f"Konnte individual_samples/-base-dir nicht aus "
                          f"{info['info_path']!r} bestimmen.")

    data_prep_cfg = training_config.get("data_prep", {}) or {}
    data_config_path = info["data_config_path"] or data_prep_cfg.get("data_config_path")
    if not data_config_path:
        raise SystemExit("Konnte data_config_path nicht bestimmen.")

    fraction = float(data_prep_cfg["fraction"])
    seeds_cfg = training_config.get("seeds")
    if seeds_cfg:
        seed = seeds_cfg["data"]
        split_seed = seeds_cfg.get("data_split", seed)
    else:
        seed = training_config["random_seed"]
        split_seed = seed
    train_fraction = float(data_prep_cfg.get("train_fraction", 0.7))
    val_fraction = float(data_prep_cfg.get("val_fraction", 0.15))
    eras_cfg = data_prep_cfg.get("eras", None)
    eras = eras_cfg if eras_cfg is not None else DEFAULT_ERAS
    use_class_groups = bool(data_prep_cfg.get("use_class_groups", True))
    weight_scheme = training_config["class_weight_scheme"]

    data_config = load_data_config(data_config_path)

    # ---- Preselection-Konfiguration (Stage A ueber build_training_dataset,
    # Stage B statische Cuts + score-Cut wie im Original) ----
    presel_cfg = training_config.get("preselection", {}) or {}
    presel_kwargs = classifier_preselection_kwargs(data_prep_cfg)
    print(f"[INFO] Classifier (Stage A) Preselection: "
          f"{'AN' if presel_kwargs['apply_classifier_preselection'] else 'aus'}")
    extra_columns = tuple(dict.fromkeys(
        presel_kwargs["extra_columns"] + static_preselection_columns(presel_cfg)
    ))

    print(f"[INFO] Lade Datensatz EINMAL aus '{base_dir}' "
          f"(Config: {info['info_path']}, eras={eras}, fraction={fraction}, "
          f"seed={seed}, split_seed={split_seed}) ...")
    (raw_data, class_weights, mass_train, mass_val, mass_test,
     classes, extra_arrays) = load_dataset(
        base_dir, data_config, fraction, seed, eras,
        train_fraction=train_fraction, val_fraction=val_fraction,
        weight_scheme=weight_scheme,
        class_groups=("auto" if use_class_groups else None),
        apply_classifier_preselection=presel_kwargs["apply_classifier_preselection"],
        extra_columns=extra_columns,
        split_seed=split_seed,
    )
    mass_by_split = {"train": mass_train, "val": mass_val, "test": mass_test}
    print(f"[INFO] Klassen (intern): {classes}")

    # ---- STEP 2a: statische Preselection-Cuts (mass/dijet-mass/mvaID/is_boosted) ----
    for split in ("train", "val", "test"):
        arrays = {"mass": mass_by_split[split]}
        for c in extra_columns:
            arrays[EXTRA_COLUMN_LOGICAL_NAME[c]] = extra_arrays[f"{c}_{split}"]
        cuts = build_static_preselection_cuts(presel_cfg, arrays)
        if not cuts:
            continue
        keep_idx, _ = compute_preselection_cutflow(cuts, n_rows=mass_by_split[split].shape[0], split_name=split)
        if keep_idx.size == 0:
            raise ValueError(f"Statische Preselection entfernt ALLE '{split}'-Events.")
        for key in (f"X_{split}", f"y_{split}", f"rel_w_{split}"):
            raw_data[key] = np.asarray(raw_data[key])[keep_idx]
        class_weights[f"cw_{split}"] = np.asarray(class_weights[f"cw_{split}"])[keep_idx]
        mass_by_split[split] = mass_by_split[split][keep_idx]
        for c in extra_columns:
            extra_arrays[f"{c}_{split}"] = extra_arrays[f"{c}_{split}"][keep_idx]

    print(f"[INFO] Nach statischer Preselection: "
          f"train={raw_data['X_train'].shape[0]}, val={raw_data['X_val'].shape[0]}, "
          f"test={raw_data['X_test'].shape[0]} Events.")

    input_size = raw_data["X_train"].shape[1]
    output_size = len(classes)
    score_cut_cfg = presel_cfg.get("score_cut", {}) or {}
    score_cut_enabled = bool(score_cut_cfg.get("enabled", False))
    score_chunk_size = training_config.get("training", {}).get("chunk_size")

    disc_cfg = training_config.get("discriminant_density_diagnostics", {}) or {}
    chunk_size = int(disc_cfg.get("chunk_size") or DISC_DENSITY_DEFAULT_CHUNK_SIZE)

    # ---- pro Checkpoint-Verzeichnis: Contour-Plots erzeugen ----
    for ckpt_out_dir in CHECKPOINT_DIRS:
        ckpt_out_dir = os.path.abspath(ckpt_out_dir)
        print(f"\n[INFO] === {ckpt_out_dir} ===")
        ckpt_info = parse_training_info(ckpt_out_dir)
        ckpt_training_config = ckpt_info["training_config"]

        if ckpt_info["base_dir"] and ckpt_info["base_dir"] != base_dir:
            print(f"[WARNING] base_dir in {ckpt_out_dir!r} ({ckpt_info['base_dir']!r}) weicht vom "
                  f"einmal geladenen Datensatz ({base_dir!r}) ab - die Contour-Plots basieren "
                  f"trotzdem auf den global geladenen Daten!")
        if ckpt_info["data_config_path"] and ckpt_info["data_config_path"] != data_config_path:
            print(f"[WARNING] data_config_path in {ckpt_out_dir!r} weicht vom global geladenen "
                  f"ab ({ckpt_info['data_config_path']!r} != {data_config_path!r}).")

        classifier = build_classifier(ckpt_training_config, input_size, output_size)

        # ---- STEP 2b: score-basierter Cut (haengt vom jeweils finalen
        # Klassifikator dieses Runs ab - deshalb pro Verzeichnis auf einer
        # KOPIE der global geladenen/statisch preselektierten Arrays) ----
        X = {s: raw_data[f"X_{s}"] for s in ("train", "val", "test")}
        Y = {s: raw_data[f"y_{s}"] for s in ("train", "val", "test")}
        W = {s: raw_data[f"rel_w_{s}"] for s in ("train", "val", "test")}

        if score_cut_enabled:
            final_cls_path = os.path.join(ckpt_out_dir, "classifier_finetuned.weights.h5")
            if not os.path.exists(final_cls_path):
                candidates = sorted(glob.glob(os.path.join(ckpt_out_dir, "classifier_finetuned*.weights.h5")))
                final_cls_path = candidates[-1] if candidates else None
            if final_cls_path is None:
                print(f"[WARNING] Kein classifier_finetuned*.weights.h5 in {ckpt_out_dir!r} - "
                      f"ueberspringe score-basierte Preselection fuer dieses Verzeichnis.")
            else:
                classifier.load_weights(final_cls_path)
                for split in ("train", "val", "test"):
                    class_scores = compute_class_scores(
                        classifier, X[split], int(score_cut_cfg.get("class_index", 0)),
                        chunk_size=score_chunk_size,
                    )
                    score_cut = build_score_preselection_cut(presel_cfg, class_scores)
                    keep_idx, _ = compute_preselection_cutflow(
                        [score_cut], n_rows=class_scores.shape[0], split_name=split,
                    )
                    if keep_idx.size == 0:
                        raise ValueError(f"Score-basierte Preselection entfernt ALLE '{split}'-"
                                          f"Events in {ckpt_out_dir!r}.")
                    X[split] = X[split][keep_idx]
                    Y[split] = Y[split][keep_idx]
                    W[split] = W[split][keep_idx]
                print(f"[INFO] Score-basierte Preselection (finaler Classifier): "
                      f"train={X['train'].shape[0]}, val={X['val'].shape[0]}, test={X['test'].shape[0]}")

        plot_out_dir = os.path.join(ckpt_out_dir, CONTOUR_SUBDIR_NAME)
        os.makedirs(plot_out_dir, exist_ok=True)

        checkpoints = find_classifier_checkpoints(ckpt_out_dir, MAX_CHECKPOINT_EPOCHS)
        print(f"[INFO] {len(checkpoints)} Checkpoint(s) fuer Contour-Plots: "
              f"{[label for _, _, label in checkpoints]}")

        wrapper = SimpleNamespace(classifier=classifier)
        for epoch, weights_path, label in checkpoints:
            classifier.load_weights(weights_path)
            for split in ("train", "val", "test"):
                hists = compute_class_dim_histograms(
                    wrapper, X[split], Y[split], W[split], classes,
                    bins=CONTOUR_BINS, score_range=CONTOUR_SCORE_RANGE, chunk_size=chunk_size,
                )
                all_pairs = _dims_from_histograms(hists)

                out_path = os.path.join(
                    plot_out_dir, f"discriminant_density_{split}_{label}.{PLOT_DTYPE}"
                )
                plot_contours_from_histograms(
                    hists, classes, CONTOUR_BINS, CONTOUR_SCORE_RANGE, out_path,
                    smooth_sigma=CONTOUR_SMOOTH_SIGMA, merge_bins=CONTOUR_MERGE_BINS,
                )

                if CONTOUR_DIM_PAIRS_COMPACT:
                    compact_pairs = [p for p in CONTOUR_DIM_PAIRS_COMPACT if p in all_pairs]
                    missing = [p for p in CONTOUR_DIM_PAIRS_COMPACT if p not in all_pairs]
                    if missing:
                        print(f"[WARNING] CONTOUR_DIM_PAIRS_COMPACT enthaelt Paare, die es fuer "
                              f"diesen Checkpoint nicht gibt (nur {len(classes)} Klassen/Dims "
                              f"vorhanden): {missing} - werden fuer den kompakten Plot ignoriert.")
                    if compact_pairs:
                        compact_out_path = os.path.join(
                            plot_out_dir,
                            f"discriminant_density_{split}_{label}{COMPACT_SUFFIX}.{PLOT_DTYPE}",
                        )
                        plot_contours_from_histograms(
                            hists, classes, CONTOUR_BINS, CONTOUR_SCORE_RANGE, compact_out_path,
                            smooth_sigma=CONTOUR_SMOOTH_SIGMA, merge_bins=CONTOUR_MERGE_BINS,
                            dim_pairs=compact_pairs,
                        )
        print(f"[INFO] Contour-Plots gespeichert unter: {plot_out_dir}")

    print("\n[INFO] Fertig.")


if __name__ == "__main__":
    main()