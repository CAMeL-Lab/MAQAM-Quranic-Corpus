"""
config.py
=========

Every machine-specific path and reusable default knob, in one place.
Edit this file when you move to a new machine or your data moves --
nothing else in the project should hardcode a path.
"""

# ──────────────────────────────────────────────────────────────────────────
# Morphology DBs -- one per --analyzer choice.
# ──────────────────────────────────────────────────────────────────────────
CM_DB_PATH = "../dbs/camel_morph_msa_v1.2.1.db"
S31_DB_PATH = ("../dbs/calima-msa-s31_0.4.2.utf8.db")
R13_DB_PATH = "../dbs/calima-msa-r13.db"

# ──────────────────────────────────────────────────────────────────────────
# Shared data/model locations used by presets.
# ──────────────────────────────────────────────────────────────────────────
TANZIL_PATH = "../data/tanzil_words.csv"
GOLD_PATH = ("../data/MAQAM_Gold_data.csv")
DATA_DIR = "../data"
MODELS_DIR = "../finetuned_models"

# Surah filter applied to the tanzil word source whenever --data-source=quran.
TANZIL_SURAHS = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]

# Maps the short --model names used in presets.py to the actual keys in
# core.models.MODEL_CHOICES.
MODEL_NAME_MAP = {
    "patb": "patb_model",
    "QTrain_imlaay_uthmani": "QTrain_imlaay_uthmani_model",
    "QTrain_imlaay_uthmani_patb": "QTrain_imlaay_uthmani_patb_model",
    "QAll_imlaay_uthmani_patb": "QAll_imlaay_uthmani_patb_model",
    "QTrain_patb_imlaay": "QTrain_patb_imlaay_model",
    "QTrain_patb_uthmani": "QTrain_patb_uthmani_model",
}

# Post-hoc clitic override rules: list of (word_list, feature, value).
DEFAULT_CLITIC_OVERRIDES = [
    (["الله", "بالله", "ولله", "فلله", "لله", "والعزى", "والله"], "prc0", "0"),
    (["الذين"], "prc2", "0"),
]

# Shared cross-preset accuracy table that every run.py invocation appends
# a row to (see core.evaluate.append_to_combined_summary).
COMBINED_SUMMARY_CSV = "../outputs/combined_summary.csv"
