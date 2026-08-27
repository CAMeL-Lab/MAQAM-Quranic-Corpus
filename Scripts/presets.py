"""
presets.py
==========

Your recurring experiment configurations, as a plain dict:
    PRESETS = {"r13_imlaay": RunConfig(...), "r13_uthmani": RunConfig(...), ...}

This file only defines data -- selecting, overriding, and running presets
lives in run_experiments.py. To add a config that only makes sense with
one script instead of both, just add it directly as a single dict entry
(see the bottom of this file for examples).
"""

from run import RunConfig
import config as cfg

SCRIPTS = ("imlaay", "uthmani")

# Fields shared by every "normal Quran gold" preset below -- edit once here
# instead of in every preset.
_BASE = dict(
    data_dir=cfg.DATA_DIR,
    models_dir=cfg.MODELS_DIR,
    tanzil_path=cfg.TANZIL_PATH,
    gold_path=cfg.GOLD_PATH,
)


def _both_scripts(base_name, **fields):
    """
    Build two preset entries from one base config: "{base_name}_imlaay" and
    "{base_name}_uthmani", identical except quran_script/word_script.

    `fields` must not already contain quran_script/word_script.
    """
    if "quran_script" in fields or "word_script" in fields:
        raise ValueError(
            f"_both_scripts({base_name!r}, ...) received an explicit "
            f"quran_script/word_script -- remove it, these are set "
            f"automatically for each of the two expanded presets."
        )
    return {
        f"{base_name}_{script}": RunConfig(quran_script=script, word_script=script, **fields)
        for script in SCRIPTS
    }


PRESETS: dict[str, RunConfig] = {}

PRESETS.update(_both_scripts(
    "r13", **_BASE,
    model="patb", analyzer="calima_r13", split="train_tune",
    source_mode="none", batch_size=1, top=5000, no_custom_analyses=True,
))

PRESETS.update(_both_scripts(
    "s31", **_BASE,
    model="patb", analyzer="calima_s31", split="train_tune",
    source_mode="none", batch_size=1, top=5000, no_custom_analyses=True,
))

PRESETS.update(_both_scripts(
    "camel_morph", **_BASE,
    model="patb", analyzer="camel_morph", split="train_tune",
    source_mode="none", batch_size=1, top=5000, no_custom_analyses=True,
))

PRESETS.update(_both_scripts(
    "qtrain_model_and_cm", **_BASE,
    model="QTrain_imlaay_uthmani", analyzer="camel_morph", split="train_tune",
    source_mode="none", batch_size=1, top=5000, no_custom_analyses=True,
))

PRESETS.update(_both_scripts(
    "qtrain_model_and_qtrain", **_BASE,
    model="QTrain_imlaay_uthmani", analyzer="camel_morph", split="train_tune",
    source_mode="quran_only", batch_size=1, top=5000, back_off="ADD_PROP",
))

PRESETS.update(_both_scripts(
    "qtrain_model_and_cm_qtrain", **_BASE,
    model="QTrain_imlaay_uthmani", analyzer="camel_morph", split="train_tune",
    source_mode="quran_prefer", batch_size=1, top=5000, back_off="ADD_PROP",
))

PRESETS.update(_both_scripts(
    "qtrain_patb_model_and_cm", **_BASE,
    model="QTrain_imlaay_uthmani_patb", analyzer="camel_morph", split="train_tune",
    source_mode="none", batch_size=1, top=5000, no_custom_analyses=True,
))

PRESETS.update(_both_scripts(
    "qtrain_patb_model_and_qtrain", **_BASE,
    model="QTrain_imlaay_uthmani_patb", analyzer="camel_morph", split="train_tune",
    source_mode="quran_only", batch_size=1, top=5000, back_off="ADD_PROP",
))

PRESETS.update(_both_scripts(
    "qtrain_patb_model_and_cm_qtrain", **_BASE,
    model="QTrain_imlaay_uthmani_patb", analyzer="camel_morph", split="train_tune",
    source_mode="quran_prefer", batch_size=1, top=5000, back_off="ADD_PROP",
))

PRESETS.update(_both_scripts(
    "qall_patb_model_and_cm_qall", **_BASE,
    model="QAll_imlaay_uthmani_patb", analyzer="camel_morph", split="all_tune",
    source_mode="quran_prefer", batch_size=1, top=5000, back_off="ADD_PROP",
))

PRESETS.update(_both_scripts(
    "patb_model_with_cm_quran_analyses", **_BASE,
    model="patb", analyzer="camel_morph", split="train_tune",
    source_mode="none", batch_size=1, top=5000, back_off="ADD_PROP",
))

# ---------------------------------------------------------------------------
# Single-script presets (only ever run against one script, not both) --
# added directly rather than through _both_scripts(). Each sets its own
# output_dir explicitly since it doesn't follow the derived-name pattern.
# ---------------------------------------------------------------------------

PRESETS["patb_model_and_cm_quran_only_imlaay"] = RunConfig(
    **_BASE,
    model="patb", analyzer="camel_morph",
    quran_script="imlaay", word_script="imlaay",
    split="train_tune", source_mode="quran_only",
    batch_size=1, top=5000, back_off="ADD_PROP",
    output_dir="outputs/patb_camel_morph_imlaay_quran_only",
)

PRESETS["patb_model_and_cm_quran_prefer_imlaay"] = RunConfig(
    **_BASE,
    model="patb", analyzer="camel_morph",
    quran_script="imlaay", word_script="imlaay",
    split="train_tune", source_mode="non_quran_prefer",
    batch_size=1, top=5000, back_off="ADD_PROP",
    output_dir="outputs/patb_camel_morph_imlaay_quran_prefer",
)

PRESETS["qtrain_patb_imlaay_model_and_cm_qtrain_imlaay_imlaay"] = RunConfig(
    **_BASE,
    model="QTrain_patb_imlaay", analyzer="camel_morph",
    quran_script="imlaay", word_script="imlaay",
    split="train_tune", source_mode="quran_prefer",
    batch_size=1, top=5000, back_off="ADD_PROP",
    output_dir="outputs/QTrain_patb_imlaay_camel_morph_imlaay",
)

PRESETS["qtrain_patb_uthmani_model_and_cm_qtrain_imlaay_uthmani"] = RunConfig(
    **_BASE,
    model="QTrain_patb_uthmani", analyzer="camel_morph",
    quran_script="uthmani", word_script="uthmani",
    split="train_tune", source_mode="quran_prefer",
    batch_size=1, top=5000, back_off="ADD_PROP",
    output_dir="outputs/QTrain_patb_imlaay_camel_morph_uthmani",
)
