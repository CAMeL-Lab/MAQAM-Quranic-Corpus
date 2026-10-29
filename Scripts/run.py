#!/usr/bin/env python3
"""
run.py
======

The one CLI for this project. Defines RunConfig -- a plain dataclass that
fully describes one experiment run -- and main(config), the pipeline that
executes it (analyzer/model setup, tanzil word attachment, quran_df
profile building, disambiguation, evaluation).

run_experiments.py imports RunConfig and main() directly and calls them
in-process (no subprocess) -- this file's __main__ block is just a
convenience for running a single one-off config from the command line.
"""

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# The repository vendors CAMeL Tools under camel_tools/camel_tools rather
# than installing it into the active interpreter.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMEL_TOOLS_ROOT = PROJECT_ROOT / "camel_tools"
if str(CAMEL_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMEL_TOOLS_ROOT))

import pandas as pd

import config as cfg
import data
import models
import evaluate


# ──────────────────────────────────────────────────────────────────────────
# RunConfig -- everything needed to run and score one experiment.
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    data_dir: str
    models_dir: str
    gold_path: str
    model: str
    analyzer: str  # "calima_r13" | "calima_s31" | "camel_morph"

    tanzil_path: Optional[str] = None
    quran_script: str = "uthmani"       # "uthmani" | "imlaay"
    word_script: str = "uthmani"        # "uthmani" | "imlaay"
    split: str = "train_tune"           # "train" | "train_tune" | "all" | "all_tune" | "none"
    source_mode: str = "none"           # "none" | "quran_only" | "quran_prefer" | "non_quran_prefer"
    batch_size: int = 1
    top: int = 5000
    back_off: str = "ADD_PROP"
    no_custom_analyses: bool = False
    data_source: str = "quran"          # "quran" | "patb" | "none"
    evaluation_mode: Optional[str] = None  # "camel_morph" | "calima"; defaults per --analyzer
    gold_apply_bw2ar: bool = False
    skip_eval: bool = False
    output_dir: Optional[str] = None    # derived automatically if left None


def derive_output_dir(config: RunConfig) -> str:
    """outputs/{model}_{analyzer}_{word_script} -- used whenever a preset
    doesn't set output_dir explicitly."""
    return f"outputs/{config.model}_{config.analyzer}_{config.word_script}"


def resolve_model_key(name: str) -> str:
    return cfg.MODEL_NAME_MAP.get(name, name)


def resolve_analyzer(analyzer_name: str):
    """Returns (model_override_or_None, db_path, default_eval_mode).

    All three analyzers now build a normal Analyzer from a db_path (r13 no
    longer uses camel_tools' bundled pretrained model/analyzer)."""
    if analyzer_name == "calima_r13":
        return None, cfg.R13_DB_PATH, "calima"
    if analyzer_name == "calima_s31":
        return None, cfg.S31_DB_PATH, "calima"
    if analyzer_name == "camel_morph":
        return None, cfg.CM_DB_PATH, "camel_morph"
    raise ValueError(
        f"Unknown analyzer {analyzer_name!r}. Choose from calima_r13, calima_s31, camel_morph."
    )


# ──────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────

def main(config: RunConfig):
    output_dir = config.output_dir or derive_output_dir(config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_override, db_path, default_eval_mode = resolve_analyzer(config.analyzer)
    model_key = model_override or resolve_model_key(config.model)
    evaluation_mode = config.evaluation_mode or default_eval_mode

    print(f"[1/6] Analyzer setup (analyzer={config.analyzer})")
    if not db_path:
        raise ValueError(
            f"No db_path resolved for analyzer={config.analyzer!r} -- "
            f"check the corresponding *_DB_PATH in config.py."
        )
    analyzer = models.build_analyzer(db_path, config.back_off)

    print(f"[2/6] Loading model: {model_key}")
    disambig = models.build_disambiguator(
        model_key, analyzer, top=config.top, models_dir=config.models_dir
    )

    print(f"[3/6] Loading gold_df from: {config.gold_path} (data-source={config.data_source})")
    if config.data_source == "quran":
        # ONLY this exact value triggers the tanzil merge. Everything else
        # (patb, none, or any future/typo'd value) safely falls through to
        # the "gold already has its own word column" path below.
        if not config.tanzil_path:
            raise ValueError("tanzil_path is required when data_source='quran'")
        gold_df = pd.read_csv(config.gold_path)
        print(f"      Attaching gold_df['word'] from tanzil CSV "
              f"(script={config.word_script}, surahs={cfg.TANZIL_SURAHS})")
        words = data.build_words_from_tanzil(
            config.tanzil_path, script_type=config.word_script, surahs=cfg.TANZIL_SURAHS
        )
        gold_df = data.attach_tanzil_words(gold_df, words)
        print(f"      Attached {len(words)} words. Sample: {list(words.head(5))}")
    else:
        gold_df = pd.read_csv(config.gold_path)
        required = {"word", "word_index"}
        missing = required - set(gold_df.columns)
        if missing:
            raise ValueError(
                f"gold_path is missing required columns: {sorted(missing)} "
                f"(data_source={config.data_source!r} expects word/word_index already present, "
                f"no tanzil merge is performed)."
            )

    if config.gold_apply_bw2ar:
        from camel_tools.utils.charmap import CharMapper
        bw2ar = CharMapper.builtin_mapper('bw2ar')
        for col in ("diac", "lex"):
            if col in gold_df.columns:
                gold_df[col] = gold_df[col].apply(lambda x: bw2ar(str(x)) if pd.notna(x) else x)
        print("      Applied bw2ar conversion to gold_df['diac']/['lex'].")

    use_custom_analyses = not config.no_custom_analyses
    source_mode = config.source_mode
    if not use_custom_analyses:
        print("[4/6] no_custom_analyses set -- skipping quran_df / custom_analyses entirely.")
        if source_mode != "none":
            print(f"      (source_mode={source_mode!r} ignored; forcing 'none')")
        source_mode = "none"
    else:
        quran_csvs = data.resolve_quran_csvs([], config.data_dir, config.quran_script, config.split)
        if quran_csvs:
            print(f"[4/6] Building quran_df from {len(quran_csvs)} CSV(s): {quran_csvs}")
            quran_df, custom_analyses = data.build_profile(quran_csvs)
            disambig._custom_analyses = custom_analyses
            print(f"      quran_df has {len(quran_df)} rows; gold_df has {len(gold_df)} rows (kept separate).")
        else:
            print(f"[4/6] No quran_df CSVs resolved for split={config.split!r} -- "
                  f"continuing without custom_analyses.")
            if source_mode != "none":
                print(f"      (source_mode={source_mode!r} ignored; forcing 'none')")
            source_mode = "none"

    print("[5/6] Running disambiguation...")
    disambig_df = models.run_disambiguation_from_df(
        gold_df, disambig, batch_size=config.batch_size, source_mode=source_mode,
    )
    disambig_df = models.apply_clitic_overrides(disambig_df, cfg.DEFAULT_CLITIC_OVERRIDES)

    out_csv = out_dir / "disambig_output.csv"
    disambig_df.to_csv(out_csv, index=False)
    print(f"      Saved disambiguation output -> {out_csv}")

    if config.skip_eval:
        print("[6/6] skip_eval set -- no gold to score against, stopping after disambiguation.")
        return None

    print(f"[6/6] Evaluating (evaluation-mode={evaluation_mode})...")
    comparison_df, summary_df = evaluate.evaluate(disambig_df, gold_df, analyzer_type=evaluation_mode)

    print("\nPer-feature accuracy:")
    accuracy_format = {"acc (%)": "{:.1f}".format}
    print(comparison_df.to_string(index=False, formatters=accuracy_format))
    print("\nSummary:")
    print(summary_df.to_string(index=False, formatters=accuracy_format))

    comparison_df.to_csv(out_dir / "comparison.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved comparison -> {out_dir / 'comparison.csv'}")
    print(f"Saved summary -> {out_dir / 'summary.csv'}")

    meta = {
        "model": model_key,
        "analyzer": config.analyzer,
        "evaluation_mode": evaluation_mode,
        "script": config.word_script,
        "source_mode": source_mode,
    }
    evaluate.append_to_combined_summary(cfg.COMBINED_SUMMARY_CSV, out_dir.name, summary_df, meta=meta)
    print(f"Appended to combined summary -> {cfg.COMBINED_SUMMARY_CSV}")

    return comparison_df, summary_df


# ──────────────────────────────────────────────────────────────────────────
# Standalone CLI (for running one-off configs without going through
# run_experiments.py / presets.py).
# ──────────────────────────────────────────────────────────────────────────

def _parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--tanzil-path", default=None)
    p.add_argument("--gold-path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--analyzer", required=True,
                   choices=["calima_r13", "calima_s31", "camel_morph"])
    p.add_argument("--quran-script", default="uthmani", choices=["uthmani", "imlaay"])
    p.add_argument("--word-script", default="uthmani", choices=["uthmani", "imlaay"])
    p.add_argument("--split", default="train_tune",
                   choices=["train", "train_tune", "all", "all_tune", "none"])
    p.add_argument("--source-mode", default="none",
                   choices=["none", "quran_only", "quran_prefer", "non_quran_prefer"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--top", type=int, default=5000)
    p.add_argument("--back-off", default="ADD_PROP")
    p.add_argument("--no-custom-analyses", action="store_true")
    p.add_argument("--data-source", default="quran", choices=["quran", "patb", "none"])
    p.add_argument("--evaluation-mode", default=None, choices=["camel_morph", "calima"])
    p.add_argument("--gold-apply-bw2ar", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    return RunConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})


if __name__ == "__main__":
    sys.exit(main(_parse_args()))
