#!/usr/bin/env python3
"""
run_experiments.py

Select one, several, or all presets from presets.PRESETS and run them via
run.main() -- in-process, no subprocess involved.

CUSTOM-DATA MODE
----------------
By default, every preset runs exactly as originally defined: against the
Quran gold sheet, scored with whatever evaluation_mode --analyzer
auto-selects.

Pass --custom-gold-path to flip ALL selected presets over to a different
test set instead, without touching anything else:
    - gold_path is overridden to point at your file
    - data_source is forced to "none" (loaded as-is -- no tanzil merge,
      since your file already has its own 'word' column and the same
      sentence_index/word_index schema)
    - data_dir / models_dir / tanzil_path / quran_script are left
      completely untouched, because they still drive the Quran/maqam
      train+tune data used to build the seeded custom-analyses lookup
      table -- that lookup is independent of which gold set you're
      scoring against.
    - the derived output dir gets a "_custom" suffix so custom-mode runs
      never silently overwrite your normal Quran-mode outputs.

Every other --custom-* flag is an independent override applied on top of
whichever preset(s) are selected -- see --help for the full list.

USAGE
-----
List available presets:
    python run_experiments.py --list

Run one preset:
    python run_experiments.py r13_imlaay

Run several presets (space separated):
    python run_experiments.py s31_imlaay camel_morph_uthmani

Run everything:
    python run_experiments.py --all

Interactive picker (no args at all):
    python run_experiments.py

Dry run (print configs without executing them):
    python run_experiments.py --all --dry-run

Auto-disambiguate output dirs when multiple selected presets collide:
    python run_experiments.py --all --suffix

Stop on first failure (default) vs. keep going:
    python run_experiments.py --all --keep-going

Run every quran preset against a different test set instead:
    python run_experiments.py --all --custom-gold-path /path/to/test.csv
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import asdict, replace

from presets import PRESETS
from run import RunConfig, derive_output_dir
import run as run_module


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

def resolved_output_dir(config: RunConfig, name: str, *, is_custom: bool, suffix: bool) -> str:
    output_dir = config.output_dir or derive_output_dir(config)
    if is_custom:
        output_dir = f"{output_dir.rstrip('/')}_custom"
    if suffix:
        output_dir = f"{output_dir.rstrip('/')}_{name}"
    return output_dir


def build_run_config(name: str, base: RunConfig, *, suffix=False,
                      custom_gold_path=None, custom_evaluation_mode=None,
                      custom_quran_script=None, custom_word_script=None,
                      custom_split=None, custom_back_off=None,
                      custom_custom_analyses=None, custom_gold_apply_bw2ar=False,
                      custom_skip_eval=False, custom_model=None) -> RunConfig:
    """Apply every --custom-* override on top of one preset's RunConfig.
    Never mutates `base` -- dataclasses.replace always returns a new object,
    so re-running in normal mode later is unaffected."""
    overrides = {}
    is_custom = custom_gold_path is not None
    if is_custom:
        overrides["gold_path"] = custom_gold_path
        overrides["data_source"] = "none"
    if custom_evaluation_mode is not None:
        overrides["evaluation_mode"] = custom_evaluation_mode
    if custom_model is not None:
        overrides["model"] = custom_model
    if custom_quran_script is not None:
        overrides["quran_script"] = custom_quran_script
    if custom_word_script is not None:
        overrides["word_script"] = custom_word_script
    if custom_split is not None:
        overrides["split"] = custom_split
    if custom_back_off is not None:
        overrides["back_off"] = custom_back_off
    if custom_custom_analyses == "off":
        overrides["no_custom_analyses"] = True
    elif custom_custom_analyses == "on":
        overrides["no_custom_analyses"] = False
    if custom_gold_apply_bw2ar:
        overrides["gold_apply_bw2ar"] = True
    if custom_skip_eval:
        overrides["skip_eval"] = True

    config = replace(base, **overrides) if overrides else base
    output_dir = resolved_output_dir(config, name, is_custom=is_custom, suffix=suffix)
    return replace(config, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def find_output_dir_collisions(selected):
    by_dir = defaultdict(list)
    for name in selected:
        config = PRESETS[name]
        output_dir = config.output_dir or derive_output_dir(config)
        by_dir[output_dir].append(name)
    return {d: names for d, names in by_dir.items() if len(names) > 1}


def interactive_pick():
    names = list(PRESETS.keys())
    print("Available presets:")
    for i, name in enumerate(names, 1):
        config = PRESETS[name]
        output_dir = config.output_dir or derive_output_dir(config)
        print(f"  {i}. {name}  (output-dir: {output_dir})")

    raw = input(
        "\nEnter numbers or names to run (comma/space separated, 'all' for everything): "
    ).strip()
    if not raw:
        return []
    if raw.lower() == "all":
        return names

    tokens = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    selected = []
    for tok in tokens:
        if tok.isdigit() and 1 <= int(tok) <= len(names):
            selected.append(names[int(tok) - 1])
        elif tok in PRESETS:
            selected.append(tok)
        else:
            print(f"  ! Unrecognized preset: {tok} (skipping)")
    return selected


def print_config(name: str, config: RunConfig):
    print(f"\n=== {name} ===")
    for field, value in asdict(config).items():
        print(f"  {field}: {value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run one or more predefined run.py experiment configurations."
    )
    parser.add_argument("presets", nargs="*", help="Names of presets to run (see --list).")
    parser.add_argument("--all", action="store_true", help="Run every preset.")
    parser.add_argument("--list", action="store_true", help="List presets and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without executing.")
    parser.add_argument("--suffix", action="store_true",
                         help="Append preset name to each output_dir to avoid collisions.")
    parser.add_argument("--keep-going", action="store_true",
                         help="Continue running remaining presets even if one fails.")
    parser.add_argument("--custom-gold-path", default=None,
                         help="Evaluate all selected presets against this test CSV instead "
                              "of the Quran gold sheet. Implies data_source='none' (loaded "
                              "as-is, no tanzil merge). data_dir/models_dir/tanzil_path and "
                              "each preset's quran_script/split are left untouched.")
    parser.add_argument("--custom-evaluation-mode", default=None, choices=["camel_morph", "calima"],
                         help="Override evaluation_mode for all selected presets.")
    parser.add_argument("--custom-quran-script", default=None, choices=["imlaay", "uthmani"],
                         help="Override quran_script for all selected presets.")
    parser.add_argument("--custom-word-script", default=None, choices=["imlaay", "uthmani"],
                         help="Override word_script for all selected presets.")
    parser.add_argument("--custom-split", default=None,
                         choices=["train", "train_tune", "all", "all_tune", "none"],
                         help="Override split for all selected presets.")
    parser.add_argument("--custom-back-off", default=None,
                         help="Override back_off for all selected presets.")
    parser.add_argument("--custom-custom-analyses", default=None, choices=["on", "off"],
                         help="Force custom_analyses on/off for all selected presets.")
    parser.add_argument("--custom-gold-apply-bw2ar", action="store_true",
                         help="Convert gold_df's diac/lex columns from Buckwalter to Arabic "
                              "script before scoring.")
    parser.add_argument("--custom-skip-eval", action="store_true",
                         help="Run disambiguation only, no scoring.")
    parser.add_argument("--custom-model", default=None,
                         help="Override model for all selected presets.")

    args = parser.parse_args()

    if args.list:
        print("Available presets:\n")
        for name, config in PRESETS.items():
            output_dir = config.output_dir or derive_output_dir(config)
            print(f"- {name}")
            print(f"    model: {config.model}")
            print(f"    analyzer: {config.analyzer}")
            print(f"    quran_script: {config.quran_script}")
            print(f"    word_script: {config.word_script}")
            print(f"    source_mode: {config.source_mode}")
            print(f"    output_dir: {output_dir}")
        return

    if args.all:
        selected = list(PRESETS.keys())
    elif args.presets:
        selected = []
        for name in args.presets:
            if name not in PRESETS:
                print(f"Unknown preset: {name}")
                print(f"Available: {', '.join(PRESETS.keys())}")
                sys.exit(1)
            selected.append(name)
    else:
        selected = interactive_pick()

    if not selected:
        print("Nothing selected. Exiting.")
        return

    collisions = find_output_dir_collisions(selected)
    if collisions and not args.suffix:
        print("WARNING: the following selected presets share the same output_dir")
        print("and will overwrite each other's results. Re-run with --suffix to")
        print("automatically disambiguate them, or pick a different set.\n")
        for d, names in collisions.items():
            print(f"  {d}: {', '.join(names)}")
        print()
        proceed = input("Continue anyway? [y/N]: ").strip().lower()
        if proceed != "y":
            print("Aborted.")
            return

    for name in selected:
        config = build_run_config(
            name, PRESETS[name], suffix=args.suffix,
            custom_gold_path=args.custom_gold_path,
            custom_evaluation_mode=args.custom_evaluation_mode,
            custom_quran_script=args.custom_quran_script,
            custom_word_script=args.custom_word_script,
            custom_split=args.custom_split,
            custom_back_off=args.custom_back_off,
            custom_custom_analyses=args.custom_custom_analyses,
            custom_gold_apply_bw2ar=args.custom_gold_apply_bw2ar,
            custom_skip_eval=args.custom_skip_eval,
            custom_model=args.custom_model,
        )
        print_config(name, config)

        if args.dry_run:
            continue

        try:
            run_module.main(config)
        except Exception as exc:
            print(f"!!! Preset '{name}' failed: {exc}")
            if not args.keep_going:
                sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
