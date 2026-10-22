# MAQAM: A Richly Annotated Corpus for Quranic Arabic Morphological Analysis and Disambiguation

Run CAMeL Tools disambiguation over a gold dataset (Quran or custom), then
score the output. Built around a `RunConfig` dataclass so a "preset" is a
real Python object, not a pile of CLI-flag strings.

## Installation

Run everything below from the repository root.

```bash
# 1. Create and activate a new environment
conda create -n maqam_env python=3.11 -y
conda activate maqam_env

# 2. Clone camel_tools into the repository root
git clone https://github.com/CAMeL-Lab/camel_tools.git

# 3. Add the updated unfactored disambiguator
cp Scripts/unfactored.py camel_tools/camel_tools/disambig/bert/unfactored.py

# 4. Install camel_tools and the remaining pipeline dependencies
pip install ./camel_tools
pip install -r requirements.txt
```

This pipeline uses an updated `BERTUnfactoredDisambiguator` (fine-tuned
model loading with a custom analyzer) that is not yet part of the
`camel_tools` release; step 3 adds it before installation.

## Fine-tuned models

Download the fine-tuned models used by the presets below. `download_models.py`
fetches each `.zip` from the
[`models-v1` release](https://github.com/CAMeL-Lab/MAQAM-Quranic-Corpus/releases/tag/models-v1)
and extracts it into `finetuned_models/` at the repository root.

```bash
cd Scripts

# List what's available
python download_models.py --list

# Download every model
python download_models.py

# Download a specific model only
python download_models.py patb_model
```

## Files

```
Scripts/
  config.py            # every hardcoded path + default knob, one place
  data.py              # gold_df loading, tanzil word attachment, quran_df/custom_analyses
  models.py            # analyzer + disambiguator construction, disambiguation loop
  evaluate.py          # scoring: normalization, per-feature/exact-match/group accuracy
  run.py               # RunConfig dataclass + the pipeline + a standalone CLI
  presets.py           # named experiment configs, as a dict of RunConfig objects
  run_experiments.py   # select/override/run one, several, or all presets
  download_models.py   # fetch the fine-tuned models from the GitHub release
  unfactored.py        # patched BERTUnfactoredDisambiguator for camel_tools
data/                  # MAQAM gold data, Q_Train/Q_Tune/Q_All splits, tanzil words
db/                    # morphology DBs: camel_morph, calima-msa-s31, calima-msa-r13
```

The scripts import each other by module name, so **run every command from
inside `Scripts/`**.

## Paths

`Scripts/config.py` already points every path at the repository's own
folders, so no editing is needed for the paper runs. Change it only if you
move data, DBs, or models elsewhere:

| Constant | Default |
|---|---|
| `CM_DB_PATH` | `db/camel_morph_msa_v1.2.1.db` |
| `S31_DB_PATH` | `db/calima-msa-s31_0.4.2.utf8.db` |
| `R13_DB_PATH` | `db/calima-msa-r13.db` |
| `DATA_DIR` | `data/` (`Q_Train_<script>.csv` / `Q_Tune_<script>.csv` / `Q_All_<script>.csv`) |
| `MODELS_DIR` | `finetuned_models/` |
| `TANZIL_PATH` | `data/tanzil_words.csv` |
| `GOLD_PATH` | `data/MAQAM_Gold_data.csv` |

## Running everything through presets (recommended)

```bash
cd Scripts

# See what's available and where each preset's output lands
python run_experiments.py --list

# Run one
python run_experiments.py r13_imlaay

# Run several
python run_experiments.py s31_imlaay camel_morph_uthmani

# Run everything
python run_experiments.py --all

# No args at all -> interactive picker
python run_experiments.py
```

**Always sanity-check first with `--dry-run`** — it prints every resolved
`RunConfig` field (paths, script, source_mode, output_dir) without loading
any models or touching disk:

```bash
python run_experiments.py --all --dry-run
```

## MAQAM test set (paper runs)

Plain preset runs, no overrides needed — just `python run_experiments.py <preset>`.

| Run | Preset | Notes |
|---|---|---|
| a-1 | `r13_imlaay` | |
| a-2 | `s31_imlaay` | |
| a-3 | `camel_morph_imlaay` | |
| b-1 | `qtrain_model_and_cm_imlaay` | |
| b-2 | `qtrain_model_and_qtrain_imlaay` | |
| b-3 | `qtrain_model_and_cm_qtrain_imlaay` | |
| c-1 | `qtrain_patb_model_and_cm_imlaay` | |
| c-2 | `qtrain_patb_model_and_qtrain_imlaay` | |
| c-3 | `qtrain_patb_model_and_cm_qtrain_imlaay` | best system (fair) |
| d | `qall_patb_model_and_cm_qall_imlaay` | upper bound (test data included in training and custom analyses) |
| e | `qtrain_patb_imlaay_model_and_cm_qtrain_imlaay_imlaay` | |

Run them all in one go:

```bash
python run_experiments.py \
  r13_imlaay s31_imlaay camel_morph_imlaay \
  qtrain_model_and_cm_imlaay qtrain_model_and_qtrain_imlaay qtrain_model_and_cm_qtrain_imlaay \
  qtrain_patb_model_and_cm_imlaay qtrain_patb_model_and_qtrain_imlaay qtrain_patb_model_and_cm_qtrain_imlaay \
  qall_patb_model_and_cm_qall_imlaay \
  qtrain_patb_imlaay_model_and_cm_qtrain_imlaay_imlaay \
  --keep-going --suffix
```

`--suffix` appends the preset name to each output folder, keeping runs that
share a model and analyzer (e.g. b-1/b-2/b-3) in separate folders.

## PATB test set (paper runs)

The PATB test set (`ATB_test_set_0.4.2.csv`) is not distributed with this
repository; it is derived from the LDC Penn Arabic Treebank, so you need
your own copy.

All 6 runs share the same base flags — only `preset`, `--custom-split`,
`--custom-custom-analyses`, and (for the b-series) `--custom-model` change:

```bash
python run_experiments.py <preset> \
  --custom-gold-path "/path/to/ATB_test_set_0.4.2.csv" \
  --custom-back-off ADD_PROP \
  --custom-evaluation-mode calima \
  --custom-gold-apply-bw2ar \
  --custom-split <split> \
  --custom-custom-analyses <on|off> \
  [--custom-model QAll_imlaay_uthmani_patb_model]   # b-series only
```

| Run | Preset | `--custom-split` | `--custom-custom-analyses` | `--custom-model` |
|---|---|---|---|---|
| a-1 | `camel_morph_imlaay` | `train` | `off` | — |
| a-2 | `patb_model_and_cm_quran_only_imlaay` | `all` | `on` | — |
| a-3 | `patb_model_and_cm_quran_prefer_imlaay` | `all` | `on` | — |
| b-1 | `camel_morph_imlaay` | `train` | `off` | `QAll_imlaay_uthmani_patb_model` |
| b-2 | `patb_model_and_cm_quran_only_imlaay` | `all` | `on` | `QAll_imlaay_uthmani_patb_model` |
| b-3 | `patb_model_and_cm_quran_prefer_imlaay` | `all` | `on` | `QAll_imlaay_uthmani_patb_model` |

(a) = each preset's own default model. (b) = same 3 presets, model forced
to `QAll_imlaay_uthmani_patb_model`.

## Running against a different (custom) test set

Pass `--custom-gold-path` to flip **all selected presets** over to a
different CSV, without touching anything else about them:

```bash
python run_experiments.py camel_morph_imlaay --custom-gold-path /path/to/test.csv --dry-run
```

Normal Quran-mode config for `camel_morph_imlaay`:

```
model: patb
analyzer: camel_morph
quran_script: imlaay
word_script: imlaay
split: train_tune
source_mode: none
batch_size: 1
top: 5000
back_off: ADD_PROP
no_custom_analyses: True
data_source: quran
evaluation_mode: None
gold_apply_bw2ar: False
skip_eval: False
output_dir: outputs/patb_camel_morph_imlaay
```

With `--custom-gold-path` added, only two fields change — `data_source`
flips to `none` (loaded as-is, no tanzil merge; your CSV must already have
its own `word`/`word_index` columns) and `output_dir` gets a `_custom`
suffix so it never overwrites the normal run:

```
data_source: none
output_dir: outputs/patb_camel_morph_imlaay_custom
```

`data_dir`/`models_dir`/`tanzil_path`/`quran_script` stay untouched, since
they still drive the Quran/maqam data used to build the custom-analyses
lookup table — that's independent of which gold set you're scoring against.

Other independent overrides, usable alone or combined with
`--custom-gold-path`:

| Flag | Overrides |
|---|---|
| `--custom-evaluation-mode {camel_morph,calima}` | scoring normalization |
| `--custom-quran-script {imlaay,uthmani}` | which Q_Train/Q_Tune/Q_All files feed quran_df |
| `--custom-word-script {imlaay,uthmani}` | which tanzil column gold words come from |
| `--custom-split {train,train_tune,all,all_tune,none}` | quran_df split |
| `--custom-back-off <mode>` | analyzer backoff mode |
| `--custom-custom-analyses {on,off}` | force custom_analyses on/off |
| `--custom-gold-apply-bw2ar` | convert gold diac/lex from Buckwalter to Arabic |
| `--custom-skip-eval` | disambiguate only, skip scoring (for unlabeled data) |
| `--custom-model <name>` | override the model for every selected preset |

To run **both** the normal Quran pass and a custom-data pass for the same
presets, just call the script twice — once with `--custom-gold-path`, once
without.

## Output

Each run writes to `outputs/<derived-or-explicit-name>/`, relative to the
directory you run from (so `Scripts/outputs/...` when following this README):

- `disambig_output.csv` — raw disambiguation output
- `comparison.csv` — per-feature accuracy (skipped if `skip_eval`/`--custom-skip-eval`)
- `summary.csv` — grouped accuracy (lexical / inflectional / clitics / all, macro avg + exact match)

Every run (unless `skip_eval`) also appends one row to
`outputs/combined_summary.csv` at the repository root (path set by
`config.COMBINED_SUMMARY_CSV`),
so you can compare accuracy across every preset you've ever run in one
table.

## License

The dataset and original annotations in the `data/` directory are licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (`CC-BY-4.0`).

The software in the `Scripts/` directory is licensed under the MIT License. See `LICENSE-CODE`.

When using the dataset, please provide appropriate attribution, link to the CC BY 4.0 license, indicate whether changes were made, and cite the associated paper.

