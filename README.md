# Disambiguation Eval Pipeline

Run CAMeL Tools disambiguation over a gold dataset (Quran or custom), then
score the output. Built around a `RunConfig` dataclass so a "preset" is a
real Python object, not a pile of CLI-flag strings.

## Files

```
config.py            # every hardcoded path + default knob, one place
data.py              # gold_df loading, tanzil word attachment, quran_df/custom_analyses
models.py            # analyzer + disambiguator construction, disambiguation loop
evaluate.py          # scoring: normalization, per-feature/exact-match/group accuracy
run.py               # RunConfig dataclass + the pipeline + a standalone CLI
presets.py           # your named experiment configs, as a dict of RunConfig objects
run_experiments.py   # select/override/run one, several, or all presets
```

All 7 files live flat in the same directory.

## One-time setup

Open **`config.py`** and point every path at your machine:

| Constant | What it is |
|---|---|
| `CM_DB_PATH` | camel_morph MorphologyDB `.db` file |
| `S31_DB_PATH` | calima-s31 MorphologyDB `.db` file |
| `R13_DB_PATH` | calima-r13 MorphologyDB `.db` file — **placeholder, fill this in** |
| `DATA_DIR` | folder with `Q_Train_<script>.csv` / `Q_Tune_<script>.csv` / `Q_All_<script>.csv` |
| `MODELS_DIR` | folder holding your fine-tuned model subfolders |
| `TANZIL_PATH` | tanzil-style word source CSV (`tanzil_uthmani`/`tanzil_imlaay` columns) |
| `GOLD_PATH` | your Quran gold evaluation sheet |

## Running everything through presets (recommended)

```bash
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
| d | `qall_patb_model_and_cm_qall_imlaay` | best system (cheating — test is in the train + custom analyses) |
| e | `qtrain_patb_imlaay_model_and_cm_qtrain_imlaay_imlaay` | |

Run them all in one go:

```bash
python run_experiments.py \
  r13_imlaay s31_imlaay camel_morph_imlaay \
  qtrain_model_and_cm_imlaay qtrain_model_and_qtrain_imlaay qtrain_model_and_cm_qtrain_imlaay \
  qtrain_patb_model_and_cm_imlaay qtrain_patb_model_and_qtrain_imlaay qtrain_patb_model_and_cm_qtrain_imlaay \
  qall_patb_model_and_cm_qall_imlaay \
  qtrain_patb_imlaay_model_and_cm_qtrain_imlaay_imlaay \
  --keep-going
```

## PATB test set (paper runs)

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

Each run writes to `outputs/<derived-or-explicit-name>/`:

- `disambig_output.csv` — raw disambiguation output
- `comparison.csv` — per-feature accuracy (skipped if `skip_eval`/`--custom-skip-eval`)
- `summary.csv` — grouped accuracy (lexical / inflectional / clitics / all, macro avg + exact match)

Every run (unless `skip_eval`) also appends one row to
`outputs/combined_summary.csv` (path set by `config.COMBINED_SUMMARY_CSV`),
so you can compare accuracy across every preset you've ever run in one
table.
