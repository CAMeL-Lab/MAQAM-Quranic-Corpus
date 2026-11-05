"""
core/evaluate.py
=================

Scoring disambig_df against gold_df:
  - make_normalizer: per-analyzer-type normalization (camel_morph / calima)
  - feature_accuracy / exact_match / group_accuracy: the metrics
  - evaluate: the full per-feature + summary report for one run
  - append_to_combined_summary: accumulate one row per run into a shared
    wide-format CSV, for comparing presets side by side
"""

import re
from pathlib import Path

import pandas as pd

LEXICAL_FEATURES = ['lex']
INFLECTIONAL_FEATURES = ['asp', 'cas', 'form_gen', 'num', 'gen', 'mod', 'form_num',
                          'per', 'pos', 'stt', 'vox']
CLITIC_FEATURES = ['prc0', 'prc1', 'prc2', 'prc3', 'enc0', 'enc1']
ALL_FEATURES = ['asp', 'cas', 'enc0', 'enc1', 'form_gen', 'gen', 'mod', 'form_num',
                'num', 'per', 'pos', 'prc0', 'prc1', 'prc2', 'prc3', 'stt', 'vox', 'lex']

DEFAULT_ANALYZER_TYPE = "camel_morph"


def ensure_columns(df, feature_cols, fill="0"):
    df = df.copy()
    for feat in feature_cols:
        if feat not in df.columns:
            df[feat] = fill
    return df


def normalize_enc(value):
    if value is None:
        return "0"
    s = str(value).strip()
    if s in ("nan", "NaN", "NA", "na", "0", "0.0", ""):
        return "0"
    return s


def normalize_lex_calima(value):
    if value is None:
        return ""
    s = str(value).strip()
    if s in ("nan", "NaN", "NA", "na", ""):
        return ""
    SUKUN, FATHA, KASRA, DAMMA = '\u0652', '\u064E', '\u0650', '\u064F'
    ALEF, YA, WAW = '\u0627', '\u064A', '\u0648'
    s = s.replace(SUKUN, '')
    s = re.sub(FATHA + ALEF, ALEF, s)
    s = re.sub(KASRA + YA, YA, s)
    s = re.sub(DAMMA + WAW, WAW, s)
    return s


def make_normalizer(analyzer_type):
    if analyzer_type == "camel_morph":
        enc_features = {"enc1", "prc0", "prc1", "prc2", "prc3", "enc0"}

        def _normalize(series, feature):
            if feature in enc_features:
                return series.apply(normalize_enc)
            return series.astype(str)
        return _normalize

    if analyzer_type == "calima":
        enc_features = {"enc0", "enc1", "prc0", "prc1", "prc2", "prc3"}

        def _normalize(series, feature):
            if feature == "lex":
                return series.apply(normalize_lex_calima)
            elif feature in enc_features:
                return series.apply(normalize_enc)
            return series.astype(str)
        return _normalize

    raise ValueError(f"Unknown analyzer_type: {analyzer_type!r}. Use 'camel_morph' or 'calima'.")


def feature_accuracy(pred_df, gold_df, feature, normalize):
    if feature not in gold_df.columns or feature not in pred_df.columns:
        return 0.0
    pred = normalize(pred_df[feature].reset_index(drop=True), feature)
    gold = normalize(gold_df[feature].reset_index(drop=True), feature)
    total = len(gold)
    if total == 0:
        return 0.0
    return (pred == gold).sum() / total


def exact_match(pred_df, gold_df, feature_cols, normalize):
    pred_df = pred_df.reset_index(drop=True)
    gold_df = gold_df.reset_index(drop=True)
    total = len(gold_df)
    if total == 0:
        return 0.0
    all_correct = pd.Series([True] * total)
    for feat in feature_cols:
        if feat not in gold_df.columns or feat not in pred_df.columns:
            all_correct[:] = False
            continue
        pred = normalize(pred_df[feat].reset_index(drop=True), feat)
        gold = normalize(gold_df[feat].reset_index(drop=True), feat)
        all_correct &= (pred == gold)
    return all_correct.sum() / total


def group_accuracy(pred_df, gold_df, feature_cols, normalize):
    if len(feature_cols) == 0:
        return 0.0
    return sum(feature_accuracy(pred_df, gold_df, f, normalize) for f in feature_cols) / len(feature_cols)


def evaluate(disambig_df, gold_df, analyzer_type=DEFAULT_ANALYZER_TYPE):
    normalize = make_normalizer(analyzer_type)

    disambig_df = ensure_columns(disambig_df, ALL_FEATURES, fill="0")
    gold_df = ensure_columns(gold_df, ALL_FEATURES, fill="0")

    results = []
    for feat in ALL_FEATURES:
        acc = feature_accuracy(disambig_df, gold_df, feat, normalize)
        results.append({'feature': feat, 'acc (%)': round(acc * 100, 2)})
    comparison_df = pd.DataFrame(results)

    summary_df = pd.DataFrame([
        {'Group': 'Lexical (macro avg)',
         'acc (%)': round(group_accuracy(disambig_df, gold_df, LEXICAL_FEATURES, normalize) * 100, 4)},
        {'Group': 'Lexical (exact match)',
         'acc (%)': round(exact_match(disambig_df, gold_df, LEXICAL_FEATURES, normalize) * 100, 4)},
        {'Group': 'Inflectional (macro avg)',
         'acc (%)': round(group_accuracy(disambig_df, gold_df, INFLECTIONAL_FEATURES, normalize) * 100, 4)},
        {'Group': 'Inflectional (exact match)',
         'acc (%)': round(exact_match(disambig_df, gold_df, INFLECTIONAL_FEATURES, normalize) * 100, 4)},
        {'Group': 'Clitics (macro avg)',
         'acc (%)': round(group_accuracy(disambig_df, gold_df, CLITIC_FEATURES, normalize) * 100, 4)},
        {'Group': 'Clitics (exact match)',
         'acc (%)': round(exact_match(disambig_df, gold_df, CLITIC_FEATURES, normalize) * 100, 4)},
        {'Group': 'ALL (macro avg)',
         'acc (%)': round(group_accuracy(disambig_df, gold_df, ALL_FEATURES, normalize) * 100, 4)},
        {'Group': 'ALL (exact match)',
         'acc (%)': round(exact_match(disambig_df, gold_df, ALL_FEATURES, normalize) * 100, 4)},
    ])
    return comparison_df, summary_df


def append_to_combined_summary(combined_csv, preset_name, summary_df, meta=None):
    """Pivot a long-format summary_df (Group / acc (%) rows) into a single
    wide row -- {preset, **meta, Group1: acc1, Group2: acc2, ...} -- and
    append it to combined_csv, creating the file with a header if needed."""
    row = {"preset": preset_name}
    if meta:
        row.update(meta)
    for _, r in summary_df.iterrows():
        row[r["Group"]] = r["acc (%)"]
    wide_df = pd.DataFrame([row])

    path = Path(combined_csv)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, wide_df], ignore_index=True)
    else:
        combined = wide_df
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined
