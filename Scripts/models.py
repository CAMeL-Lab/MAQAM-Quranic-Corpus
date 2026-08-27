"""
core/models.py
==============

Everything about building the analyzer/disambiguator and running
disambiguation over a dataframe:
  - build_analyzer / build_disambiguator: setup
  - run_disambiguation_from_df: the actual disambiguation loop
  - apply_clitic_overrides: post-hoc feature patches (e.g. the
    الله / الذين clitic fixes)

See core/data.py for gold_df / quran_df loading, and core/evaluate.py for
scoring disambig_df against gold_df.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data import PROFILE_SOURCE_TAG

DEFAULT_TOP = 5000

# Columns used to group words into "sentences" for disambiguation.
DEFAULT_GROUP_COLS = ("sentence_index",)

# Map a short model key to a folder name under models_dir.
# Add / remove entries freely.
MODEL_CHOICES = {
    "QTrain_patb_imlaay_model":         "QTrain_patb_imlaay_model",
    "QAll_patb_imlaay_model":           "QAll_patb_imlaay_model",
    "QAll_imlaay_model":                "QAll_imlaay_model",
    "QTrain_imlaay_model":              "QTrain_imlaay_model",
    "QTrain_patb_uthmani_model":        "QTrain_patb_uthmani_model",
    "QAll_patb_uthmani_model":          "QAll_patb_uthmani_model",
    "QAll_uthmani_model":               "QAll_uthmani_model",
    "QTrain_uthmani_model":             "QTrain_uthmani_model",
    "QTrain_imlaay_uthmani_model":      "QTrain_imlaay_uthmani_model",
    "QAll_imlaay_uthmani_model":        "QAll_imlaay_uthmani_model",
    "QTrain_imlaay_uthmani_patb_model": "QTrain_imlaay_uthmani_patb_model",
    "QAll_imlaay_uthmani_patb_model":   "QAll_imlaay_uthmani_patb_model",
    "patb_model":                       "patb_model",
    # Special case: use the built-in CALIMA r13 pretrained model instead of a
    # local fine-tuned folder. Handled specially in build_disambiguator().
    "calima_r13_pretrained":            None,
}


# ──────────────────────────────────────────────────────────────────────────
# Setup: analyzer + disambiguator
# ──────────────────────────────────────────────────────────────────────────

def build_analyzer(db_path, back_off):
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    db = MorphologyDB(db_path)
    return Analyzer(db, back_off)


def build_disambiguator(model_key, analyzer, top=DEFAULT_TOP, models_dir=None):
    from camel_tools.disambig.bert import BERTUnfactoredDisambiguator

    if model_key not in MODEL_CHOICES:
        raise ValueError(
            f"Unknown model key {model_key!r}. Choose from: {sorted(MODEL_CHOICES)}"
        )

    folder = MODEL_CHOICES[model_key]
    if model_key == "calima_r13_pretrained" or folder is None:
        # Deliberately do NOT override _analyzer here: the r13 pretrained
        # disambiguator ships with its own bundled analyzer.
        disambig = BERTUnfactoredDisambiguator.pretrained(pretrained_cache=False)
        return disambig

    if not models_dir:
        raise ValueError(f"models_dir is required to load fine-tuned model {model_key!r}.")

    model_path = str(Path(models_dir) / folder)
    disambig = BERTUnfactoredDisambiguator(
        model_path=model_path, analyzer=analyzer, top=top
    )
    disambig._analyzer = analyzer
    return disambig


# ──────────────────────────────────────────────────────────────────────────
# Disambiguation picker + runner
# ──────────────────────────────────────────────────────────────────────────

def _pick_analysis(analyses, assume_sorted, source_mode, source_tag=PROFILE_SOURCE_TAG):
    def _is_backoff_nounprop(a):
        return (a.analysis.get('source') == 'backoff'
                and a.analysis.get('pos') == 'noun_prop')

    def _drop_backoff_nounprop(analyses):
        kept = [a for a in analyses if not _is_backoff_nounprop(a)]
        return kept if kept else analyses

    if source_mode == 'none':
        analyses = _drop_backoff_nounprop(analyses)
        return analyses[0] if assume_sorted else max(analyses, key=lambda a: a.score)

    if source_mode == "quran_only":
        kept = [a for a in analyses
                if a.analysis.get('source') == source_tag or _is_backoff_nounprop(a)]
        if kept:
            analyses = kept
        for a in analyses:
            if a.analysis.get('source') == source_tag:
                return a
        return analyses[0] if assume_sorted else max(analyses, key=lambda a: a.score)

    if source_mode == "quran_prefer":
        analyses = _drop_backoff_nounprop(analyses)
        best = analyses[0] if assume_sorted else max(analyses, key=lambda a: a.score)
        top_group = [a for a in analyses if a.score == best.score]
        for a in top_group:
            if a.analysis.get('source') == source_tag:
                return a
        return best

    if source_mode == "non_quran_prefer":
        analyses = _drop_backoff_nounprop(analyses)
        best = analyses[0] if assume_sorted else max(analyses, key=lambda a: a.score)
        top_group = [a for a in analyses if a.score == best.score]
        for a in top_group:
            if a.analysis.get('source') != source_tag:
                return a
        return best

    raise ValueError(
        f"Unknown source_mode: {source_mode!r}. "
        f"Choose from None, 'quran_only', 'quran_prefer', 'non_quran_prefer'."
    )


def run_disambiguation_from_df(df, disambig, batch_size=1, assume_sorted=True,
                                group_cols=DEFAULT_GROUP_COLS,
                                source_mode="none", source_tag=PROFILE_SOURCE_TAG):
    required = {"word_index", "word"} | set(group_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    df = df.copy()
    if len(group_cols) > 1:
        df["_sentence_key"] = df[list(group_cols)].astype(str).agg("_".join, axis=1)
    else:
        df["_sentence_key"] = df[group_cols[0]]
    sort_cols = list(group_cols) + ["word_index"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    grouped = df.groupby("_sentence_key", sort=False)
    sentences_list, indices_list, keys_list = [], [], []
    for key, g in grouped:
        sentences_list.append(g["word"].astype(str).tolist())
        indices_list.append(g["word_index"].tolist())
        keys_list.append(key)

    all_flattened_analyses = []
    for start in tqdm(range(0, len(sentences_list), batch_size), desc="Disambiguating"):
        batch_sentences = sentences_list[start:start + batch_size]
        batch_indices = indices_list[start:start + batch_size]
        batch_keys = keys_list[start:start + batch_size]
        batch_results = disambig.disambiguate_sentences(batch_sentences)

        for b_idx, sentence in enumerate(batch_results):
            sentence_key = batch_keys[b_idx]
            orig_indices = batch_indices[b_idx]
            for word_pos, word_obj in enumerate(sentence):
                original_idx = orig_indices[word_pos]
                base = {
                    "sentence_key": sentence_key,
                    "word_index": word_pos,
                    "word": word_obj.word,
                    "original_word_index": original_idx,
                }
                if not word_obj.analyses:
                    base.update({"diac": None, "score": None, "quran_found": False})
                    all_flattened_analyses.append(base)
                    continue
                top = _pick_analysis(word_obj.analyses, assume_sorted, source_mode, source_tag)
                base.update({"diac": top.diac, "score": top.score})
                analysis = dict(top.analysis)
                base["quran_found"] = analysis.get('source') == source_tag
                base.update(analysis)
                all_flattened_analyses.append(base)

    return pd.DataFrame(all_flattened_analyses)


def apply_clitic_overrides(disambig_df, overrides):
    for word_list, feature, value in overrides:
        mask = disambig_df['word'].isin(word_list)
        disambig_df.loc[mask, feature] = value
    return disambig_df
