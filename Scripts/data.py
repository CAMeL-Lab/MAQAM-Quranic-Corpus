"""
core/data.py
============

Everything about loading and shaping input dataframes:
  - gold_df: your evaluation set (word/word_index + gold feature columns)
  - tanzil word attachment: filling gold_df['word'] from a separate
    tanzil-style word source CSV
  - quran_df / custom_analyses: the profile corpus used to seed the
    disambiguator with known analyses (kept strictly separate from gold_df)

None of this module touches the analyzer/disambiguator or scoring --
see core/models.py and core/evaluate.py for those.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Tag written into custom_analyses['source'] for rows that came from the
# quran_df profile corpus (as opposed to the analyzer's own analyses).
PROFILE_SOURCE_TAG = "quran_source"

# Which Q_*_<script>.csv files make up each --split.
QURAN_SPLIT_FILES = {
    "train":      ["Q_Train_{script}.csv"],
    "train_tune": ["Q_Train_{script}.csv", "Q_Tune_{script}.csv"],
    "all":        ["Q_All_{script}.csv"],
    "all_tune":   ["Q_All_{script}.csv", "Q_Tune_{script}.csv"],
    "none":       [],
}

# Full CAMeL analysis feature set -- every profile row is padded out to
# have all of these columns before being turned into custom_analyses.
CAMEL_FEATURES = [
    'diac', 'lex', 'caphi', 'gloss', 'bw', 'pos', 'catib6', 'ud', 'root',
    'pattern', 'prc3', 'prc2', 'prc1', 'prc0', 'per', 'asp', 'vox', 'mod',
    'form_num', 'form_gen', 'gen', 'num', 'stt', 'cas', 'enc0', 'enc1', 'rat',
    'source', 'd1seg', 'd2seg', 'd3seg', 'atbseg', 'd1tok', 'd2tok', 'd3tok',
    'atbtok', 'bwtok', 'pos_logprob', 'lex_logprob', 'pos_lex_logprob',
    'stem', 'stemgloss', 'stemcat'
]


# ──────────────────────────────────────────────────────────────────────────
# gold_df
# ──────────────────────────────────────────────────────────────────────────

def load_gold_df(gold_csv, apply_bw2ar=False, cols_to_convert=("diac", "lex")):
    """Load an arbitrary gold-standard CSV. Optionally convert Buckwalter
    columns to Arabic script (set apply_bw2ar=True if your gold CSV is BW)."""
    gold_df = pd.read_csv(gold_csv)
    if apply_bw2ar:
        from camel_tools.utils.charmap import CharMapper
        bw2ar = CharMapper.builtin_mapper('bw2ar')
        for col in cols_to_convert:
            if col in gold_df.columns:
                gold_df[col] = gold_df[col].apply(
                    lambda x: bw2ar(str(x)) if pd.notna(x) else x
                )
    required = {"word", "word_index"}
    missing = required - set(gold_df.columns)
    if missing:
        raise ValueError(f"gold_csv is missing required columns: {sorted(missing)}")
    return gold_df


# ──────────────────────────────────────────────────────────────────────────
# tanzil word attachment
# ──────────────────────────────────────────────────────────────────────────

def build_words_from_tanzil(tanzil_csv, script_type="uthmani", surahs=None):
    """Load a tanzil-style word source CSV and return the word column for the
    requested script ('uthmani' -> 'tanzil_uthmani', 'imlaay' -> 'tanzil_imlaay'),
    optionally filtered to a list of surah_index values."""
    tanzil_df = pd.read_csv(tanzil_csv)
    if surahs:
        tanzil_df = tanzil_df[tanzil_df["surah_index"].isin(surahs)].reset_index(drop=True)
    col = f"tanzil_{script_type}"
    if col not in tanzil_df.columns:
        raise ValueError(f"Column {col!r} not found in {tanzil_csv}. "
                          f"Available columns: {list(tanzil_df.columns)}")
    return tanzil_df[col].reset_index(drop=True)


def attach_tanzil_words(gold_df, words_series):
    """Attach a 'word' column to gold_df from a tanzil word series. The two
    must align 1:1 by row order (same length)."""
    if len(words_series) != len(gold_df):
        raise ValueError(
            f"Row count mismatch: tanzil words ({len(words_series)}) vs "
            f"gold_df ({len(gold_df)}). They must align 1:1 by row order -- "
            f"check your surah filter / gold_csv."
        )
    gold_df = gold_df.copy()
    gold_df["word"] = words_series.values
    return gold_df


# ──────────────────────────────────────────────────────────────────────────
# quran_df / custom_analyses (profile corpus, separate from gold_df)
# ──────────────────────────────────────────────────────────────────────────

def resolve_quran_csvs(explicit_csvs, quran_dir, script_type, split):
    """Decide which files make up quran_df: explicit --profile-csv list wins;
    otherwise build the list from quran_dir/script/split using QURAN_SPLIT_FILES."""
    if explicit_csvs:
        return list(explicit_csvs)
    if not split or split == "none":
        return []
    patterns = QURAN_SPLIT_FILES.get(split)
    if patterns is None:
        raise ValueError(f"Unknown quran split {split!r}. Choose from: {sorted(QURAN_SPLIT_FILES)}")
    return [str(Path(quran_dir) / p.format(script=script_type)) for p in patterns]


def build_profile(profile_csvs, source_tag=PROFILE_SOURCE_TAG):
    """Builds `quran_df` (the profile/reference corpus -- separate from
    gold_df) plus the `custom_analyses` override table fed into the
    disambiguator. This is the generalized replacement for the old Quran
    train+tune concatenation + log-prob computation.

    profile_csvs: list of CSV paths. Each should have sentence_index,
    word_index, word, lex, pos (and ideally the rest of CAMEL_FEATURES).
    Multiple CSVs are concatenated with sentence_index offsets so they don't
    collide (e.g. a train split + a tune split).

    Returns (quran_df, custom_analyses). Returns (None, {}) if profile_csvs
    is empty -- i.e. no custom analyses will be used.
    """
    if not profile_csvs:
        return None, {}

    frames = []
    offset = 0
    for path in profile_csvs:
        df = pd.read_csv(path)
        if "sentence_index" in df.columns:
            df = df.copy()
            df["sentence_index"] = df["sentence_index"] + offset
            offset = df["sentence_index"].max() + 1
        frames.append(df)
    profile_df = pd.concat(frames, ignore_index=True)

    total = len(profile_df)

    profile_df['lex'] = profile_df.get('lex', '').fillna('')
    lex_counts = profile_df['lex'].value_counts()
    profile_df['lex_logprob'] = profile_df['lex'].apply(
        lambda x: np.log(lex_counts[x] / total) if x != '' else -99
    )

    profile_df['pos'] = profile_df.get('pos', '').fillna('')
    pos_counts = profile_df['pos'].value_counts()
    profile_df['pos_logprob'] = profile_df['pos'].apply(
        lambda x: np.log(pos_counts[x] / total) if x != '' else -99
    )

    profile_df['_pos_lex'] = profile_df['pos'] + '_' + profile_df['lex']
    pos_lex_counts = profile_df['_pos_lex'].value_counts()
    profile_df['pos_lex_logprob'] = profile_df['_pos_lex'].apply(
        lambda x: np.log(pos_lex_counts[x] / total) if x != '_' else -99
    )
    profile_df.drop(columns=['_pos_lex'], inplace=True)

    for feat in CAMEL_FEATURES:
        if feat not in profile_df.columns:
            profile_df[feat] = ''
    profile_df['source'] = source_tag
    for col in ['pos_lex_logprob', 'lex_logprob', 'pos_logprob']:
        profile_df[col] = pd.to_numeric(profile_df[col], errors='coerce').fillna(-99)

    custom_analyses = (
        profile_df.groupby(['word'], sort=False)
        .apply(lambda g: g[CAMEL_FEATURES].to_dict('records'))
        .rename('analyses')
        .reset_index()
        .pipe(lambda df: {row['word']: row['analyses'] for _, row in df.iterrows()})
    )
    return profile_df, custom_analyses
