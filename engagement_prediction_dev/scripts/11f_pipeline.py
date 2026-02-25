"""
11f_pipeline.py
===============
LinkedIn Engagement Prediction Pipeline — Experiment 11f

Converts raw LinkedIn posts (same schema as linkedin_posts_new.csv) into
(X, y) ready for the 11f per-tier classifiers.

Steps replicated from:
  01_data_loading_cleaning.ipynb  -> clean_data()
  02_text_preprocessing.ipynb     -> preprocess_text()
  03_feature_engineering.ipynb    -> engineer_features()
  11f_per_tier_models.ipynb       -> add_tier_and_target(), run_pipeline()

Usage
-----
    from scripts.11f_pipeline import run_pipeline
    X, y = run_pipeline(df_raw)         # df_raw = raw posts DataFrame
    # X: 71 content features ready for tier models
    # y: engagement_rate + follower_tier

Required columns in df_raw:
    content, reactions, comments, followers
Optional (used if present):
    media_type
"""

import re
import warnings
import numpy as np
import pandas as pd
import textstat
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ── Load heavy NLP models once at import time ──────────────────────────────────
print("Loading NLP models (spaCy, VADER)...")
_nlp = spacy.load("en_core_web_sm")
_vader = SentimentIntensityAnalyzer()
print("  Done.")

# ── The 71 content features that 11f models expect ────────────────────────────
# Order matches selected_features_data.csv after dropping DROP_COLS.
FEATURE_COLS = [
    "sentiment_compound",
    "text_difficult_words_count",
    "total_engagement_elements",
    "readability_flesch_kincaid",
    "text_lexical_diversity",
    "readability_gunning_fog",
    "sentence_count",
    "text_avg_sentence_length",
    "topic_count",
    "sentiment_x_readability",
    "ner_location_count",
    "style_question_marks",
    "ner_org_count",
    "style_has_parentheses",
    "ner_person_count",
    "ner_date_count",
    "unique_emoji_count",
    "hashtag_count_extracted",
    "style_quote_marks",
    "has_aspirational",
    "has_location_mention",
    "length_score",
    "emoji_count",
    "is_multi_topic",
    "style_parentheses_count",
    "has_contrast",
    "style_all_caps_words",
    "has_personal_story",
    "style_emoji_count",
    "style_has_emoji",
    "style_has_all_caps",
    "style_number_count",
    "has_direct_address",
    "topic_tech",
    "style_exclamation_marks",
    "has_family",
    "topic_personal_dev",
    "style_has_question",
    "link_penalty_score",
    "ner_money_count",
    "has_transformation",
    "has_person_mention",
    "style_bullet_count",
    "url_count",
    "style_has_numbers",
    "has_vulnerability",
    "topic_business",
    "has_specific_numbers",
    "has_value_promise",
    "mention_count",
    "style_has_exclamation",
    "topic_finance",
    "has_adversity_learning",
    "topic_leadership",
    "topic_career",
    "ner_product_count",
    "style_has_quotes",
    "has_entities",
    "link_spam_penalty",
    "style_has_bullets",
    "is_low_effort_link",
    "is_promotional",
    "has_hidden_truth",
    "hook_x_power_score",
    "has_org_mention",
    "hook_score",
    "ner_event_count",
    "has_announcement_hook",
    "has_specific_time_content",
    "has_recency_hook",
    "has_underdog",
]

# Follower tier breakpoints: micro / small / medium / large
_TIER_BINS = [0, 10_000, 50_000, 200_000, float("inf")]
_TIER_LABELS = [0, 1, 2, 3]
TIER_NAMES = {0: "micro (<10k)", 1: "small (10k-50k)", 2: "medium (50k-200k)", 3: "large (>200k)"}


# ── STEP 1: Cleaning (01_data_loading_cleaning.ipynb) ─────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate 01_data_loading_cleaning steps:
      - Drop rows with missing content or engagement columns
      - Type-coerce numeric columns, clip negatives
      - Remove content-length outliers (1st–99th percentile)

    Note: follower/reaction capping at 95th percentile is intentionally
    skipped here — inference should use actual values, not training caps.
    """
    df = df.copy()

    for col in ["content", "reactions", "comments", "followers"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from input DataFrame.")

    # Drop missing content
    df = df[df["content"].notna() & (df["content"].str.strip() != "")].copy()

    # Numeric coercion
    for col in ["reactions", "comments", "followers"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    df["reactions"] = df["reactions"].astype(int)
    df["comments"] = df["comments"].astype(int)

    # Ensure media_type exists (used in feature engineering)
    if "media_type" not in df.columns:
        df["media_type"] = None

    # Content-length outlier removal: 1st–99th percentile (matches notebook)
    lengths = df["content"].str.len()
    p1, p99 = lengths.quantile(0.01), lengths.quantile(0.99)
    df = df[(lengths >= p1) & (lengths <= p99)].copy()

    df.reset_index(drop=True, inplace=True)
    return df


# ── STEP 2: Text Preprocessing (02_text_preprocessing.ipynb) ─────────────────

_URL_PAT = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
_MENTION_PAT = r"@[A-Za-z0-9_\s]+"
_HASHTAG_PAT = r"#[A-Za-z0-9_]+"
_EMOJI_PAT = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)


def _extract_urls(text):
    if pd.isna(text):
        return 0, []
    urls = re.findall(_URL_PAT, str(text))
    return len(urls), urls


def _extract_mentions(text):
    if pd.isna(text):
        return 0, []
    mentions = [m.strip() for m in re.findall(_MENTION_PAT, str(text))]
    return len(mentions), mentions


def _extract_hashtags(text):
    if pd.isna(text):
        return 0, []
    return len(re.findall(_HASHTAG_PAT, str(text))), re.findall(_HASHTAG_PAT, str(text))


def _extract_emojis(text):
    if pd.isna(text):
        return 0, 0, []
    found = _EMOJI_PAT.findall(str(text))
    flat = "".join(found)
    return len(flat), len(set(flat)), list(flat)


def _clean_content(text):
    if pd.isna(text):
        return ""
    t = re.sub(_URL_PAT, "[URL]", str(text))
    t = re.sub(_MENTION_PAT, "[MENTION]", t)
    t = _EMOJI_PAT.sub("", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9#\s.!?,;:\-\'\"]+", "", t)
    return t.lower().strip()


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Extract URL/mention/hashtag/emoji features and create clean_content."""
    df = df.copy()

    df[["url_count", "urls_list"]] = df["content"].apply(
        lambda x: pd.Series(_extract_urls(x))
    )
    df["has_external_link"] = (df["url_count"] > 0).astype(int)

    df[["mention_count", "mentions_list"]] = df["content"].apply(
        lambda x: pd.Series(_extract_mentions(x))
    )

    df[["hashtag_count_extracted", "hashtags_list"]] = df["content"].apply(
        lambda x: pd.Series(_extract_hashtags(x))
    )

    df[["emoji_count", "unique_emoji_count", "emojis_list"]] = df["content"].apply(
        lambda x: pd.Series(_extract_emojis(x))
    )

    df["clean_content"] = df["content"].apply(_clean_content)
    df["word_count_clean"] = df["clean_content"].str.split().str.len().fillna(0).astype(int)
    df["sentence_count"] = df["clean_content"].apply(
        lambda x: max(1, len(re.split(r"[.!?]+", str(x)))) if pd.notna(x) else 1
    )

    return df


# ── STEP 3: Feature Engineering (03_feature_engineering.ipynb) ───────────────

def _length_score(wc):
    if pd.isna(wc) or wc < 50:  return -12
    if wc < 80:                  return -3
    if wc < 100:                 return 5
    if wc <= 200:                return 8
    if wc <= 300:                return 3
    return -15


def _first_sentence(text):
    if pd.isna(text) or text == "":
        return ""
    parts = re.split(r"[.!?]+", str(text))
    return parts[0].strip().lower() if parts else ""


def _detect_hook(s):
    if not s:
        return "no_hook", 0
    if re.search(r"\bnever\b.*\b(thought|believed|imagined|expected)", s):
        return "never_narrative", 15
    if re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", s):
        return "specific_time", 12
    if s.startswith('"') or s.startswith("'"):
        return "quote_hook", 10
    if re.search(r"\b(stop|start|quit|avoid|never)\s+(doing|using|saying|thinking)", s):
        return "contrarian", 7
    if re.search(r"\bi\s+used\s+to\s+(think|believe|assume)", s):
        return "belief_transformation", 6
    if re.search(r"\b(it's official|today|finally|announcing)\b", s):
        return "announcement", 6
    if re.search(r"\beveryone('s| is)\b", s):
        return "everyone_pattern", 5
    if re.search(r"\bjust\s+(realized|learned|discovered|noticed)", s):
        return "realization", 5
    if re.search(r"\b(hours? ago|last (week|month)|yesterday|recently)\b", s):
        return "recency", 4
    return "no_hook", 0


_POWER_W = {
    "underdog": 9, "transformation": 8, "cta_question": 8, "hidden_truth": 10,
    "vulnerability": 7, "family": 8, "specific_time_content": 6, "specific_numbers": 4,
    "adversity_learning": 5, "value_promise": 4, "list_format": 5, "contrast": 5,
    "aspirational": 6, "direct_address": 3, "personal_story": 5,
}


def _power_patterns(text):
    empty = {f"has_{k}": 0 for k in _POWER_W}
    empty.update({"power_pattern_count": 0, "power_pattern_score": 0})
    if pd.isna(text) or text == "":
        return empty
    t = str(text).lower()
    p = {
        "underdog":              int(bool(re.search(r"\b(immigrant|refugee|struggle|overcome|against all odds|bootstrapped|from nothing)\b", t))),
        "transformation":        int(bool(re.search(r"\b(before.*after|used to.*now|transformed|changed my life|went from.*to|journey)\b", t))),
        "cta_question":          int(bool(re.search(r"\b(what do you think|agree or disagree|comment below|share your|thoughts?)\?", t))),
        "hidden_truth":          int(bool(re.search(r"\b(nobody (posts|talks|mentions)|no one (talks|discusses)|hidden truth|secret)\b", t))),
        "vulnerability":         int(bool(re.search(r"\b(failed|mistake|wrong|scared|afraid|vulnerable|honest|transparent|raw|real talk)\b", t))),
        "family":                int(bool(re.search(r"\b(daughter|son|kids|children|parent|mom|dad|family|wife|husband)\b", t))),
        "specific_time_content": int(bool(re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b|\b(morning|afternoon|evening|midnight)\b", t))),
        "specific_numbers":      int(bool(re.search(r"\b\d+%|\$\d+|\d+x|\d+k\b|\b(increased|decreased|grew).*\d+", t))),
        "adversity_learning":    int(bool(re.search(r"\b(learned|lesson|taught me|experience taught|failure taught)\b", t))),
        "value_promise":         int(bool(re.search(r"\b(here(s| are|s how)|\d+ (ways|tips|steps|secrets|lessons|strategies))\b", t))),
        "list_format":           int(bool(re.search(r"(\n\s*[-•*\d+\.]\s+)|(first.*second.*third)|^\s*\d+\.\s+", t))),
        "contrast":              int(bool(re.search(r"\b(but|however|instead|whereas|unlike|versus|vs\.?)\b", t))),
        "aspirational":          int(bool(re.search(r"\b(become|achieve|reach|attain|success|freedom|wealth|dream)\b", t))),
        "direct_address":        int(bool(re.search(r"\b(you (will|can|should|become|achieve)|your)\b", t))),
        "personal_story":        int(bool(re.search(r"\b(i (was|did|went|worked|started)|my (story|experience|journey))\b", t))),
    }
    result = {f"has_{k}": v for k, v in p.items()}
    result["power_pattern_count"] = sum(p.values())
    result["power_pattern_score"] = sum(p[k] * _POWER_W[k] for k in p)
    return result


def _promotional_score(text):
    if pd.isna(text) or text == "":
        return 0
    t = str(text).lower()
    score = 0
    for kw in ["our product", "we built", "we launched", "buy now", "sign up",
                "register now", "limited time", "special offer", "discount"]:
        if kw in t:
            score += 2
    for kw in ["product", "service", "solution", "demo", "launch", "release",
                "announcement", "introducing", "features", "platform"]:
        if re.search(r"\b" + kw + r"\b", t):
            score += 1
    return score


def _ner(text):
    empty = {"PERSON": 0, "ORG": 0, "GPE": 0, "DATE": 0,
             "MONEY": 0, "PRODUCT": 0, "EVENT": 0, "total_entities": 0}
    if pd.isna(text) or str(text).strip() == "":
        return empty
    doc = _nlp(str(text)[:50_000])
    counts = {k: 0 for k in empty}
    for ent in doc.ents:
        if ent.label_ in counts:
            counts[ent.label_] += 1
    counts["total_entities"] = sum(counts[k] for k in
                                   ["PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT"])
    return counts


def _readability(text):
    if pd.isna(text) or str(text).strip() == "":
        return {"fk": 0.0, "fog": 0.0, "ease": 0.0}
    t = str(text)
    return {
        "fk":   textstat.flesch_kincaid_grade(t),
        "fog":  textstat.gunning_fog(t),
        "ease": textstat.flesch_reading_ease(t),
    }


def _text_stats(text):
    if pd.isna(text) or str(text).strip() == "":
        return {"ld": 0.0, "asl": 0.0, "dw": 0}
    words = str(text).lower().split()
    wc = len(words) or 1
    sc = textstat.sentence_count(str(text)) or 1
    return {
        "ld":  len(set(words)) / wc,
        "asl": wc / sc,
        "dw":  textstat.difficult_words(str(text)),
    }


def _media_score(media_type):
    if pd.isna(media_type):
        return 0
    m = str(media_type).lower()
    if "video" in m:                        return 10
    if "carousel" in m or "document" in m:  return 8
    if "image" in m or "photo" in m:        return 5
    return 0


_TOPICS = {
    "tech":         ["technology", "ai", "software", "data", "digital", "tech", "innovation", "machine learning"],
    "business":     ["business", "marketing", "sales", "strategy", "growth", "entrepreneur", "startup"],
    "career":       ["career", "job", "hiring", "resume", "interview", "professional", "workplace"],
    "leadership":   ["leadership", "management", "team", "leader", "ceo", "executive"],
    "personal_dev": ["learning", "skills", "development", "education", "training", "course"],
    "finance":      ["finance", "investment", "money", "funding", "financial", "revenue"],
}

_STYLE_EMOJI = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps from 03_feature_engineering.ipynb.
    Requires that preprocess_text() has already been called (needs clean_content,
    word_count_clean, has_external_link, emoji_count, etc.).
    """
    df = df.copy()

    # Length score
    df["length_score"] = df["word_count_clean"].apply(_length_score)

    # Media features (has_media needed for link_spam_penalty)
    df["media_score"] = df["media_type"].apply(_media_score)
    df["has_media"] = (df["media_score"] > 0).astype(int)

    # Hook patterns
    df["_first_sent"] = df["clean_content"].apply(_first_sentence)
    df[["hook_type", "hook_score"]] = df["_first_sent"].apply(
        lambda x: pd.Series(_detect_hook(x))
    )
    for hook_type in ["never_narrative", "specific_time", "quote_hook", "contrarian",
                       "belief_transformation", "announcement", "everyone_pattern",
                       "realization", "recency"]:
        df[f"has_{hook_type}_hook"] = (df["hook_type"] == hook_type).astype(int)

    # Power patterns
    pp_df = pd.DataFrame(df["clean_content"].apply(_power_patterns).tolist())
    df = pd.concat([df, pp_df], axis=1)

    # Engagement elements composite (needed before link penalties)
    df["total_engagement_elements"] = (df["hook_score"] > 0).astype(int) + df["power_pattern_count"]

    # Promotional flag
    df["promotional_score"] = df["clean_content"].apply(_promotional_score)
    df["is_promotional"] = (df["promotional_score"] >= 2).astype(int)

    # Link penalties
    df["link_penalty_score"] = df["has_external_link"] * -18
    df["is_low_effort_link"] = (
        (df["word_count_clean"] < 80)
        & (df["has_external_link"] == 1)
        & (df["total_engagement_elements"] < 2)
    ).astype(int)
    df["link_spam_penalty"] = (
        (df["word_count_clean"] < 60)
        & (df["has_external_link"] == 1)
        & (df["has_media"] == 0)
    ).astype(int) * -10

    # Sentiment (on original content)
    sentiments = df["content"].apply(
        lambda x: _vader.polarity_scores(x) if pd.notna(x) else {"compound": 0.0}
    )
    df["sentiment_compound"] = sentiments.apply(lambda x: x["compound"])

    # Named Entity Recognition (on original content)
    ner_results = df["content"].apply(_ner)
    df["ner_person_count"]   = ner_results.apply(lambda x: x["PERSON"])
    df["ner_org_count"]      = ner_results.apply(lambda x: x["ORG"])
    df["ner_location_count"] = ner_results.apply(lambda x: x["GPE"])
    df["ner_date_count"]     = ner_results.apply(lambda x: x["DATE"])
    df["ner_money_count"]    = ner_results.apply(lambda x: x["MONEY"])
    df["ner_product_count"]  = ner_results.apply(lambda x: x["PRODUCT"])
    df["ner_event_count"]    = ner_results.apply(lambda x: x["EVENT"])
    df["ner_total_entities"] = ner_results.apply(lambda x: x["total_entities"])
    df["has_person_mention"]  = (df["ner_person_count"]   > 0).astype(int)
    df["has_org_mention"]     = (df["ner_org_count"]      > 0).astype(int)
    df["has_location_mention"]= (df["ner_location_count"] > 0).astype(int)
    df["has_entities"]        = (df["ner_total_entities"] > 0).astype(int)

    # Readability (on original content)
    read_results = df["content"].apply(_readability)
    df["readability_flesch_kincaid"] = read_results.apply(lambda x: x["fk"])
    df["readability_gunning_fog"]    = read_results.apply(lambda x: x["fog"])
    df["readability_flesch_ease"]    = read_results.apply(lambda x: x["ease"])

    # Text statistics (on original content)
    ts = df["content"].apply(_text_stats)
    df["text_lexical_diversity"]    = ts.apply(lambda x: x["ld"])
    df["text_avg_sentence_length"]  = ts.apply(lambda x: x["asl"])
    df["text_difficult_words_count"]= ts.apply(lambda x: x["dw"])

    # Style features (on original content)
    df["style_question_marks"]    = df["content"].str.count(r"\?").fillna(0).astype(int)
    df["style_has_question"]      = (df["style_question_marks"] > 0).astype(int)
    df["style_exclamation_marks"] = df["content"].str.count(r"!").fillna(0).astype(int)
    df["style_has_exclamation"]   = (df["style_exclamation_marks"] > 0).astype(int)
    df["style_emoji_count"]       = df["content"].str.count(_STYLE_EMOJI).fillna(0).astype(int)
    df["style_has_emoji"]         = (df["style_emoji_count"] > 0).astype(int)
    df["style_all_caps_words"]    = df["content"].str.findall(r"\b[A-Z]{3,}\b").str.len().fillna(0).astype(int)
    df["style_has_all_caps"]      = (df["style_all_caps_words"] > 0).astype(int)
    df["style_quote_marks"]       = df["content"].str.count(r'["\']').fillna(0).astype(int)
    df["style_has_quotes"]        = (df["style_quote_marks"] >= 2).astype(int)
    df["style_parentheses_count"] = df["content"].str.count(r"[()]").fillna(0).astype(int)
    df["style_has_parentheses"]   = (df["style_parentheses_count"] >= 2).astype(int)
    df["style_number_count"]      = df["content"].str.findall(r"\d+").str.len().fillna(0).astype(int)
    df["style_has_numbers"]       = (df["style_number_count"] > 0).astype(int)
    df["style_bullet_count"]      = df["content"].str.findall("[•●○▪▫■□★☆→➤✓✔]").str.len().fillna(0).astype(int)
    df["style_has_bullets"]       = (df["style_bullet_count"] > 0).astype(int)

    # Topics (on clean_content)
    for name, keywords in _TOPICS.items():
        pat = "|".join([rf"\b{kw}\b" for kw in keywords])
        df[f"topic_{name}"] = df["clean_content"].str.contains(
            pat, case=False, regex=True, na=False
        ).astype(int)
    topic_cols = [f"topic_{n}" for n in _TOPICS]
    df["topic_count"]    = df[topic_cols].sum(axis=1)
    df["is_multi_topic"] = (df["topic_count"] > 1).astype(int)

    # Derived composite scores
    df["hook_x_power_score"]     = df["hook_score"] * df["power_pattern_score"]
    df["sentiment_x_readability"]= df["sentiment_compound"] * df["readability_flesch_ease"]

    # Drop temporary columns
    df.drop(columns=["_first_sent"], inplace=True, errors="ignore")

    return df


# ── STEP 4: Add tier routing & engagement rate ─────────────────────────────────

def add_tier_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engagement_rate and follower_tier.
    follower_tier is used for routing to the correct model — it is NOT
    passed as a feature to any model (zero variance within each tier).
    """
    df = df.copy()
    # Avoid division by zero: posts with 0 followers get NaN (excluded later)
    df["engagement_rate"] = np.where(
        df["followers"] > 0,
        (df["reactions"] + df["comments"]) / (df["followers"] / 1000.0),
        np.nan,
    )
    df["follower_tier"] = pd.cut(
        df["followers"],
        bins=_TIER_BINS,
        labels=_TIER_LABELS,
        include_lowest=True,
    ).astype("Int64").astype(int)
    return df


# ── Main public function ───────────────────────────────────────────────────────

def run_pipeline(df_raw: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Full pipeline: raw LinkedIn posts -> (X, y).

    Parameters
    ----------
    df_raw : DataFrame
        Raw LinkedIn posts. Required columns: content, reactions, comments, followers.
        Optional: media_type (assumed None/missing if absent).
    verbose : bool
        Print progress steps.

    Returns
    -------
    X : DataFrame, shape (n, 71)
        Content features, ready to slice by follower_tier and pass to the
        corresponding 11f tier model.
    y : DataFrame
        Columns: engagement_rate (float), follower_tier (int 0-3).
        Rows with zero followers are dropped (engagement_rate undefined).
    """
    if verbose:
        print(f"[pipeline] Input: {len(df_raw)} rows")

    df = clean_data(df_raw)
    if verbose:
        print(f"[pipeline] After cleaning: {len(df)} rows")

    df = preprocess_text(df)
    if verbose:
        print("[pipeline] Text preprocessing done")

    df = engineer_features(df)
    if verbose:
        print("[pipeline] Feature engineering done")

    df = add_tier_and_target(df)

    # Drop rows where engagement_rate is undefined (zero followers)
    df = df[df["engagement_rate"].notna()].copy()
    if verbose:
        print(f"[pipeline] After dropping zero-follower rows: {len(df)}")

    # Ensure all 71 feature columns exist (fill missing with 0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLS].fillna(0).reset_index(drop=True)
    y = df[["engagement_rate", "follower_tier"]].reset_index(drop=True)

    if verbose:
        print(f"[pipeline] Done -> X: {X.shape}, y: {y.shape}")
        tier_dist = y["follower_tier"].value_counts().sort_index()
        for t, n in tier_dist.items():
            print(f"           tier {t} ({TIER_NAMES[t]}): {n} posts")

    return X, y


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/linkedin_posts_new.csv"
    raw = pd.read_csv(path)
    print(f"Loaded {len(raw)} rows from {path}")
    X, y = run_pipeline(raw)
    print(f"\nFinal shapes -> X: {X.shape}, y: {y.shape}")
    print(y["follower_tier"].value_counts().sort_index())
