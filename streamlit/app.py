"""
TrendPilot — LinkedIn Post Creator
Multi-step Streamlit app:
  1. Input LinkedIn bio + follower count → identify trending topics
  2. Select topic → LLM recommends post titles
  3. Select post title → generate 3 post variants
  4. Engagement prediction (reactions + comments) for each variant
  5. Select & copy final post
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import subprocess
import joblib
import unicodedata
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "03_engagement_prediction")
sys.path.insert(0, SCRIPT_DIR)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrendPilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Trend identification (from 01_trend_identification_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _import_trend_module():
    """Import the trend module once and cache it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trend_id",
        os.path.join(SCRIPT_DIR, "01_trend_identification_v2.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_trend_analysis(profile_text: str) -> pd.DataFrame:
    mod = _import_trend_module()
    return mod.get_trending_topics(profile_text)


def run_topic_selection(trending_df: pd.DataFrame, profile_text: str, chosen_topic: str | None = None) -> str:
    mod = _import_trend_module()
    return mod.select_post_topic(trending_df, profile_text, chosen_topic=chosen_topic)


def parse_post_titles(llm_output: str) -> list[dict]:
    """Parse the structured LLM response into a list of {signal, title} dicts."""
    titles = []
    for i in range(1, 4):
        signal_match = re.search(rf"SIGNAL\s+{i}:\s*(.+)", llm_output, re.IGNORECASE)
        title_match = re.search(rf"POST\s+TITLE\s+{i}:\s*(.+)", llm_output, re.IGNORECASE)
        if title_match:
            titles.append({
                "signal": signal_match.group(1).strip() if signal_match else "",
                "title": title_match.group(1).strip(),
            })
    return titles


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Post generation (from 02_Post_generator.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

MASTER_SYSTEM_PROMPT = """
You are an expert LinkedIn content strategist and copywriter.

Your goal is to generate authentic, high-engagement LinkedIn posts
that maximize reach and interaction while avoiding promotional suppression.

Optimize for:
- Authentic storytelling over promotion
- Strong first-sentence hooks
- Specific details (numbers, moments, experiences)
- Personal insights or transformations

Hard constraints:
- Target length: 100–200 words
- NO external links
- Minimal promotional language
- Tone: human, reflective, conversational
- Platform: LinkedIn (B2B audience)
- Add relevant hashtags in the last line

Output should feel like a real professional sharing a genuine insight.
"""

HOOK_STYLES = [
    "Contrarian — challenge a common assumption in the field",
    "Personal transformation — 'I used to believe X, now I know Y'",
    "Hidden insight — something that most people overlook",
]


def _build_user_prompt(topic: str, profile_context: str, hook_style: str) -> str:
    return f"""
Create a high-performing LinkedIn post using the following inputs.

Topic:
{topic}

Core personal context (author's background):
{profile_context}

Hook preference:
{hook_style}

Constraints:
- 100–200 words
- No external links
- No promotional language
- Short paragraphs (2–3 sentences max per paragraph)
- End with a thought-provoking question to drive comments
"""


@st.cache_resource(show_spinner=False)
def _get_openai_client():
    from openai import OpenAI
    return OpenAI()


def generate_post_variants(post_title: str, profile_text: str) -> list[dict]:
    """Generate 3 post variants for a given post title."""
    client = _get_openai_client()
    variants = []
    for hook in HOOK_STYLES:
        prompt = _build_user_prompt(post_title, profile_text, hook)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.75,
        )
        text = response.choices[0].message.content.strip()
        variants.append({
            "hook_style": hook.split("—")[0].strip(),
            "post_text": text,
            "word_count": len(text.split()),
        })
    return variants


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Feature engineering for engagement prediction inference
# ══════════════════════════════════════════════════════════════════════════════

_EMOJI_PAT = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F9FF"
    r"\U0001FA00-\U0001FA9F\U00002600-\U000027BF\U0001F1E0-\U0001F1FF]+",
    flags=re.UNICODE,
)
_URL_PAT = re.compile(r"https?://\S+")

_TOPICS = {
    "tech":         r"\b(technology|ai|software|data|digital|innovation|machine learning|llm|gpt|cloud|api)\b",
    "business":     r"\b(business|marketing|sales|strategy|growth|entrepreneur|startup|revenue|market)\b",
    "career":       r"\b(career|job|hiring|resume|interview|professional|workplace|promotion|salary)\b",
    "leadership":   r"\b(leadership|management|team|leader|ceo|executive|manager|culture)\b",
    "personal_dev": r"\b(learning|skills|development|education|training|course|mindset|habit)\b",
    "finance":      r"\b(finance|investment|money|funding|financial|revenue|profit|equity)\b",
}

_PP_WEIGHTS = {
    "underdog": 9, "transformation": 8, "cta_question": 8, "hidden_truth": 10,
    "vulnerability": 7, "family": 8, "specific_time_content": 6,
    "specific_numbers": 4, "adversity_learning": 5, "value_promise": 4,
    "list_format": 5, "contrast": 5, "aspirational": 6,
    "direct_address": 3, "personal_story": 5,
}


def _length_score(wc: int) -> int:
    if   100 <= wc <= 200: return  8
    elif  80 <= wc <  100: return  5
    elif 200 <  wc <= 300: return  3
    elif  50 <= wc <   80: return -3
    elif wc < 50:          return -12
    else:                  return -15


def _hook_score(first_sent: str) -> tuple[str, int]:
    s = first_sent.lower()
    if re.search(r"\bnever\b.*\b(thought|believed|imagined|expected)", s):
        return "never_narrative", 15
    if re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", s):
        return "specific_time", 12
    if s.startswith(('"', "'")):
        return "quote_hook", 10
    if re.search(r"\b(stop|start|quit|avoid|never)\s+(doing|using|saying|thinking)", s):
        return "contrarian", 7
    if re.search(r"\bi\s+used\s+to\s+(think|believe|assume)", s):
        return "belief_transformation", 6
    if re.search(r"\b(it'?s official|today|finally|announcing)\b", s):
        return "announcement", 6
    if re.search(r"\beveryone('s| is)\b", s):
        return "everyone_pattern", 5
    if re.search(r"\bjust\s+(realized|learned|discovered|noticed)", s):
        return "realization", 5
    if re.search(r"\b(hours? ago|last (week|month)|yesterday|recently)\b", s):
        return "recency", 4
    return "no_hook", 0


def _power_patterns(text: str) -> tuple[dict, int, int]:
    tl = text.lower()
    flags = {
        "underdog":              int(bool(re.search(r"\b(immigrant|refugee|struggle|overcome|against all odds|bootstrapped|from nothing)\b", tl))),
        "transformation":        int(bool(re.search(r"\b(used to.*now|transformed|changed my life|went from.*to)\b", tl))),
        "cta_question":          int(bool(re.search(r"\b(what do you think|agree or disagree|comment below|share your|thoughts)\?", tl))),
        "hidden_truth":          int(bool(re.search(r"\b(nobody (posts|talks|mentions)|no one (talks|discusses)|hidden truth)\b", tl))),
        "vulnerability":         int(bool(re.search(r"\b(failed|mistake|wrong|scared|afraid|vulnerable|honest|transparent|real talk)\b", tl))),
        "family":                int(bool(re.search(r"\b(daughter|son|kids|children|parent|mom|dad|family|wife|husband)\b", tl))),
        "specific_time_content": int(bool(re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b|\b(morning|afternoon|evening|midnight)\b", tl))),
        "specific_numbers":      int(bool(re.search(r"\b\d+%|\$\d+|\d+x|\d+k\b", tl))),
        "adversity_learning":    int(bool(re.search(r"\b(learned|lesson|taught me|experience|failure|setback)\b", tl))),
        "value_promise":         int(bool(re.search(r"\b(here'?s how|step[s]?:|tip[s]?:|\d+ ways|playbook)\b", tl))),
        "list_format":           int(bool(re.search(r"^\s*[-\u2022*\d][\s.)].+", text, re.MULTILINE))),
        "contrast":              int(bool(re.search(r"\b(but|however|yet|while|instead|rather)\b", tl))),
        "aspirational":          int(bool(re.search(r"\b(dream|vision|future|inspire|achieve|success|goal)\b", tl))),
        "direct_address":        int(bool(re.search(r"\b(you|your|you're|you've)\b", tl))),
        "personal_story":        int(bool(re.search(r"\b(i |my |me |myself )\b", tl))),
    }
    count = sum(flags.values())
    score = sum(_PP_WEIGHTS[k] * v for k, v in flags.items())
    return flags, count, score


def _promo_score(text: str) -> int:
    _HIGH = ["our product", "we built", "we launched", "buy now", "sign up",
             "register now", "limited time", "special offer", "discount"]
    _MED  = ["product", "service", "solution", "demo", "launch", "release",
             "introducing", "platform"]
    tl = text.lower()
    return (sum(2 for kw in _HIGH if kw in tl) +
            sum(1 for kw in _MED if re.search(r"\b" + kw + r"\b", tl)))


def extract_features(post_text: str, followers: int, post_title: str, loo_mean: float) -> dict:
    """Extract ALL features needed for both models at inference time."""
    log_f = np.log1p(followers)

    words = post_text.split()
    wc = len(words)
    sents = re.split(r"[.!?]+", post_text)
    sc = max(1, len(sents))
    lines = post_text.strip().split("\n")
    lc = max(1, len(lines))

    first_line = lines[0] if lines else ""
    first_sent = sents[0].strip() if sents else ""
    hook_type, hook_s = _hook_score(first_sent)

    hashtags = re.findall(r"#\w+", post_text)
    num_ht = len(hashtags)
    ht_bucket = 0 if num_ht == 0 else (1 if num_ht <= 2 else (2 if num_ht <= 5 else (3 if num_ht <= 10 else 4)))

    urls = _URL_PAT.findall(post_text)
    num_links = len(urls)

    emoji_matches = _EMOJI_PAT.findall(post_text)
    emoji_total = sum(len(e) for e in emoji_matches)
    emoji_unique = len(set("".join(emoji_matches)))

    pp_flags, pp_count, pp_score = _power_patterns(post_text)
    promo = _promo_score(post_text)
    lscore = _length_score(wc)

    _pattern_density = 12 if pp_count >= 6 else (7 if pp_count >= 4 else (4 if pp_count == 3 else -7))
    _promo_pen = -12 if promo >= 6 else (-8 if promo >= 4 else (-4 if promo >= 2 else 0))
    link_penalty = -18 if num_links > 0 else 0
    low_effort = int(wc < 80 and num_links > 0 and pp_count < 2)

    base_score = max(0, min(100,
        50 + lscore + hook_s + pp_score + link_penalty +
        _pattern_density + _promo_pen + low_effort * -15
    ))

    has_vuln = pp_flags.get("vulnerability", 0)
    has_cta = int(bool(re.search(
        r"\b(share|comment|follow|like|repost|what do you think|thoughts\?|agree\?)\b",
        post_text, re.I
    )))

    lb = 0 if wc <= 50 else (1 if wc <= 150 else (2 if wc <= 300 else (3 if wc <= 500 else 4)))

    feats = {
        # author
        "log_followers":            log_f,
        "time_spent":               0.0,
        "author_loo_log_mean":      loo_mean,
        "author_post_count":        1,
        # media
        "is_post":                  1,
        "is_article":               0,
        "is_repost":                0,
        "has_video":                0,
        "has_carousel":             0,
        "has_image":                0,
        "has_media":                0,
        "media_score":              0,
        # hashtags
        "num_hashtags":             num_ht,
        "has_hashtags":             int(num_ht > 0),
        "hashtag_bucket":           float(ht_bucket),
        # links
        "num_content_links":        num_links,
        "has_external_link":        int(num_links > 0),
        "link_penalty_score":       link_penalty,
        # text stats
        "char_count":               len(post_text),
        "word_count":               wc,
        "sentence_count":           sc,
        "line_count":               lc,
        "line_break_count":         post_text.count("\n"),
        "avg_word_length":          np.mean([len(w) for w in words]) if words else 0.0,
        "avg_sentence_length":      wc / sc,
        "post_density":             wc / lc,
        "is_long_form":             int(wc > 500),
        "first_line_words":         len(first_line.split()),
        "first_line_short":         int(len(first_line.split()) <= 12),
        "num_exclamations":         post_text.count("!"),
        "num_questions":            post_text.count("?"),
        "has_exclamation":          int(post_text.count("!") > 0),
        "has_question":             int(post_text.count("?") > 0),
        "num_caps_words":           sum(1 for w in words if len(w) > 1 and w.isupper()),
        "num_numbers":              len(re.findall(r"\b\d+\b", post_text)),
        "has_numbers":              int(bool(re.findall(r"\b\d+\b", post_text))),
        "bullet_count":             sum(1 for l in lines if re.match(r"^\s*[-\u2022*]\s", l)),
        "has_bullets":              int(any(re.match(r"^\s*[-\u2022*]\s", l) for l in lines)),
        "has_numbered_list":        int(bool(re.search(r"^\s*\d+[.)]\s", post_text, re.MULTILINE))),
        # style
        "style_quote_marks":        post_text.count('"') + post_text.count("'"),
        "style_has_quotes":         int((post_text.count('"') + post_text.count("'")) >= 2),
        "style_parentheses":        post_text.count("(") + post_text.count(")"),
        "style_has_parentheses":    int((post_text.count("(") + post_text.count(")")) >= 2),
        "mention_count":            len(re.findall(r"@\w+", post_text)),
        "url_in_content":           num_links,
        # emojis
        "emoji_count":              emoji_total,
        "unique_emoji_count":       emoji_unique,
        "has_emoji":                int(emoji_total > 0),
        # diversity + lengths
        "lexical_diversity":        len(set(post_text.lower().split())) / max(1, wc),
        "length_bucket":            float(lb),
        "length_score":             lscore,
        # hooks
        "hook_score":               hook_s,
        "has_announcement_hook":    int(hook_type == "announcement"),
        "has_recency_hook":         int(hook_type == "recency"),
        "has_personal_hook":        int(bool(re.match(r"^(I |After |When |Today |Yesterday |In \d)", post_text.strip()))),
        "starts_with_number":       int(bool(re.match(r"^\s*\d", post_text.strip()))),
        "has_announcement":         int(bool(re.search(r"\b(excited|thrilled|proud|happy|delighted|announcing|announced)\b", post_text, re.I))),
        "has_question_hook":        int(post_text.strip().startswith(("What ", "How ", "Why ", "Who ", "Is ", "Are ", "Do ", "Can "))),
        "has_career_content":       int(bool(re.search(r"\b(job|career|hired|fired|role|position|company|startup|founder|ceo|promotion)\b", post_text, re.I))),
        "has_ai_tech":              int(bool(re.search(r"\b(AI|GPT|LLM|machine learning|deep learning|neural|ChatGPT|artificial intelligence)\b", post_text, re.I))),
        "has_cta":                  has_cta,
        # power patterns
        "power_pattern_count":      pp_count,
        "power_pattern_score":      pp_score,
        "has_underdog":             pp_flags.get("underdog", 0),
        "has_transformation":       pp_flags.get("transformation", 0),
        "has_cta_question":         pp_flags.get("cta_question", 0),
        "has_hidden_truth":         pp_flags.get("hidden_truth", 0),
        "has_vulnerability":        pp_flags.get("vulnerability", 0),
        "has_family":               pp_flags.get("family", 0),
        "has_specific_time_content":pp_flags.get("specific_time_content", 0),
        "has_specific_numbers":     pp_flags.get("specific_numbers", 0),
        "has_adversity_learning":   pp_flags.get("adversity_learning", 0),
        "has_value_promise":        pp_flags.get("value_promise", 0),
        "has_list_format":          pp_flags.get("list_format", 0),
        "has_contrast":             pp_flags.get("contrast", 0),
        "has_aspirational":         pp_flags.get("aspirational", 0),
        "has_direct_address":       pp_flags.get("direct_address", 0),
        "has_personal_story":       pp_flags.get("personal_story", 0),
        "personal_story_score":     int(bool(re.match(r"^(I |After |When |Today |Yesterday |In \d)", post_text.strip()))) + pp_flags.get("vulnerability", 0) + int(bool(re.search(r"\b(excited|thrilled|proud|happy|delighted|announcing|announced)\b", post_text, re.I))),
        # promotional
        "promotional_score":        promo,
        "is_promotional":           int(promo >= 2),
        "is_heavy_promo":           int(promo >= 6),
        "is_low_effort_link":       low_effort,
        # composite
        "base_score":               base_score,
        # topics
        **{f"topic_{t}": int(bool(re.search(p, post_text, re.I))) for t, p in _TOPICS.items()},
        "topic_count":              sum(1 for t, p in _TOPICS.items() if re.search(p, post_text, re.I)),
        "is_multi_topic":           int(sum(1 for t, p in _TOPICS.items() if re.search(p, post_text, re.I)) > 1),
        # headline
        "headline_word_count":      len(post_title.split()) if post_title else 0,
        "headline_has_emoji":       int(bool(_EMOJI_PAT.search(post_title))),
        # interactions
        "log_followers_x_is_post":  log_f * 1,
        "log_followers_x_has_vuln": log_f * has_vuln,
        "log_followers_x_has_cta":  log_f * has_cta,
        "log_followers_x_personal": log_f * int(bool(re.match(r"^(I |After |When |Today |Yesterday |In \d)", post_text.strip()))),
        "loo_x_is_post":            loo_mean * 1,
        "loo_x_has_vuln":           loo_mean * has_vuln,
        "loo_x_word_count":         loo_mean * wc,
        "hook_x_power_score":       hook_s * pp_score,
        "loo_x_base_score":         loo_mean * base_score,
        "sentiment_x_base_score":   0.0,  # no VADER at inference — zero-filled
    }
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Model loading + prediction
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_models():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl_r = joblib.load(os.path.join(MODEL_DIR, "hgbr_reactions.pkl"))
        mdl_c = joblib.load(os.path.join(MODEL_DIR, "hgbr_comments.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_names_reactions.json")) as f:
        feat_r = json.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names_comments.json")) as f:
        feat_c = json.load(f)
    with open(os.path.join(MODEL_DIR, "loo_stats_reactions.json")) as f:
        loo_r = json.load(f)
    with open(os.path.join(MODEL_DIR, "loo_stats_comments.json")) as f:
        loo_c = json.load(f)
    return mdl_r, mdl_c, feat_r, feat_c, loo_r, loo_c


def predict_engagement(post_text: str, followers: int, post_title: str) -> dict:
    """Return predicted reactions and comments for a post."""
    mdl_r, mdl_c, feat_r, feat_c, loo_r, loo_c = _load_models()

    global_loo_r = loo_r["global_log_mean"]
    global_loo_c = loo_c["global_log_mean"]

    feats_r = extract_features(post_text, followers, post_title, global_loo_r)
    feats_c = extract_features(post_text, followers, post_title, global_loo_c)

    row_r = np.array([[feats_r.get(f, 0.0) for f in feat_r]])
    row_c = np.array([[feats_c.get(f, 0.0) for f in feat_c]])

    pred_r = float(np.expm1(mdl_r.predict(row_r)[0]))
    pred_c = float(np.expm1(mdl_c.predict(row_c)[0]))

    return {
        "reactions": max(0, round(pred_r)),
        "comments":  max(0, round(pred_c)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — Visual generation (04_visual_generation/service/visual_service.py)
# ══════════════════════════════════════════════════════════════════════════════

VISUAL_SERVICE_SCRIPT = os.path.join(SCRIPT_DIR, "04_visual_generation", "service", "visual_service.py")
VISUAL_SERVICE_DIR    = os.path.join(SCRIPT_DIR, "04_visual_generation", "service")


def generate_visual(post_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Call visual_service.py with the post text.
    Returns (image_path, error_message). One will be None.
    """
    result = subprocess.run(
        [sys.executable, VISUAL_SERVICE_SCRIPT, post_text],
        capture_output=True,
        text=True,
        cwd=VISUAL_SERVICE_DIR,
        timeout=120,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if not stdout:
        return None, f"Visual service produced no output.\n{stderr}"

    # The service prints one JSON line to stdout
    try:
        last_json_line = next(
            (l for l in reversed(stdout.splitlines()) if l.strip().startswith("{")),
            None,
        )
        if not last_json_line:
            return None, f"No JSON found in output:\n{stdout}"
        data = json.loads(last_json_line)
    except Exception as e:
        return None, f"Could not parse visual service output: {e}\n{stdout}"

    if "error" in data:
        return None, data["error"]

    path = data.get("path")
    if path and os.path.exists(path):
        return path, None

    return None, f"Image path not found: {path}"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — Session-state helpers
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "step": 1,
        "profile_text": "",
        "followers": 1000,
        "trending_df": None,
        "llm_output": "",
        "parsed_titles": [],
        "chosen_topic": None,
        "chosen_title": None,
        "selected_title_idx": None,   # which title card is highlighted in step 3
        "post_variants": [],
        "final_post": None,
        "visual_path": None,
        "visual_error": None,
        # loading flags — set True before st.rerun() to disable all UI on next pass
        "_step2_loading": False,
        "_step3_loading": False,
        "_step4_selected": None,      # variant index selected; disables other buttons
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _go(step: int):
    st.session_state["step"] = step


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _inject_css():
    st.markdown("""
    <style>
    /* ── Background ─────────────────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(160deg, #0d1117 0%, #161b2e 45%, #1a1040 100%);
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stBottom"] { background: transparent; }

    /* ── Global text ─────────────────────────────────────────────────────────── */
    html, body, .stApp, .stMarkdown, p, span, label, div {
        color: #e2e8f0;
    }
    h1 { color: #f1f5f9 !important; font-weight: 700; }
    h2, h3 { color: #cbd5e1 !important; }
    .stCaption, .stCaption p { color: #64748b !important; }

    /* ── Title area ──────────────────────────────────────────────────────────── */
    [data-testid="stVerticalBlock"] > div:first-child h1 {
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Bordered containers / cards ─────────────────────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(8px);
        transition: border-color 0.2s;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(129,140,248,0.35) !important;
    }
    /* Selected card (step 3) — override hover so it stays blue */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(button[kind="primary"]) {
        border: 2px solid #818cf8 !important;
        background: rgba(129,140,248,0.08) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────────────── */
    .stButton > button {
        background: rgba(255,255,255,0.06);
        color: #cbd5e1;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.18s ease;
    }
    .stButton > button:hover:not(:disabled) {
        background: rgba(255,255,255,0.11);
        border-color: rgba(255,255,255,0.25);
        color: #f1f5f9;
        transform: translateY(-1px);
    }
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #fff !important;
        border: none;
        box-shadow: 0 4px 15px rgba(79,70,229,0.35);
    }
    .stButton > button[kind="primary"]:hover:not(:disabled) {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
        box-shadow: 0 6px 20px rgba(79,70,229,0.5);
        transform: translateY(-1px);
    }
    .stButton > button:disabled {
        opacity: 0.38 !important;
        cursor: not-allowed !important;
        transform: none !important;
    }

    /* ── Text areas & inputs ─────────────────────────────────────────────────── */
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.05) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
    }
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(129,140,248,0.2) !important;
    }

    /* ── Radio buttons ───────────────────────────────────────────────────────── */
    [data-testid="stRadio"] label { color: #cbd5e1 !important; }
    [data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }

    /* ── Dataframe ───────────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"], .stDataFrame iframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Metrics ─────────────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { color: #a5b4fc !important; }
    [data-testid="stMetricLabel"] { color: #64748b !important; }

    /* ── Alerts (success / warning / error) ─────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border-left-width: 4px !important;
    }
    div[data-baseweb="notification"][data-kind="positive"] {
        background: rgba(16,185,129,0.12) !important;
        border-left-color: #10b981 !important;
    }
    div[data-baseweb="notification"][data-kind="warning"] {
        background: rgba(245,158,11,0.12) !important;
        border-left-color: #f59e0b !important;
    }
    div[data-baseweb="notification"][data-kind="negative"] {
        background: rgba(239,68,68,0.12) !important;
        border-left-color: #ef4444 !important;
    }

    /* ── Code blocks ─────────────────────────────────────────────────────────── */
    [data-testid="stCode"] {
        background: rgba(0,0,0,0.35) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
    }

    /* ── Expanders ───────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
    }

    /* ── Divider ─────────────────────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Spinner text ────────────────────────────────────────────────────────── */
    [data-testid="stSpinner"] p { color: #94a3b8 !important; }

    /* ── Scrollbar ───────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.15);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }
    </style>
    """, unsafe_allow_html=True)


def _progress_bar():
    steps = ["Profile", "Topics", "Post Title", "Posts", "Done"]
    cur = st.session_state["step"]
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps), start=1):
        if i < cur:
            col.markdown(f"<div style='text-align:center;color:#10b981;font-weight:600;font-size:13px'>✓ {label}</div>", unsafe_allow_html=True)
        elif i == cur:
            col.markdown(f"<div style='text-align:center;color:#818cf8;font-weight:700;font-size:13px'>▶ {label}</div>", unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align:center;color:#334155;font-size:13px'>{label}</div>", unsafe_allow_html=True)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION G — Steps
# ══════════════════════════════════════════════════════════════════════════════

def step1_profile():
    st.subheader("Step 1 — Your LinkedIn Profile")
    st.caption("Paste your LinkedIn bio so we can identify trending topics relevant to your expertise.")

    profile = st.text_area(
        "LinkedIn Bio",
        value=st.session_state["profile_text"],
        height=200,
        placeholder="Paste your LinkedIn bio here…",
        key="_bio_input",
    )
    followers = st.number_input(
        "LinkedIn Followers",
        min_value=0,
        max_value=10_000_000,
        value=st.session_state["followers"],
        step=100,
        help="Used by the engagement prediction model",
        key="_followers_input",
    )

    if st.button("Analyze Trends →", type="primary"):
        if not profile.strip():
            st.error("Please paste your LinkedIn bio before continuing.")
            return
        st.session_state["profile_text"] = profile.strip()
        st.session_state["followers"] = int(followers)
        with st.spinner("Extracting topics from your bio and fetching Google Trends data…"):
            try:
                df = run_trend_analysis(profile.strip())
                st.session_state["trending_df"] = df
                _go(2)
                st.rerun()
            except Exception as e:
                st.error(f"Trend analysis failed: {e}")


def step2_topics():
    st.subheader("Step 2 — Trending Topics")
    st.caption("These topics were extracted from your bio and scored against Google Trends. Pick one to focus your post.")

    # ── Loading pass: spinner only, no interactive UI ─────────────────────────
    if st.session_state["_step2_loading"]:
        with st.spinner("LLM is picking the best search signals and writing post titles…"):
            try:
                output = run_topic_selection(
                    st.session_state["trending_df"],
                    st.session_state["profile_text"],
                    chosen_topic=st.session_state["chosen_topic"],
                )
                st.session_state["llm_output"] = output
                st.session_state["parsed_titles"] = parse_post_titles(output)
            except Exception as e:
                st.session_state["_step2_loading"] = False
                st.error(f"Topic selection failed: {e}")
                return
        st.session_state["_step2_loading"] = False
        _go(3); st.rerun()
        return

    # ── Normal UI ─────────────────────────────────────────────────────────────
    df: pd.DataFrame = st.session_state["trending_df"]
    top5 = df.head(5).reset_index(drop=True)

    display_df = pd.DataFrame({
        "#": range(1, len(top5) + 1),
        "Topic": top5["topic"],
        "Trend Score": top5["trend_score"].apply(lambda x: f"{x:.1f}"),
        "Top Searches": top5["top_queries"].apply(
            lambda q: ", ".join(x["query"] for x in q[:4]) if q else "—"
        ),
        "Rising Searches": top5["rising_queries"].apply(
            lambda q: ", ".join(
                "{} ({})".format(x["query"], "Breakout" if str(x["value"]) == "Breakout" else "+{}%".format(x["value"]))
                for x in q[:3]
            ) if q else "—"
        ),
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("**Choose a topic — or let the AI pick the best one:**")
    topic_options = ["🤖 Let AI pick the best topic"] + top5["topic"].tolist()
    chosen = st.radio("Topic", topic_options, label_visibility="collapsed")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            _go(1); st.rerun()
    with col2:
        if st.button("Generate Post Titles →", type="primary"):
            chosen_topic = None if chosen.startswith("🤖") else chosen
            st.session_state["chosen_topic"] = chosen_topic
            st.session_state["_step2_loading"] = True
            st.rerun()


def step3_post_title():
    st.subheader("Step 3 — Choose a Post Title")
    st.caption("The AI analysed trending search signals and drafted these titles. Pick the one that resonates most.")

    # ── Loading pass: spinner only, no interactive UI ─────────────────────────
    if st.session_state["_step3_loading"]:
        with st.spinner("Generating 3 post variants…"):
            try:
                variants = generate_post_variants(
                    st.session_state["chosen_title"],
                    st.session_state["profile_text"],
                )
                st.session_state["post_variants"] = variants
            except Exception as e:
                st.session_state["_step3_loading"] = False
                st.session_state["selected_title_idx"] = None
                st.error(f"Post generation failed: {e}")
                return
        st.session_state["_step3_loading"] = False
        st.session_state["selected_title_idx"] = None
        _go(4); st.rerun()
        return

    titles = st.session_state["parsed_titles"]
    if not titles:
        st.warning("Could not parse post titles from the LLM output.")
        with st.expander("Raw LLM output"):
            st.text(st.session_state["llm_output"])
        if st.button("← Back"):
            _go(2); st.rerun()
        return

    selected_idx = st.session_state.get("selected_title_idx")

    # Inject CSS: selected card gets a blue border + tinted background
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"]:has(button[kind="primary"]) {
        border: 2px solid #1E88E5 !important;
        background: #f0f7ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for i, t in enumerate(titles):
        is_selected = selected_idx == i
        with st.container(border=True):
            st.markdown(f"**Option {i + 1}**")
            if t["signal"]:
                st.caption(f"Search signal: *{t['signal']}*")
            st.markdown(f"### {t['title']}")

            if is_selected:
                st.button("✓ Selected", key=f"title_{i}", type="primary", disabled=True)
            else:
                if st.button(
                    f"Select Option {i + 1}",
                    key=f"title_{i}",
                    type="secondary",
                    disabled=selected_idx is not None,
                ):
                    st.session_state["selected_title_idx"] = i
                    st.rerun()

    # Reasoning expander
    reasoning_match = re.search(r"REASONING:(.*)", st.session_state["llm_output"], re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        with st.expander("Why these titles?"):
            st.markdown(reasoning_match.group(1).strip())

    st.divider()
    col_back, col_gap, col_next = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Back", disabled=selected_idx is not None):
            st.session_state["selected_title_idx"] = None
            _go(2); st.rerun()
    with col_next:
        if st.button(
            "Generate Posts →",
            type="primary",
            disabled=selected_idx is None,
        ):
            st.session_state["chosen_title"] = titles[selected_idx]["title"]
            st.session_state["_step3_loading"] = True
            st.rerun()


def step4_posts():
    st.subheader("Step 4 — Compare Posts & Engagement Predictions")
    st.caption(
        f"**Post title:** {st.session_state['chosen_title']}\n\n"
        "Three hook styles generated. Engagement is predicted by our HGBR model trained on real LinkedIn data."
    )

    variants = st.session_state["post_variants"]
    followers = st.session_state["followers"]
    post_title = st.session_state["chosen_title"]
    is_loading = "predictions" not in st.session_state

    # Back button at the top — disabled while predictions are loading
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back", disabled=is_loading):
            st.session_state.pop("predictions", None)
            st.session_state["_step4_selected"] = None
            _go(3); st.rerun()

    if is_loading:
        with st.spinner("Running engagement prediction model…"):
            preds = [predict_engagement(v["post_text"], followers, post_title) for v in variants]
            st.session_state["predictions"] = preds
        st.rerun()   # re-render with enabled back button + full content
        return

    preds = st.session_state["predictions"]

    # Summary table
    summary = pd.DataFrame([
        {
            "Variant": f"Variant {i+1} — {v['hook_style']}",
            "Words": v["word_count"],
            "Predicted Reactions": preds[i]["reactions"],
            "Predicted Comments": preds[i]["comments"],
        }
        for i, v in enumerate(variants)
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    selected_variant = st.session_state.get("_step4_selected")

    # Individual cards
    for i, (v, p) in enumerate(zip(variants, preds), start=1):
        with st.container(border=True):
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.markdown(f"**Variant {i} — {v['hook_style']}**")
                st.text_area(
                    f"Post text {i}",
                    value=v["post_text"],
                    height=250,
                    key=f"post_text_{i}",
                    label_visibility="collapsed",
                )
            with col_right:
                st.metric("Predicted Reactions", f"{p['reactions']:,}")
                st.metric("Predicted Comments", f"{p['comments']:,}")
                st.caption(f"{v['word_count']} words")
                is_this_selected = selected_variant == i
                btn_label = "✓ Selected" if is_this_selected else "Select Variant & Generate Image"
                btn_disabled = selected_variant is not None and not is_this_selected
                if st.button(btn_label, type="primary", key=f"select_{i}", disabled=btn_disabled):
                    if not is_this_selected:
                        st.session_state["_step4_selected"] = i
                        st.session_state["final_post"] = v["post_text"]
                        st.session_state.pop("predictions", None)
                        st.session_state["visual_path"] = None
                        st.session_state["visual_error"] = None
                        _go(5); st.rerun()


def step5_final():
    st.subheader("Step 5 — Your Final Post")
    st.success("Your post is ready! Copy it and paste it directly into LinkedIn.")

    # ── Visual generation pass: spinner only, then rerun ──────────────────────
    if st.session_state["visual_path"] is None and st.session_state["visual_error"] is None:
        with st.spinner("Generating visual for your post…"):
            try:
                path, err = generate_visual(st.session_state["final_post"])
            except Exception as e:
                path, err = None, str(e)
            st.session_state["visual_path"] = path
            st.session_state["visual_error"] = err
        st.rerun()
        return

    # ── Display pass ──────────────────────────────────────────────────────────
    st.markdown(f"**{st.session_state['chosen_title']}**")
    st.write(st.session_state["final_post"])

    st.divider()

    if st.session_state["visual_path"] and os.path.exists(st.session_state["visual_path"]):
        _, col_img, _ = st.columns([1, 2, 1])
        with col_img:
            st.image(st.session_state["visual_path"], use_container_width=True)
    elif st.session_state["visual_error"]:
        st.warning("Image generation failed.")
        with st.expander("Error details"):
            st.text(st.session_state["visual_error"])

    # ── Actions ───────────────────────────────────────────────────────────────
    st.divider()
    col_regen, col_copy, col_start = st.columns([2, 2, 2])

    with col_regen:
        if st.button("🔁 Regenerate image"):
            st.session_state["visual_path"] = None
            st.session_state["visual_error"] = None
            st.rerun()

    with col_copy:
        st.caption("Copy post text:")
        st.code(st.session_state["final_post"], language=None)

    with col_start:
        if st.button("🔄 Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION H — Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _inject_css()

    st.title("🚀 TrendPilot — LinkedIn Post Creator")
    st.caption("Trend-driven post generation with engagement prediction")

    _init_state()
    _progress_bar()

    step = st.session_state["step"]
    if   step == 1: step1_profile()
    elif step == 2: step2_topics()
    elif step == 3: step3_post_title()
    elif step == 4: step4_posts()
    elif step == 5: step5_final()


if __name__ == "__main__":
    main()
