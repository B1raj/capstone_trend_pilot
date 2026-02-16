# Trend Finder Module — Progress Report

**Project:** TrendPilot — AI-Powered LinkedIn Content Strategy Tool
**Module:** Trend Identification & Post Topic Recommendation
**Date:** February 16, 2026
**Author:** Biraj Mishra

---

## 1. Objective

Build a system that takes a LinkedIn professional's profile bio and automatically identifies **timely, specific, trending topics** relevant to their expertise — then recommends a concrete LinkedIn post angle backed by real-time search data from Google Trends.

---

## 2. Iteration Summary

| Iteration | Approach | Topic Extraction | Trend Source | Outcome |
|-----------|----------|-----------------|--------------|---------|
| 1 | BuzzSumo + News API | Keyword matching | News articles | Generic, noisy, low relevance |
| 2 | Google Trends + spaCy NLP | Named Entity Recognition | Google Trends `related_queries` | Ambiguous entities, wrong context |
| 3 | Google Trends + GPT-4o Prompt Engineering | LLM-driven extraction | Google Trends `interest_over_time` + `related_queries` (top + rising) | Specific, relevant, actionable topics |
| 4 | Iteration 3 + Interactive Loop + Version Safety | Same as Iter 3 | Same as Iter 3 | User control, factual accuracy, re-selection |

---

## 3. Iteration Details

### Iteration 1: BuzzSumo + News API

**Approach:**
Used BuzzSumo API to find popular content and News API to find trending articles. Keywords were extracted from the user's LinkedIn bio using simple keyword matching, then matched against news headlines and BuzzSumo trending content.

**Problems Encountered:**
- **Generic topics:** Results were dominated by broad news stories (e.g., "AI is transforming business", "Cloud computing market growth") that had no specific angle for a LinkedIn post.
- **Noisy results:** News API returned articles about politics, sports, entertainment — not filtered to the user's professional domain.
- **Low relevance:** No mechanism to score how relevant a trending topic was to the user's specific skill set.

**Example Output:**
```
Topics found:
- "AI Revolution in Healthcare" (news article, not user's domain)
- "Cloud Computing Market Worth $1.5T by 2030" (generic market report)
- "Top 10 Programming Languages 2026" (listicle, not actionable)
```

**Decision:** Abandon news-based approach. Shift to Google Trends for real-time search interest data that better reflects what professionals are actually searching for.

---

### Iteration 2: Google Trends + spaCy NLP

**File:** `trend_finder.py` (596 lines)

**Approach:**
Used spaCy's `en_core_web_sm` NLP model to extract named entities (ORG, PRODUCT, PROPN, etc.) from the user's LinkedIn bio, then queried Google Trends `related_queries` API for each extracted entity.

**Key Components:**
- **Entity extraction** using spaCy NER with 11 entity labels (PERSON, ORG, GPE, PRODUCT, etc.)
- **Proper noun extraction** for tokens tagged as `PROPN`
- **Noun phrase extraction** for multi-word technical terms
- **TrendCache class** with 7-day TTL, composite cache keys (`keyword|geo|timeframe`)
- **Retry logic** with exponential backoff for Google Trends rate limiting (429 errors)

**Entity Extraction Prompt (spaCy NER — no prompt, rule-based):**
```python
# spaCy extracts entities by label type:
ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "NORP", "FAC", "LOC"
}

# Also extracts proper nouns (PROPN) and capitalized noun phrases
for token in doc:
    if token.pos_ == "PROPN":
        # extracted as entity
```

**Problems Encountered:**

1. **Ambiguous entities — the "Cloud" problem:**
   spaCy has no domain awareness. When a bio mentioned "cloud architecture" or "Spring Cloud", spaCy extracted **"Cloud"** as a generic proper noun. Google Trends then returned results for **cloud weather** — not cloud computing:
   ```
   Entity extracted: "Cloud" (PROPN)
   Google Trends top queries:
     - "cloud weather today" (value: 100)
     - "cloud types" (value: 65)
     - "cumulus cloud" (value: 37)
   ```

2. **"Spring" problem:**
   Similarly, "Spring" from "Spring Cloud" or "Spring Boot" was extracted standalone. Google Trends returned:
   ```
   Entity extracted: "Spring" (PROPN)
   Google Trends top queries:
     - "spring 2026" (value: 100)
     - "spring fever" (value: 74)
     - "hot spring" (value: 49)
   Rising queries:
     - "spring fever episodes" (+51500%)
     - "best places to visit in spring" (+44000%)
   ```
   Completely unrelated to the Spring Framework.

3. **"Kong" problem:**
   The API gateway tool "Kong" was extracted correctly, but Google Trends returned:
   ```
   Entity extracted: "Kong" (ORG)
   Google Trends top queries:
     - "king kong" (value: 100)
     - "hong kong china" (value: 54)
     - "donkey kong switch" (value: 17)
     - "godzilla x kong" (value: 14)
   ```
   The technology "Kong API Gateway" was completely lost in the noise of the word "Kong."

4. **Rate limiting:**
   The error log (`trend_finder.log`) shows aggressive 429 errors from Google Trends:
   ```
   2026-02-07 13:45:37 - WARNING - Retry 1/2 for 'java programming':
     Google returned a response with code 429
   2026-02-07 13:45:40 - WARNING - Retry 2/2 for 'java programming':
     Google returned a response with code 429
   2026-02-07 13:45:45 - ERROR - Could not fetch trends for 'java programming':
     Google returned a response with code 429
   ```
   Multiple keywords failed entirely due to rate limiting, even with exponential backoff.

5. **Multi-word term fragmentation:**
   spaCy often split compound technical terms. "API Gateways (Axway, Kong, Apigee)" was parsed into separate entities: "Axway", "Kong", "Apigee" — losing the context that these are API gateway tools.

**Mitigations attempted:**
- Added multi-word term prioritization (skip "Spring" if "Spring Cloud" exists)
- Added entity text cleaning to handle stray parentheses
- Added retry logic with exponential backoff

**Decision:** spaCy NER is not suitable for technical topic extraction from professional bios. The model lacks domain context and cannot distinguish "Cloud (technology)" from "Cloud (weather)." Replace NLP entity extraction with LLM-based prompt engineering.

---

### Iteration 3: Google Trends + GPT-4o Prompt Engineering

**File:** `trend_identification_v2.py` (initial version)

**Approach:**
Replaced spaCy NER entirely with a GPT-4o prompt that understands professional context. Added `interest_over_time` API for trend scoring alongside `related_queries` (top + rising) for search context. The LLM both extracts topics AND recommends a post angle.

**Key Changes from Iteration 2:**
1. **LLM topic extraction** instead of spaCy NER
2. **Trend scoring** via `interest_over_time` (mean score over 3 months)
3. **Top + rising queries** fed into the post recommendation prompt
4. **Two-stage LLM pipeline:** extract topics → score via Google Trends → recommend post

**Topic Extraction Prompt (GPT-4o):**
```
You are analyzing a LinkedIn professional bio.

Extract 8–12 specific, post-worthy technical topics suitable for LinkedIn.
Only include:
- Technologies
- Platforms
- Tools
- Frameworks
- Engineering practices

Exclude:
- Job titles
- Generic words (e.g., cloud, data, software)
- Soft skills
- Single generic nouns

Return the result as a comma-separated list.
```

**Why this prompt works:**
- Explicitly **excludes generic words** like "cloud", "data", "software" — solving the Iteration 2 ambiguity problem
- Asks for **"specific, post-worthy technical topics"** — the LLM understands that "Spring Cloud" is a technology while "Spring" is ambiguous
- Filters out **job titles and soft skills** that polluted Iteration 2 results

**Post Recommendation Prompt (GPT-4o):**
The second LLM call receives the top trending topics along with their Google Trends search data (top queries + rising queries) and the user's bio. Key prompt instructions:

```
Instructions:
1. Focus on SPECIFICITY over generality. Look at the rising/breakout
   searches — these reveal new releases, announcements, hot debates,
   or breakthroughs that people care about RIGHT NOW.
2. The post title MUST reference a specific thing (a new feature, release,
   tool, comparison, migration, controversy, or real-world use case) —
   NOT a generic thought-leadership angle.
3. If rising queries mention a specific product launch, version, integration,
   or comparison — use that as the post hook.
4. The post should feel like the author is reacting to something happening
   NOW, not writing a textbook intro.
```

**Example — Iteration 3 Output (with real cached data):**

For a LinkedIn bio mentioning API gateways, microservices, AWS, and integration architecture, the system produced:

```
Topic Extraction (LLM):
  AWS, OpenShift, Spring Cloud, Microservices, Apigee, Kong,
  REST APIs, Integration Architecture, API Gateways, Zuul, WSO2, Axway

Top Trending Topics (sorted by Google Trends score):
  1. AWS            (score: 55.2)
  2. OpenShift      (score: 51.7)
  3. Spring Cloud   (score: 50.8)
  4. Integration Architecture (score: 39.2)
  5. Microservices  (score: 31.2)

AWS — Top Searches:
  what is aws, aws news, aws ai, aws amazon, aws cloud
AWS — Rising Searches:
  aws devops agent (+1600%), aws transform custom (+1300%),
  aws security agent (+300%), aws news today (+250%)

Apigee — Rising Searches:
  apigee news (+250%), apigee pricing (+200%), kong (+190%)
```

**Improvements over Iteration 2:**
- "Cloud" no longer appears as a standalone topic — the LLM extracts "Spring Cloud" instead
- "Kong" still appears, but now alongside "Apigee" and "API Gateways" — giving proper context
- Rising queries like "aws devops agent (+1600%)" provide specific, timely hooks for posts
- Trend scores allow ranking topics by actual search interest

**Remaining Issues:**
- **Blank related queries:** Some topics (OpenShift, Spring Cloud, Zuul) returned empty `top_queries` and `rising_queries`. Root cause: the code called `build_payload()` twice per keyword — once for `interest_over_time`, again for `related_queries` — and the second call was rate-limited by Google.
- **LLM hallucinated version numbers:** When asked to recommend a post about OpenShift, the LLM produced:
  ```
  TOPIC: OpenShift
  POST TITLE: "OpenShift 4.12: What Its New GitOps Features Mean for
               Streamlined Cloud-Native Deployments"
  ```
  OpenShift 4.12 is outdated (current version is 4.21). The LLM fabricated the version number from its training data — it was not in the search data.

---

### Iteration 4: Interactive Loop + Version Safety + Query Fix

**File:** `trend_identification_v2.py` (current version, 272 lines)

**Changes Made:**

#### Fix 1: Blank Related Queries (Merged API Calls)

**Before (Iteration 3):** Two separate functions, two `build_payload` calls per keyword:
```python
def fetch_trend_score(keyword):
    pytrends.build_payload([keyword], ...)  # 1st API call
    df = pytrends.interest_over_time()
    return float(df[keyword].mean())

def fetch_related_queries(keyword):
    pytrends.build_payload([keyword], ...)  # 2nd API call (rate-limited!)
    related = pytrends.related_queries()
    ...
```

**After (Iteration 4):** Single function, one `build_payload` call:
```python
def fetch_trend_data(keyword):
    pytrends.build_payload([keyword], ...)  # Single API call

    # Trend score
    df = pytrends.interest_over_time()
    score = float(df[keyword].mean()) if (not df.empty ...) else 0.0

    # Related queries (same payload — no extra API call)
    related = pytrends.related_queries()
    ...
    return score, top_queries, rising_queries
```

This eliminated the rate-limiting issue that caused blank queries. Cache validation was also updated to re-fetch entries that were saved without query data:
```python
# Old: accepted stale entries without queries
if cache_entry and is_cache_valid(cache_entry["timestamp"]):

# New: re-fetches if queries are missing
if cache_entry and is_cache_valid(...) and cache_entry.get("top_queries") is not None:
```

#### Fix 2: Version Number Hallucination (Prompt Guard)

Added a critical rule to the post recommendation prompt:

```
CRITICAL RULE — Version numbers and factual accuracy:
- NEVER invent or guess version numbers, release names, or feature names.
  Your training data may be outdated.
- ONLY mention a specific version/release if it explicitly appears in the
  search query data above.
- If no version is mentioned in the search data, write the post title
  WITHOUT a version number. Use phrasing like "latest release",
  "new update", or focus on the concept/trend instead.
- NEVER fabricate announcements, launches, or features that are not
  evidenced by the search data.
```

Added explicit bad/good examples:
```
BAD: "OpenShift 4.12: What Its New GitOps Features Mean..."
     (version not from search data = hallucinated)

GOOD: "OpenShift's latest update doubles down on GitOps — here's
       what changed for cloud-native teams"
```

#### Fix 3: Interactive Topic Selection Loop

**Before (Iteration 3):** Program ran once, LLM auto-picked a topic, then exited.

**After (Iteration 4):** User sees top 5 topics and can:
- Enter `1-5` to choose a specific topic for post recommendation
- Enter `0` to let the AI auto-pick from the top 5
- Enter `q` to quit
- After each recommendation, the menu loops back — user can explore multiple topics without restarting

```
--- Top 5 Trending Topics (Profile-Driven) ---

  1. AWS  (trend score: 55.2)
     Top searches: what is aws, aws news, aws ai, aws amazon, aws cloud
     Rising searches: aws devops agent (+1600%), aws transform custom (+1300%)

  2. OpenShift  (trend score: 51.7)

  3. Spring Cloud  (trend score: 50.8)

  4. Integration Architecture  (trend score: 39.2)

  5. Microservices  (trend score: 31.2)
     Top searches: microservices architecture, microservices best practices
     Rising searches: microservices best practices (+350%)

  0. Let AI pick the best topic automatically
  q. Quit

Choose a topic number (1-5), 0 for AI pick, or q to quit:
```

#### Fix 4: Error Visibility

**Before:** Silent exception swallowing:
```python
except Exception:
    score = 0.0
```

**After:** Errors are printed:
```python
except Exception as e:
    print(f"  [warn] Failed to fetch trends for '{topic}': {e}")
    score = 0.0
```

#### Fix 5: JSON Serialization Safety

Added explicit type conversion for query values to prevent numpy int64 serialization errors:
```python
top_queries = [
    {"query": row["query"], "value": int(row["value"])}   # int()
    for _, row in top_df.head(10).iterrows()
]
rising_queries = [
    {"query": row["query"], "value": str(row["value"])}   # str()
    for _, row in rising_df.head(10).iterrows()
]
```

---

## 4. Architecture (Current — Iteration 4)

```
┌──────────────────┐
│  LinkedIn Bio    │
│  (User Input)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  GPT-4o Topic    │   Extract 8-12 specific technical topics
│  Extraction      │   (excludes generic words, job titles)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Google Trends   │   For each topic:
│  API (pytrends)  │   - interest_over_time → trend score
│                  │   - related_queries → top + rising searches
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Rank & Display  │   Sort by trend score, show top 5
│  Top 5 Topics    │   with search context
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  User Selection  │   Pick topic 1-5, or let AI choose
│  (Interactive)   │◄─── Loop back after each recommendation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  GPT-4o Post     │   Generate specific, timely post title
│  Recommendation  │   grounded in search data (no hallucinated versions)
└──────────────────┘
```

---

## 5. Key Learnings

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| Generic topics (Iter 1) | News APIs return broad content, not professional context | Switched to Google Trends for real search interest |
| "Cloud" = weather (Iter 2) | spaCy NER has no domain awareness | Replaced with GPT-4o prompt that excludes generic words |
| "Spring" = season (Iter 2) | Same — NLP model lacks tech context | LLM extracts "Spring Cloud" as a whole term |
| Blank related queries (Iter 3) | Double `build_payload()` caused rate limiting | Merged into single `fetch_trend_data()` call |
| Hallucinated version numbers (Iter 3) | LLM invents versions from stale training data | Added critical prompt rule: only cite versions from search data |
| One-shot exit (Iter 3) | Program terminated after one recommendation | Added interactive loop with re-selection |
| Silent failures (Iter 3) | `except Exception: pass` hid errors | Now prints `[warn]` with error message |

---

## 6. Files

| File | Lines | Purpose |
|------|-------|---------|
| `trend_finder.py` | 596 | Iteration 2 — spaCy NER + Google Trends (deprecated) |
| `trend_identification_v2.py` | 272 | Iteration 3 & 4 — GPT-4o + Google Trends (current) |
| `trend_cache.json` | ~785 | Shared cache file (7-day TTL) |
| `trend_finder.log` | 79 | Error logs from Iteration 2 (429 rate-limit errors) |
| `requirements.txt` | 3 | Dependencies: spacy, pytrends, pandas |

---

## 7. Next Steps

- Integrate trend finder output with the post generation agent (`streamlit_poc/agents/post_generator.py`)
- Add geographic customization (currently hardcoded to US)
- Explore caching related queries more aggressively to reduce API calls
- Add support for multi-profile batch analysis
