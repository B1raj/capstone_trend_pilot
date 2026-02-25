"""
Trend Finder: Analyze user paragraph and find Google Trends for mentioned entities.

This module extracts proper nouns and named entities from user input,
then queries Google Trends to find top and rising queries for each topic.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import spacy
from pytrends.request import TrendReq

# Configure file logger
logger = logging.getLogger("trend_finder")
logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler("trend_finder.log")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(_file_handler)


class TrendCache:
    """Local cache for Google Trends data to avoid repeated API calls."""

    def __init__(self, cache_file: str = "trend_cache.json", max_age_days: int = 7):
        """
        Initialize the trend cache.

        Args:
            cache_file: Path to the cache JSON file
            max_age_days: Maximum age in days before data is considered stale
        """
        self.cache_file = Path(cache_file)
        self.max_age_days = max_age_days
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error("Failed to load cache file '%s': %s", self.cache_file, e)
                return {"metadata": {"created": datetime.now().isoformat()}, "trends": {}}
        return {"metadata": {"created": datetime.now().isoformat()}, "trends": {}}

    def _save_cache(self):
        """Save cache to disk."""
        self._cache["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def _make_key(self, keyword: str, geo: str, timeframe: str) -> str:
        """Create a unique cache key."""
        return f"{keyword.lower()}|{geo}|{timeframe}"

    def _is_stale(self, fetch_date: str) -> bool:
        """Check if cached data is older than max_age_days."""
        try:
            fetched = datetime.fromisoformat(fetch_date)
            return datetime.now() - fetched > timedelta(days=self.max_age_days)
        except (ValueError, TypeError):
            return True

    def get(self, keyword: str, geo: str, timeframe: str) -> Optional[dict]:
        """
        Get cached trend data if available and not stale.

        Args:
            keyword: The search term
            geo: Geographic region code
            timeframe: Time range for trends

        Returns:
            Cached trend data or None if not found/stale
        """
        key = self._make_key(keyword, geo, timeframe)
        if key in self._cache["trends"]:
            entry = self._cache["trends"][key]
            if not self._is_stale(entry.get("fetch_date")):
                return entry.get("data")
            else:
                print(f"    (Cache expired for '{keyword}')")
        return None

    def set(self, keyword: str, geo: str, timeframe: str, data: dict):
        """
        Store trend data in cache.

        Args:
            keyword: The search term
            geo: Geographic region code
            timeframe: Time range for trends
            data: Trend data to cache
        """
        key = self._make_key(keyword, geo, timeframe)
        self._cache["trends"][key] = {
            "fetch_date": datetime.now().isoformat(),
            "keyword": keyword,
            "geo": geo,
            "timeframe": timeframe,
            "data": data
        }
        self._save_cache()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = len(self._cache["trends"])
        stale = sum(
            1 for entry in self._cache["trends"].values()
            if self._is_stale(entry.get("fetch_date"))
        )
        return {
            "total_entries": total,
            "fresh_entries": total - stale,
            "stale_entries": stale,
            "cache_file": str(self.cache_file)
        }

    def clear_stale(self):
        """Remove stale entries from cache."""
        keys_to_remove = [
            key for key, entry in self._cache["trends"].items()
            if self._is_stale(entry.get("fetch_date"))
        ]
        for key in keys_to_remove:
            del self._cache["trends"][key]
        self._save_cache()
        return len(keys_to_remove)


class TrendFinder:
    """Find Google Trends for entities mentioned in user text."""

    # Entity types to extract (proper nouns and named entities)
    ENTITY_LABELS = {
        "PERSON",      # People names
        "ORG",         # Organizations, companies
        "GPE",         # Countries, cities, states
        "PRODUCT",     # Products, objects, vehicles
        "EVENT",       # Named events
        "WORK_OF_ART", # Titles of books, songs, etc.
        "LAW",         # Named documents
        "LANGUAGE",    # Named languages
        "NORP",        # Nationalities, religious or political groups
        "FAC",         # Buildings, airports, highways, bridges
        "LOC",         # Non-GPE locations
    }

    def __init__(
        self, geo: str = "SG", timeframe: str = "today 1-m",
        cache_file: str = "trend_cache.json", cache_max_age_days: int = 7
    ):
        """
        Initialize TrendFinder.

        Args:
            geo: Geographic location code (default: SG for Singapore)
            timeframe: Time range for trends (default: today 1-m for past month)
            cache_file: Path to the local cache file
            cache_max_age_days: Days before cached data is considered stale
        """
        self.geo = geo
        self.timeframe = timeframe
        self.cache = TrendCache(cache_file=cache_file, max_age_days=cache_max_age_days)
        self._load_nlp_model()
        self._init_pytrends()

    def _init_pytrends(self):
        """Initialize pytrends with proper request settings."""
        import requests
        # Create session with proper headers
        self.pytrends = TrendReq(
            hl="en-US",
            tz=360,
            timeout=(10, 25),
            requests_args={
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
            }
        )

    def _load_nlp_model(self):
        """Load spaCy NLP model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found, downloading...")
            print("Downloading spaCy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text: str) -> list[dict]:
        """
        Extract proper nouns and named entities from text.

        Uses the exact text as provided without stemming to preserve
        the original form (e.g., 'design patterns' vs 'design pattern').

        Args:
            text: User input paragraph

        Returns:
            List of dictionaries with entity text and label
        """
        doc = self.nlp(text)
        entities = []
        seen = set()

        def clean_entity(text: str) -> str:
            """Clean entity text by removing stray parentheses and extra whitespace."""
            import re
            # Remove leading/trailing parentheses, commas, and whitespace
            text = re.sub(r'^[\(\)\s,]+|[\(\)\s,]+$', '', text)
            # Remove unbalanced parentheses at the end (like "API Gateways (Axway")
            if text.count('(') != text.count(')'):
                # Remove content from unbalanced opening paren
                text = re.sub(r'\s*\([^)]*$', '', text)
                # Or remove unbalanced closing paren
                text = re.sub(r'^[^(]*\)\s*', '', text)
            return text.strip()

        def is_valid_entity(text: str) -> bool:
            """Check if entity text is valid (not just punctuation or too short)."""
            if len(text) < 2:
                return False
            # Skip if mostly punctuation
            alpha_count = sum(1 for c in text if c.isalpha())
            return alpha_count >= 2

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in self.ENTITY_LABELS:
                # Use exact text as provided (no stemming/lemmatization)
                entity_text = clean_entity(ent.text)
                if is_valid_entity(entity_text) and entity_text.lower() not in seen:
                    seen.add(entity_text.lower())
                    entities.append({
                        "text": entity_text,
                        "label": ent.label_,
                        "original": entity_text  # Keep original form
                    })

        # Also extract proper nouns that might not be recognized as entities
        for token in doc:
            if token.pos_ == "PROPN":
                token_text = clean_entity(token.text)
                if is_valid_entity(token_text) and token_text.lower() not in seen:
                    seen.add(token_text.lower())
                    entities.append({
                        "text": token_text,
                        "label": "PROPN",
                        "original": token_text
                    })

        # Extract noun phrases that might be technical terms
        for chunk in doc.noun_chunks:
            chunk_text = clean_entity(chunk.text)
            # Check if it contains proper nouns or is capitalized
            if any(token.pos_ == "PROPN" for token in chunk):
                if is_valid_entity(chunk_text) and chunk_text.lower() not in seen:
                    seen.add(chunk_text.lower())
                    entities.append({
                        "text": chunk_text,
                        "label": "NOUN_PHRASE",
                        "original": chunk_text
                    })

        return entities

    def get_trends_for_keyword(
        self, keyword: str, top_n: int = 3, max_retries: int = 3,
        use_cache: bool = True
    ) -> Optional[dict]:
        """
        Get Google Trends data for a specific keyword.

        Args:
            keyword: The search term (used as-is without modification)
            top_n: Number of top queries to return
            max_retries: Maximum number of retry attempts on failure
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary with top queries and rising queries, or None if no data
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(keyword, self.geo, self.timeframe)
            if cached is not None:
                print(f"    (Using cached data)")
                # Apply top_n limit to cached data
                result = cached.copy()
                result["top_queries"] = result.get("top_queries", [])[:top_n]
                result["rising_queries"] = result.get("rising_queries", [])[:top_n]
                result["_from_cache"] = True
                return result

        # Fetch from API
        for attempt in range(max_retries):
            try:
                # Reinitialize pytrends connection on retry
                if attempt > 0:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    self._init_pytrends()

                # Build payload with exact keyword (no modification)
                self.pytrends.build_payload(
                    [keyword],
                    cat=0,
                    timeframe=self.timeframe,
                    geo=self.geo,
                    gprop=""
                )

                # Get related queries
                related = self.pytrends.related_queries()

                if keyword not in related or related[keyword] is None:
                    return None

                result = {
                    "keyword": keyword,
                    "top_queries": [],
                    "rising_queries": []
                }

                # Extract top queries (store more than top_n for cache)
                top_df = related[keyword].get("top")
                if top_df is not None and not top_df.empty:
                    top_queries = top_df.head(10).to_dict("records")  # Cache up to 10
                    result["top_queries"] = [
                        {"query": q["query"], "value": q["value"]}
                        for q in top_queries
                    ]

                # Extract rising queries (store more than top_n for cache)
                rising_df = related[keyword].get("rising")
                if rising_df is not None and not rising_df.empty:
                    rising_queries = rising_df.head(10).to_dict("records")  # Cache up to 10
                    result["rising_queries"] = [
                        {"query": q["query"], "value": q["value"]}
                        for q in rising_queries
                    ]

                # Cache the result
                if result["top_queries"] or result["rising_queries"]:
                    self.cache.set(keyword, self.geo, self.timeframe, result)

                # Return with top_n limit
                return_result = result.copy()
                return_result["top_queries"] = result["top_queries"][:top_n]
                return_result["rising_queries"] = result["rising_queries"][:top_n]
                return return_result

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning("Retry %d/%d for '%s': %s", attempt + 1, max_retries - 1, keyword, e)
                    print(f"    Retry {attempt + 1}/{max_retries - 1} for '{keyword}'...")
                else:
                    logger.error("Could not fetch trends for '%s': %s", keyword, e, exc_info=True)
                    print(f"  Warning: Could not fetch trends for '{keyword}': {e}")
                    return None

        return None

    def analyze(
        self, text: str, top_n: int = 3, delay: float = 1.0,
        prioritize_multi_word: bool = True
    ) -> dict:
        """
        Analyze text and find trends for all mentioned entities.

        Args:
            text: User input paragraph about themselves
            top_n: Number of top queries to collect per topic
            delay: Delay between API calls to avoid rate limiting
            prioritize_multi_word: If True, skip single words that are part of
                                   multi-word terms (e.g., skip 'Spring' if
                                   'Spring Cloud' exists)

        Returns:
            Dictionary containing entities and their associated trends
        """
        print("\n" + "=" * 60)
        print("TREND FINDER - Analyzing your profile")
        print("=" * 60)

        # Step 1: Extract entities
        print("\n[1/2] Extracting entities from text...")
        entities = self.extract_entities(text)

        # Prioritize multi-word terms by filtering out single words
        # that are contained in multi-word phrases
        if prioritize_multi_word and entities:
            multi_word = [e for e in entities if ' ' in e['text']]
            single_word = [e for e in entities if ' ' not in e['text']]

            # Filter single words that appear in multi-word terms
            filtered_single = []
            for sw in single_word:
                sw_lower = sw['text'].lower()
                is_part_of_multi = any(
                    sw_lower in mw['text'].lower()
                    for mw in multi_word
                )
                if not is_part_of_multi:
                    filtered_single.append(sw)

            # Combine: multi-word terms first, then remaining single words
            entities = multi_word + filtered_single

        if not entities:
            print("No entities found in the provided text.")
            return {"entities": [], "trends": []}

        print(f"Found {len(entities)} entities:")
        for ent in entities:
            print(f"  - {ent['text']} ({ent['label']})")

        # Step 2: Get trends for each entity
        print(f"\n[2/2] Fetching Google Trends (geo={self.geo}, timeframe={self.timeframe})...")
        trends_results = []

        for i, entity in enumerate(entities):
            keyword = entity["original"]  # Use original form without modification
            print(f"\n  [{i+1}/{len(entities)}] Searching trends for: '{keyword}'")

            trends = self.get_trends_for_keyword(keyword, top_n=top_n)

            from_cache = trends.get("_from_cache", False) if trends else False

            if trends and (trends["top_queries"] or trends["rising_queries"]):
                # Remove internal flag before storing result
                trends.pop("_from_cache", None)
                trends["entity_label"] = entity["label"]
                trends_results.append(trends)

                # Display results
                if trends["top_queries"]:
                    print(f"    Top queries:")
                    for q in trends["top_queries"]:
                        print(f"      - {q['query']}: {q['value']}")

                if trends["rising_queries"]:
                    print(f"    Rising queries:")
                    for q in trends["rising_queries"]:
                        value = q["value"]
                        if isinstance(value, str) and "%" in value:
                            print(f"      - {q['query']}: {value}")
                        else:
                            print(f"      - {q['query']}: +{value}%")
            else:
                print(f"    No trending data found.")

            # Rate limiting delay only when an actual API call was made
            if not from_cache and i < len(entities) - 1:
                time.sleep(delay + 1.5)  # Add extra buffer

        # Show cache statistics
        cache_stats = self.cache.get_stats()
        print(f"\n  Cache: {cache_stats['fresh_entries']} fresh, "
              f"{cache_stats['stale_entries']} stale entries")

        return {
            "entities": entities,
            "trends": trends_results,
            "settings": {
                "geo": self.geo,
                "timeframe": self.timeframe,
                "top_n": top_n
            },
            "cache_stats": cache_stats
        }

    def format_results(self, results: dict) -> str:
        """Format results as a readable string."""
        output = []
        output.append("\n" + "=" * 60)
        output.append("ANALYSIS RESULTS")
        output.append("=" * 60)

        if not results["trends"]:
            output.append("\nNo trending topics found for the mentioned entities.")
            return "\n".join(output)

        output.append(f"\nFound trends for {len(results['trends'])} topics:\n")

        for trend in results["trends"]:
            output.append(f"\n{'─' * 40}")
            output.append(f"Topic: {trend['keyword']} ({trend['entity_label']})")
            output.append(f"{'─' * 40}")

            if trend["top_queries"]:
                output.append("\n  Top Queries:")
                for q in trend["top_queries"]:
                    output.append(f"    • {q['query']}: {q['value']}")

            if trend["rising_queries"]:
                output.append("\n  Rising Queries (Breakout/Growth):")
                for q in trend["rising_queries"]:
                    value = q["value"]
                    if isinstance(value, str):
                        output.append(f"    ↗ {q['query']}: {value}")
                    else:
                        output.append(f"    ↗ {q['query']}: +{value}%")

        return "\n".join(output)


def main():
    """Main entry point for the Trend Finder application."""
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        cache = TrendCache()

        if cmd == "--cache-stats":
            stats = cache.get_stats()
            print("\nCache Statistics:")
            print(f"  File: {stats['cache_file']}")
            print(f"  Total entries: {stats['total_entries']}")
            print(f"  Fresh entries: {stats['fresh_entries']}")
            print(f"  Stale entries: {stats['stale_entries']}")
            return

        elif cmd == "--clear-cache":
            removed = cache.clear_stale()
            print(f"\nRemoved {removed} stale cache entries.")
            return

        elif cmd == "--help":
            print("\nTrend Finder - Usage:")
            print("  python trend_finder.py           Run interactive mode")
            print("  python trend_finder.py --cache-stats   Show cache statistics")
            print("  python trend_finder.py --clear-cache   Remove stale cache entries")
            print("  python trend_finder.py --help          Show this help")
            return

    print("\n" + "=" * 60)
    print("  TREND FINDER")
    print("  Discover trending topics from your profile")
    print("=" * 60)

    print("\nPlease enter a paragraph about yourself.")
    print("Include your skills, interests, technologies you work with, etc.")
    print("(Press Enter twice when done)\n")

    # Collect multi-line input
    lines = []
    while True:
        try:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        except EOFError:
            break

    text = "\n".join(lines).strip()

    if not text:
        print("No input provided. Exiting.")
        return

    print(f"\nReceived input ({len(text)} characters)")

    # Initialize and run analysis
    finder = TrendFinder(geo="SG", timeframe="today 1-m")
    results = finder.analyze(text, top_n=3)

    # Display formatted results
    print(finder.format_results(results))

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
