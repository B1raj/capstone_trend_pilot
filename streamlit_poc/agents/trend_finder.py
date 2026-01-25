"""
Trend Finder Agent: Matches trending topics with user's LinkedIn profile.
"""

import json
from typing import Dict, List
from utils.trend_fallbacks import get_trends_with_fallback
from utils.llm_clients import call_llm
import config


class TrendFinderAgent:
    """
    Agent that finds relevant trending topics based on user's profile.
    """

    def __init__(self):
        self.name = "TrendFinderAgent"

    def find_relevant_trends(self, profile_data: Dict[str, str]) -> List[Dict]:
        """
        Find trending topics most relevant to the user's profile.

        Uses multi-tier fallback system:
        1. X/Twitter API (if X_API_KEY is set)
        2. HuggingFace trending papers
        3. Reddit r/programming hot posts
        4. Mock trends (last resort)

        Args:
            profile_data: Dictionary containing LinkedIn profile information

        Returns:
            List of relevant trends with scores and rationale
        """
        print("\n" + "="*60)
        print("🔍 FETCHING TRENDING TOPICS")
        print("="*60)

        # Use the fallback system to get trends
        # This will automatically try X API, HuggingFace, Reddit, then mock trends
        use_x_api = bool(config.X_API_KEY)
        trends = get_trends_with_fallback(limit=10, use_x_api=use_x_api)

        print(f"\n✅ Retrieved {len(trends)} trending topics")
        if trends and trends[0].get("source"):
            print(f"📊 Source: {trends[0]['source']}")

        # Print raw trending topics
        print("\n" + "-"*60)
        print("📋 RAW TRENDING TOPICS (Before Relevance Scoring):")
        print("-"*60)
        for i, trend in enumerate(trends, 1):
            print(f"\n{i}. {trend.get('topic', 'N/A')[:70]}")
            print(f"   Description: {trend.get('description', 'N/A')[:100]}...")
            keywords = trend.get('relevance_keywords', [])
            if keywords:
                print(f"   Keywords: {', '.join(keywords[:5])}")
            print(f"   Source: {trend.get('source', 'Unknown')}")
        print("-"*60)
        print("="*60 + "\n")

        # Create profile summary for analysis
        profile_summary = self._create_profile_summary(profile_data)

        # Analyze each trend for relevance
        relevant_trends = []
        for trend in trends:
            score, rationale = self._score_trend_relevance(
                trend, profile_summary
            )
            relevant_trends.append({
                "topic": trend["topic"],
                "description": trend["description"],
                "keywords": trend.get("relevance_keywords", []),
                "relevance_score": score,
                "rationale": rationale,
                "source": trend.get("source", "Unknown")
            })

        # Sort by relevance score (descending)
        relevant_trends.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Print detailed relevance analysis
        print("\n" + "="*70)
        print("🎯 TOPIC RELEVANCE ANALYSIS")
        print("="*70)
        print(f"Total topics analyzed: {len(relevant_trends)}")
        print(f"Profile context: {profile_summary[:150]}...")
        print("\n" + "-"*70)
        print("📊 RELEVANCE SCORES (All Topics):")
        print("-"*70)

        for i, trend in enumerate(relevant_trends, 1):
            score = trend['relevance_score']
            topic = trend['topic']
            source = trend.get('source', 'Unknown')

            # Visual score bar
            bar_length = int(score / 5)  # 20 chars max for 100 score
            bar = "█" * bar_length + "░" * (20 - bar_length)

            print(f"\n{i}. [{bar}] {score}/100")
            print(f"   Topic: {topic[:60]}...")
            print(f"   Source: {source}")
            print(f"   Rationale: {trend.get('rationale', 'N/A')[:100]}...")
            if trend.get('keywords'):
                print(f"   Keywords: {', '.join(trend['keywords'][:5])}")

        print("\n" + "="*70)
        print("✅ TOP 3 SELECTED TOPICS:")
        print("="*70)

        for i, trend in enumerate(relevant_trends[:3], 1):
            print(f"\n{i}. {trend['topic']}")
            print(f"   Score: {trend['relevance_score']}/100")
            print(f"   Description: {trend['description'][:120]}...")
            print(f"   Source: {trend.get('source', 'Unknown')}")

        print("\n" + "="*70 + "\n")

        # Return top 3 most relevant trends
        return relevant_trends[:3]

    def _create_profile_summary(self, profile_data: Dict[str, str]) -> str:
        """
        Create a concise summary of the profile for analysis.

        Args:
            profile_data: Profile information dictionary

        Returns:
            Formatted profile summary string
        """
        parts = []

        if profile_data.get("headline"):
            parts.append(f"Professional headline: {profile_data['headline']}")

        if profile_data.get("about"):
            parts.append(f"Background: {profile_data['about']}")

        if profile_data.get("current_role"):
            parts.append(f"Current role: {profile_data['current_role']}")

        if profile_data.get("skills"):
            skills = ", ".join(profile_data["skills"])
            parts.append(f"Skills: {skills}")

        return " | ".join(parts)

    def _score_trend_relevance(
        self, trend: Dict[str, str], profile_summary: str
    ) -> tuple[int, str]:
        """
        Score how relevant a trend is to the user's profile.

        Args:
            trend: Trend dictionary with topic, description, and keywords
            profile_summary: Summary of user's profile

        Returns:
            Tuple of (score 0-100, rationale string)
        """
        system_prompt = """You are an expert at matching trending topics with professional backgrounds.
Your task is to analyze how relevant a trending topic is to a person's professional profile.

Consider:
- Does their background align with this topic?
- Do their skills relate to this trend?
- Would they have credible insights on this topic?
- Would their network find this topic interesting from them?

Respond ONLY with a JSON object in this exact format:
{
    "score": <number between 0-100>,
    "rationale": "<1-2 sentence explanation>"
}"""

        user_prompt = f"""Profile Summary:
{profile_summary}

Trending Topic: {trend['topic']}
Description: {trend['description']}
Keywords: {', '.join(trend['relevance_keywords'])}

Score this trend's relevance to this profile (0-100) and explain why."""

        try:
            response = call_llm(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider="openai",
                temperature=0.3  # Lower temperature for consistent scoring
            )

            # Parse JSON response
            result = json.loads(response)
            score = int(result.get("score", 0))
            rationale = result.get("rationale", "No rationale provided")

            # Ensure score is in valid range
            score = max(0, min(100, score))

            return score, rationale

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: basic keyword matching if LLM response fails
            return self._fallback_scoring(trend, profile_summary)

    def _fallback_scoring(
        self, trend: Dict[str, str], profile_summary: str
    ) -> tuple[int, str]:
        """
        Fallback scoring method using keyword matching.

        Args:
            trend: Trend dictionary
            profile_summary: Profile summary string

        Returns:
            Tuple of (score, rationale)
        """
        profile_lower = profile_summary.lower()
        keywords = trend["relevance_keywords"]

        matches = sum(1 for keyword in keywords if keyword.lower() in profile_lower)
        score = min(100, int((matches / len(keywords)) * 100))

        rationale = (
            f"Matched {matches} out of {len(keywords)} keywords from the trend. "
            f"{'High' if score > 60 else 'Moderate' if score > 30 else 'Low'} relevance detected."
        )

        return score, rationale
