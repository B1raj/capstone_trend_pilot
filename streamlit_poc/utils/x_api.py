"""
Twitter/X API integration for fetching trending technology topics.
"""

import requests
import time
from typing import List, Dict, Optional
import config


class XAPIClient:
    """
    Client for interacting with Twitter/X API v2.
    """

    BASE_URL = "https://api.twitter.com/2"

    def __init__(self, bearer_token: Optional[str] = None, retry_attempts: int = 1, retry_delay: int = 5):
        """
        Initialize X API client.

        Args:
            bearer_token: X API Bearer Token (defaults to config.X_API_KEY)
            retry_attempts: Number of retry attempts for failed requests (default: 1, no retries)
            retry_delay: Delay in seconds between retries (default: 5)
        """
        self.bearer_token = bearer_token or config.X_API_KEY
        if not self.bearer_token:
            raise ValueError("X_API_KEY is required but not set in environment variables")
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.last_request_time = 0

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }

    def _rate_limit_delay(self):
        """Ensure minimum delay between API calls to avoid rate limiting."""
        # Minimal delay for fast failover (1 second instead of 5)
        min_delay = 1.0
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < min_delay:
            sleep_time = min_delay - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request_with_retry(self, url: str, params: Dict, operation: str = "API request") -> Optional[Dict]:
        """
        Make an API request with rate limiting (retries disabled for fast failover).

        Args:
            url: API endpoint URL
            params: Query parameters
            operation: Description of the operation for logging

        Returns:
            Response JSON or None if request failed
        """
        for attempt in range(self.retry_attempts):
            try:
                # Add delay before request to avoid rate limiting
                self._rate_limit_delay()

                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=10
                )

                # Handle rate limiting (429) - fail fast
                if response.status_code == 429:
                    print(f"❌ X API rate limited (429) for {operation} - failing fast to use fallback sources")
                    return None

                # Handle other errors - fail fast
                if response.status_code != 200:
                    print(f"❌ X API error for {operation}: {response.status_code} - failing fast to use fallback sources")
                    return None

                return response.json()

            except requests.exceptions.RequestException as e:
                print(f"❌ X API request failed for {operation}: {str(e)} - failing fast to use fallback sources")
                return None

        return None

    def get_trending_topics(self, category: str = "technology", limit: int = 10) -> List[Dict[str, str]]:
        """
        Fetch trending technology topics from X/Twitter.

        This uses a combination of search and trend APIs to find relevant tech topics.

        Args:
            category: Category filter (default: "technology")
            limit: Maximum number of trends to return

        Returns:
            List of dictionaries containing topic, description, and keywords
        """
        trends = []

        try:
            # Search for trending tech-related tweets
            search_queries = [
                "#technology",
                "#AI",
                "#MachineLearning",
                "#CloudComputing",
                "#DevOps",
                "#WebDevelopment",
                "#DataScience",
                "#Cybersecurity",
                "#Programming",
                "#SoftwareEngineering"
            ]

            for query in search_queries[:limit]:
                trend_data = self._search_recent_tweets(query, max_results=10)
                if trend_data:
                    trends.append(trend_data)

            return trends[:limit]

        except Exception as e:
            raise Exception(f"Failed to fetch trends from X API: {str(e)}")

    def _search_recent_tweets(self, query: str, max_results: int = 10) -> Optional[Dict[str, str]]:
        """
        Search recent tweets for a query and extract trending topic information.

        Args:
            query: Search query
            max_results: Maximum number of tweets to analyze

        Returns:
            Dictionary with topic information or None if failed
        """
        try:
            # X API v2 recent search endpoint
            url = f"{self.BASE_URL}/tweets/search/recent"

            params = {
                "query": f"{query} -is:retweet lang:en",
                "max_results": min(max_results, 100),  # API limit is 100
                "tweet.fields": "public_metrics,created_at,entities",
                "expansions": "author_id"
            }

            # Use retry mechanism
            data = self._make_request_with_retry(url, params, operation=f"search query '{query}'")

            if not data or not data.get("data"):
                return None

            # Analyze tweets to extract trend information
            tweets = data["data"]
            total_engagement = sum(
                tweet.get("public_metrics", {}).get("like_count", 0) +
                tweet.get("public_metrics", {}).get("retweet_count", 0)
                for tweet in tweets
            )

            # Extract common keywords from tweets
            keywords = self._extract_keywords_from_tweets(tweets)

            # Generate description based on tweet content
            description = self._generate_topic_description(query, tweets)

            return {
                "topic": query.replace("#", "").replace("_", " ").title(),
                "description": description,
                "relevance_keywords": keywords,
                "engagement": total_engagement,
                "tweet_count": len(tweets)
            }

        except Exception as e:
            print(f"Error processing tweets for query '{query}': {str(e)}")
            return None

    def _extract_keywords_from_tweets(self, tweets: List[Dict]) -> List[str]:
        """
        Extract common keywords from tweets.

        Args:
            tweets: List of tweet objects

        Returns:
            List of extracted keywords
        """
        keywords = set()

        for tweet in tweets:
            # Extract hashtags
            entities = tweet.get("entities", {})
            hashtags = entities.get("hashtags", [])

            for tag in hashtags[:5]:  # Limit to top 5 per tweet
                keywords.add(tag.get("tag", "").lower())

        return list(keywords)[:10]  # Return top 10 keywords

    def _generate_topic_description(self, query: str, tweets: List[Dict]) -> str:
        """
        Generate a description for the topic based on tweet content.

        Args:
            query: Original search query
            tweets: List of tweet objects

        Returns:
            Generated description string
        """
        if not tweets:
            return f"Trending topic: {query}"

        # Use the most engaged tweet as basis for description
        most_engaged = max(
            tweets,
            key=lambda t: (
                t.get("public_metrics", {}).get("like_count", 0) +
                t.get("public_metrics", {}).get("retweet_count", 0)
            )
        )

        tweet_text = most_engaged.get("text", "")

        # Clean up and truncate
        description = tweet_text[:200].strip()

        # Remove URLs
        import re
        description = re.sub(r'http\S+|www\S+', '', description)

        # Remove extra whitespace
        description = ' '.join(description.split())

        return description or f"Trending discussions about {query}"

    def get_woeid_trends(self, woeid: int = 1) -> List[Dict[str, str]]:
        """
        Get trending topics for a specific location (WOEID).

        Note: This requires elevated API access.

        Args:
            woeid: Where On Earth ID (1 = worldwide)

        Returns:
            List of trending topics
        """
        try:
            # Note: This endpoint requires API v1.1 and elevated access
            url = f"https://api.twitter.com/1.1/trends/place.json"

            params = {"id": woeid}

            # Use retry mechanism
            data = self._make_request_with_retry(url, params, operation=f"WOEID trends (woeid={woeid})")

            if not data:
                return []

            if not data or not data[0].get("trends"):
                return []

            trends = []
            for trend in data[0]["trends"][:10]:
                trends.append({
                    "topic": trend.get("name", ""),
                    "description": trend.get("query", ""),
                    "relevance_keywords": [trend.get("name", "").lower()],
                    "tweet_volume": trend.get("tweet_volume", 0)
                })

            return trends

        except Exception as e:
            print(f"Failed to fetch WOEID trends: {str(e)}")
            return []


def get_tech_trends_from_x(limit: int = 10) -> List[Dict[str, str]]:
    """
    Convenience function to fetch technology trends from X.

    Note: Retries are disabled for fast failover to alternative sources.
    If X API fails, it will immediately return empty list and fallback sources will be used.

    Args:
        limit: Maximum number of trends to return

    Returns:
        List of trending technology topics (empty list if API fails)
    """
    try:
        client = XAPIClient(retry_attempts=1)  # No retries, fail fast
        return client.get_trending_topics(category="technology", limit=limit)
    except Exception as e:
        print(f"❌ Error fetching trends from X: {str(e)}")
        # Return empty list for fast failover
        return []
