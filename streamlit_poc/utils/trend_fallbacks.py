"""
Fallback trend sources when X API is unavailable.
Fetches trending topics from HuggingFace Papers and Reddit.
"""

import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup


def get_huggingface_trends(limit: int = 10) -> List[Dict[str, str]]:
    """
    Fetch trending AI/ML papers from HuggingFace.

    Args:
        limit: Maximum number of trends to return

    Returns:
        List of trending topics from HuggingFace papers
    """
    trends = []

    try:
        url = "https://huggingface.co/papers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find paper entries
        # HuggingFace papers are typically in article or div elements
        papers = soup.find_all('article', limit=limit * 2)  # Get extra in case some fail

        if not papers:
            # Try alternative selectors
            papers = soup.find_all('div', class_='paper-item', limit=limit * 2)

        for paper in papers[:limit]:
            try:
                # Extract title
                title_elem = paper.find('h3') or paper.find('h2') or paper.find(['a', 'strong'])
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)

                # Extract description/abstract
                description_elem = paper.find('p') or paper.find('div', class_='abstract')
                description = ""
                if description_elem:
                    description = description_elem.get_text(strip=True)[:200]

                if not description:
                    description = f"Latest research on {title}"

                # Extract keywords from title
                keywords = _extract_keywords_from_text(title)

                trends.append({
                    "topic": title,
                    "description": description,
                    "relevance_keywords": keywords,
                    "source": "HuggingFace Papers"
                })

            except Exception as e:
                print(f"Error parsing HuggingFace paper: {e}")
                continue

        print(f"Fetched {len(trends)} trends from HuggingFace")
        return trends

    except Exception as e:
        print(f"Failed to fetch HuggingFace trends: {e}")
        return []


def get_reddit_programming_trends(limit: int = 10) -> List[Dict[str, str]]:
    """
    Fetch trending topics from Reddit r/programming.

    Args:
        limit: Maximum number of trends to return

    Returns:
        List of trending topics from Reddit
    """
    trends = []

    try:
        # Use Reddit JSON API (no auth required for public posts)
        url = "https://www.reddit.com/r/programming/hot.json"
        headers = {
            "User-Agent": "Mozilla/5.0 (Python TrendFetcher)"
        }

        params = {
            "limit": limit * 2  # Get extra in case some are not suitable
        }

        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        if not data.get("data") or not data["data"].get("children"):
            print("No posts found from Reddit")
            return []

        posts = data["data"]["children"]

        for post_data in posts[:limit]:
            try:
                post = post_data.get("data", {})

                title = post.get("title", "")
                selftext = post.get("selftext", "")
                score = post.get("score", 0)
                num_comments = post.get("num_comments", 0)

                # Skip low-engagement posts
                if score < 10:
                    continue

                # Create description
                description = selftext[:200] if selftext else f"Discussion with {num_comments} comments and {score} upvotes"

                # Extract keywords
                keywords = _extract_keywords_from_text(title)

                trends.append({
                    "topic": title,
                    "description": description,
                    "relevance_keywords": keywords,
                    "source": "Reddit r/programming",
                    "engagement": score
                })

            except Exception as e:
                print(f"Error parsing Reddit post: {e}")
                continue

        print(f"Fetched {len(trends)} trends from Reddit r/programming")
        return trends

    except Exception as e:
        print(f"Failed to fetch Reddit trends: {e}")
        return []


def _extract_keywords_from_text(text: str) -> List[str]:
    """
    Extract relevant keywords from text.

    Args:
        text: Input text

    Returns:
        List of extracted keywords
    """
    # Common tech keywords to look for
    tech_keywords = [
        "ai", "ml", "machine learning", "deep learning", "neural network",
        "python", "javascript", "rust", "go", "java", "typescript",
        "docker", "kubernetes", "cloud", "aws", "azure", "gcp",
        "react", "vue", "angular", "node", "api", "rest", "graphql",
        "database", "sql", "nosql", "postgres", "mongodb",
        "devops", "cicd", "microservices", "serverless",
        "security", "cryptography", "blockchain", "web3",
        "llm", "gpt", "transformer", "diffusion", "generative",
        "optimization", "performance", "scalability", "architecture"
    ]

    text_lower = text.lower()
    found_keywords = []

    for keyword in tech_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)

    # Also extract capitalized words (likely important terms)
    import re
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    found_keywords.extend([w.lower() for w in words[:5]])

    return list(set(found_keywords))[:10]  # Return unique keywords, max 10


def get_trends_with_fallback(limit: int = 10, use_x_api: bool = True) -> List[Dict[str, str]]:
    """
    Get trending topics with automatic fallback mechanism.

    Tries sources in order:
    1. X/Twitter API (if use_x_api=True and X_API_KEY is set)
    2. HuggingFace Papers
    3. Reddit r/programming
    4. Mock trends (last resort)

    Args:
        limit: Maximum number of trends to return
        use_x_api: Whether to try X API first

    Returns:
        List of trending topics
    """
    from utils.mock_trends import get_mock_trends
    import config

    trends = []

    # Try X API first if enabled (no retries, fast failover)
    if use_x_api and config.X_API_KEY:
        try:
            from utils.x_api import get_tech_trends_from_x
            print("Attempting to fetch trends from X/Twitter API (no retries, fast failover)...")
            trends = get_tech_trends_from_x(limit=limit)
            if trends:
                print(f"✅ Successfully fetched {len(trends)} trends from X API")
                return trends
            else:
                print("❌ X API returned no trends, trying fallback sources...")
        except Exception as e:
            print(f"❌ X API failed: {e}, trying fallback sources...")

    # Fallback 1: HuggingFace Papers
    if not trends:
        try:
            print("Attempting to fetch trends from HuggingFace Papers...")
            trends = get_huggingface_trends(limit=limit)
            if trends:
                print(f"✅ Successfully fetched {len(trends)} trends from HuggingFace")
                return trends
            else:
                print("❌ HuggingFace returned no trends")
        except Exception as e:
            print(f"❌ HuggingFace failed: {e}")

    # Fallback 2: Reddit r/programming
    if not trends:
        try:
            print("Attempting to fetch trends from Reddit r/programming...")
            trends = get_reddit_programming_trends(limit=limit)
            if trends:
                print(f"✅ Successfully fetched {len(trends)} trends from Reddit")
                return trends
            else:
                print("❌ Reddit returned no trends")
        except Exception as e:
            print(f"❌ Reddit failed: {e}")

    # Fallback 3: Mock trends (last resort)
    if not trends:
        print("⚠️ All external sources failed, using mock trends")
        trends = get_mock_trends()

    return trends[:limit]
