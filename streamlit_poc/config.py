"""
Configuration settings for the LinkedIn Post Generator.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (but don't override existing shell variables)
# This ensures shell environment variables from .zshrc take priority
load_dotenv(override=False)

# API Keys - Prioritizes shell environment variables (from .zshrc)
# Falls back to .env file only if shell variables are not set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
X_API_KEY = os.getenv("X_API_KEY")  # Twitter/X API Bearer Token

# LinkedIn OAuth Configuration
# IMPORTANT: Set these in your environment variables or .env file
# Get credentials from: https://www.linkedin.com/developers/apps
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI", "http://localhost:8501/oauth/callback")
# Note: 'profile' scope should include basic profile fields like headline
# If headline is not available, the app will create a reasonable default
LINKEDIN_SCOPES = ["openid", "profile", "email"]  # Required scopes for profile access

# Model Configurations
OPENAI_MODEL = "gpt-4"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

# Engagement Prediction Settings
ENGAGEMENT_SCORE_THRESHOLD = 70  # Minimum score to accept a post
MAX_REGENERATION_ATTEMPTS = 3    # Maximum times to regenerate posts

# Post Generation Settings
POST_MIN_WORDS = 150
POST_MAX_WORDS = 300
POSTS_PER_LLM = 3  # Number of variations each LLM generates

# Mock Trends Categories
TREND_CATEGORIES = [
    "Artificial Intelligence & Machine Learning",
    "Cloud Computing & DevOps",
    "Cybersecurity",
    "Web3 & Blockchain",
    "Software Engineering Best Practices",
    "Data Science & Analytics",
    "Frontend & Backend Development",
    "Mobile Development"
]

# LinkedIn Scraping Settings
SCRAPING_TIMEOUT = 10000  # milliseconds
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Mermaid Diagram Types
DIAGRAM_TYPES = ["flowchart", "sequence", "architecture", "mindmap"]
