"""
LinkedIn profile access using OAuth API or web scraping fallback.
"""

import re
from typing import Dict, Optional
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import config
from utils.linkedin_oauth import LinkedInOAuth


def extract_profile_data_oauth(access_token: str) -> Dict[str, str]:
    """
    Extract profile data using LinkedIn OAuth access token.

    Args:
        access_token: LinkedIn OAuth access token

    Returns:
        Dictionary containing profile data (name, headline, about, etc.)

    Raises:
        Exception: If profile fetch fails
    """
    try:
        oauth_handler = LinkedInOAuth()
        profile_data = oauth_handler.get_detailed_profile(access_token)

        # Ensure all required fields exist
        if not profile_data.get("name"):
            raise ValueError("Unable to fetch profile name from LinkedIn API")

        # Add default values for missing fields
        profile_data.setdefault("headline", "Professional")
        profile_data.setdefault("about", "")
        profile_data.setdefault("current_role", "")
        profile_data.setdefault("skills", [])

        # Print all fetched data to console for debugging
        print("\n" + "="*70)
        print("📊 LINKEDIN PROFILE DATA FETCHED")
        print("="*70)
        import json
        print(json.dumps(profile_data, indent=2, ensure_ascii=False))
        print("="*70 + "\n")

        return profile_data

    except Exception as e:
        raise Exception(f"Failed to fetch profile via OAuth: {str(e)}")


def extract_profile_data(linkedin_url: str) -> Dict[str, str]:
    """
    Extract profile data from a LinkedIn URL.

    Args:
        linkedin_url: The LinkedIn profile URL

    Returns:
        Dictionary containing profile data (name, headline, about, experience, skills)

    Raises:
        ValueError: If URL is invalid or profile is private
        Exception: If scraping fails
    """
    # Validate LinkedIn URL
    if not re.match(r'https?://(www\.)?linkedin\.com/in/[\w-]+/?', linkedin_url):
        raise ValueError("Invalid LinkedIn URL. Expected format: https://linkedin.com/in/username")

    profile_data = {
        "name": "",
        "headline": "",
        "about": "",
        "current_role": "",
        "skills": []
    }

    try:
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=config.USER_AGENT
            )
            page = context.new_page()

            # Navigate to LinkedIn profile
            try:
                page.goto(linkedin_url, timeout=config.SCRAPING_TIMEOUT)
                page.wait_for_load_state("networkidle", timeout=config.SCRAPING_TIMEOUT)
            except PlaywrightTimeout:
                raise Exception("Page load timeout. The profile might be private or require login.")

            # Check if we hit a login wall
            if "authwall" in page.url or "login" in page.url:
                raise ValueError(
                    "This profile appears to be private or requires login. "
                    "Please ensure the profile is set to public."
                )

            # Extract name
            try:
                name_selector = "h1.text-heading-xlarge"
                name_element = page.query_selector(name_selector)
                if name_element:
                    profile_data["name"] = name_element.inner_text().strip()
            except:
                pass

            # Extract headline
            try:
                headline_selector = "div.text-body-medium"
                headline_element = page.query_selector(headline_selector)
                if headline_element:
                    profile_data["headline"] = headline_element.inner_text().strip()
            except:
                pass

            # Extract about section
            try:
                about_selector = "div.display-flex.ph5.pv3"
                about_element = page.query_selector(about_selector)
                if about_element:
                    about_text = about_element.inner_text().strip()
                    # Remove "About" heading if present
                    about_text = re.sub(r'^About\s*', '', about_text, flags=re.IGNORECASE)
                    profile_data["about"] = about_text[:500]  # Limit to 500 chars
            except:
                pass

            # Extract experience (current role)
            try:
                experience_selector = "div.pvs-list__paged-list-item"
                experience_elements = page.query_selector_all(experience_selector)
                if experience_elements:
                    first_exp = experience_elements[0].inner_text().strip()
                    profile_data["current_role"] = first_exp[:200]  # Limit to 200 chars
            except:
                pass

            # Extract skills (simplified - just look for any skill keywords in page)
            try:
                page_content = page.content().lower()
                common_skills = [
                    "python", "javascript", "java", "react", "node.js", "aws", "azure",
                    "machine learning", "data science", "devops", "kubernetes", "docker",
                    "sql", "nosql", "api", "cloud", "agile", "leadership", "management"
                ]
                found_skills = [skill for skill in common_skills if skill in page_content]
                profile_data["skills"] = found_skills[:10]  # Top 10 skills
            except:
                pass

            browser.close()

            # Validate we got some data
            if not profile_data["name"] and not profile_data["headline"]:
                raise Exception(
                    "Could not extract profile data. The profile might be private, "
                    "or the page structure has changed."
                )

            return profile_data

    except Exception as e:
        raise Exception(f"Failed to scrape LinkedIn profile: {str(e)}")


def get_profile_summary(profile_data: Dict[str, str]) -> str:
    """
    Create a text summary of the profile data.

    Args:
        profile_data: Dictionary containing profile information

    Returns:
        Formatted string summary of the profile
    """
    summary_parts = []

    if profile_data.get("name"):
        summary_parts.append(f"Name: {profile_data['name']}")

    if profile_data.get("headline"):
        summary_parts.append(f"Headline: {profile_data['headline']}")

    if profile_data.get("about"):
        summary_parts.append(f"About: {profile_data['about']}")

    if profile_data.get("current_role"):
        summary_parts.append(f"Current Role: {profile_data['current_role']}")

    if profile_data.get("skills"):
        skills_str = ", ".join(profile_data["skills"])
        summary_parts.append(f"Skills: {skills_str}")

    return "\n\n".join(summary_parts)
