"""
LinkedIn OAuth handler for authentication and API access.
"""

import requests
import urllib.parse
from typing import Dict, Optional
import config


class LinkedInOAuth:
    """
    Handles LinkedIn OAuth 2.0 authentication flow.
    """

    # LinkedIn OAuth endpoints
    AUTHORIZATION_URL = "https://www.linkedin.com/oauth/v2/authorization"
    TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
    PROFILE_URL = "https://api.linkedin.com/v2/userinfo"

    def __init__(self):
        self.client_id = config.LINKEDIN_CLIENT_ID
        self.client_secret = config.LINKEDIN_CLIENT_SECRET
        self.redirect_uri = config.LINKEDIN_REDIRECT_URI
        self.scopes = config.LINKEDIN_SCOPES

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Generate LinkedIn authorization URL for OAuth flow.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL to redirect user to
        """
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
        }

        if state:
            params["state"] = state

        query_string = urllib.parse.urlencode(params)
        return f"{self.AUTHORIZATION_URL}?{query_string}"

    def exchange_code_for_token(self, authorization_code: str) -> Dict[str, str]:
        """
        Exchange authorization code for access token.

        Args:
            authorization_code: Authorization code from OAuth callback

        Returns:
            Dictionary containing access_token and other token info

        Raises:
            Exception: If token exchange fails
        """
        print(f"\n{'='*70}")
        print("🔄 EXCHANGING AUTHORIZATION CODE FOR TOKEN")
        print(f"{'='*70}")
        print(f"Client ID: {self.client_id[:20]}..." if self.client_id else "Client ID: NOT SET")
        print(f"Redirect URI: {self.redirect_uri}")
        print(f"Authorization Code: {authorization_code[:30]}...")
        print(f"{'='*70}\n")

        data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        try:
            print(f"Sending token exchange request to LinkedIn...")
            response = requests.post(
                self.TOKEN_URL,
                data=data,
                headers=headers,
                timeout=10
            )

            print(f"Response Status: {response.status_code}")

            if response.status_code != 200:
                print(f"❌ Token exchange failed!")
                print(f"Response: {response.text}")
                response.raise_for_status()

            token_data = response.json()
            print(f"✅ Successfully received access token")
            print(f"Token type: {token_data.get('token_type', 'N/A')}")
            print(f"Expires in: {token_data.get('expires_in', 'N/A')} seconds")
            print(f"{'='*70}\n")

            return token_data

        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to exchange code for token: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nLinkedIn Error: {error_detail}"
                except:
                    error_msg += f"\nResponse Text: {e.response.text}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)

    def get_user_profile(self, access_token: str) -> Dict[str, str]:
        """
        Fetch user profile using access token.

        Args:
            access_token: LinkedIn OAuth access token

        Returns:
            Dictionary containing user profile information

        Raises:
            Exception: If profile fetch fails
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        try:
            # Get basic profile info
            response = requests.get(
                self.PROFILE_URL,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            profile_data = response.json()

            # Transform to our expected format
            return {
                "name": profile_data.get("name", ""),
                "email": profile_data.get("email", ""),
                "given_name": profile_data.get("given_name", ""),
                "family_name": profile_data.get("family_name", ""),
                "picture": profile_data.get("picture", ""),
                "locale": profile_data.get("locale", {}).get("language", ""),
                "sub": profile_data.get("sub", "")  # LinkedIn user ID
            }

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch user profile: {str(e)}")

    def get_detailed_profile(self, access_token: str) -> Dict[str, str]:
        """
        Fetch detailed profile information using LinkedIn API v2.

        Note: This requires additional API permissions beyond basic OAuth.

        Args:
            access_token: LinkedIn OAuth access token

        Returns:
            Dictionary containing detailed profile information
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }

        profile_info = {
            "name": "",
            "email": "",
            "headline": "",
            "about": "",
            "current_role": "",
            "skills": []
        }

        try:
            # Get basic profile from userinfo
            basic_profile = self.get_user_profile(access_token)
            profile_info["name"] = basic_profile.get("name", "")
            profile_info["email"] = basic_profile.get("email", "")

            # Store all fields from basic profile
            profile_info["given_name"] = basic_profile.get("given_name", "")
            profile_info["family_name"] = basic_profile.get("family_name", "")
            profile_info["picture"] = basic_profile.get("picture", "")
            profile_info["locale"] = basic_profile.get("locale", {}).get("language", "") if isinstance(basic_profile.get("locale"), dict) else basic_profile.get("locale", "")
            profile_info["sub"] = basic_profile.get("sub", "")

            print("\n" + "="*70)
            print("🔍 LINKEDIN API - USERINFO RESPONSE")
            print("="*70)
            import json
            print(json.dumps(basic_profile, indent=2, ensure_ascii=False))
            print("="*70 + "\n")

            # Method 1: Try to get headline from /v2/me endpoint
            profile_url = "https://api.linkedin.com/v2/me"

            response = requests.get(
                profile_url,
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                me_data = response.json()

                print("\n" + "="*70)
                print("🔍 LINKEDIN API - /v2/me RESPONSE")
                print("="*70)
                import json
                print(json.dumps(me_data, indent=2, ensure_ascii=False))
                print("="*70 + "\n")

                # Extract headline if available (localized field)
                if "headline" in me_data:
                    headline_data = me_data.get("headline", {})
                    if isinstance(headline_data, dict):
                        # Try localized format
                        localized = headline_data.get("localized", {})
                        if localized:
                            # Try en_US or first available locale
                            profile_info["headline"] = localized.get("en_US", "") or next(iter(localized.values()), "")
                    elif isinstance(headline_data, str):
                        # Direct string format
                        profile_info["headline"] = headline_data

            # Method 2: If headline is still empty, try the profile API endpoint
            if not profile_info["headline"]:
                try:
                    # Try the older lite profile endpoint (may require r_liteprofile scope)
                    lite_profile_url = "https://api.linkedin.com/v2/me?projection=(id,firstName,lastName,profilePicture,headline)"

                    response = requests.get(
                        lite_profile_url,
                        headers=headers,
                        timeout=10
                    )

                    if response.status_code == 200:
                        lite_data = response.json()

                        print("\n" + "="*70)
                        print("🔍 LINKEDIN API - LITE PROFILE RESPONSE (with projection)")
                        print("="*70)
                        import json
                        print(json.dumps(lite_data, indent=2, ensure_ascii=False))
                        print("="*70 + "\n")

                        # Try to extract headline from projection
                        if "headline" in lite_data:
                            headline = lite_data.get("headline", "")
                            if isinstance(headline, dict):
                                localized = headline.get("localized", {})
                                profile_info["headline"] = localized.get("en_US", "") or next(iter(localized.values()), "")
                            elif isinstance(headline, str):
                                profile_info["headline"] = headline

                except Exception as e:
                    print(f"Failed to fetch from lite profile endpoint: {e}")

            # Method 3: If still empty, create a default headline from name/email
            if not profile_info["headline"]:
                print("⚠️  Warning: Could not fetch headline from LinkedIn API")
                # Try to create a reasonable default
                if profile_info["email"]:
                    # Extract domain for a hint
                    domain = profile_info["email"].split("@")[-1].split(".")[0]
                    profile_info["headline"] = f"Professional at {domain.title()}"
                else:
                    profile_info["headline"] = "LinkedIn Professional"

            # Print final extracted profile summary
            print("\n" + "="*70)
            print("✅ FINAL EXTRACTED PROFILE DATA")
            print("="*70)
            print(f"Name: {profile_info.get('name', 'N/A')}")
            print(f"Email: {profile_info.get('email', 'N/A')}")
            print(f"Headline: {profile_info.get('headline', 'N/A')}")
            print(f"Given Name: {profile_info.get('given_name', 'N/A')}")
            print(f"Family Name: {profile_info.get('family_name', 'N/A')}")
            print(f"LinkedIn ID (sub): {profile_info.get('sub', 'N/A')}")
            print(f"Locale: {profile_info.get('locale', 'N/A')}")
            print(f"Picture URL: {profile_info.get('picture', 'N/A')[:50]}..." if profile_info.get('picture') else "Picture URL: N/A")
            print(f"About: {profile_info.get('about', 'N/A')[:100]}..." if profile_info.get('about') else "About: N/A")
            print(f"Current Role: {profile_info.get('current_role', 'N/A')}")
            print(f"Skills: {', '.join(profile_info.get('skills', [])[:5])}" if profile_info.get('skills') else "Skills: N/A")
            print("="*70 + "\n")

            return profile_info

        except requests.exceptions.RequestException as e:
            print(f"Error fetching LinkedIn profile: {e}")
            # Return basic profile if detailed fetch fails
            if not profile_info["headline"] and profile_info["name"]:
                profile_info["headline"] = "LinkedIn Professional"
            return profile_info

    def get_profile_by_url(self, access_token: str, profile_url: str) -> Dict[str, str]:
        """
        Fetch a LinkedIn profile using the profile URL and access token.

        Note: LinkedIn API does not support fetching arbitrary user profiles.
        This method can only fetch the authenticated user's own profile.

        Args:
            access_token: LinkedIn OAuth access token
            profile_url: LinkedIn profile URL (must be the authenticated user's profile)

        Returns:
            Dictionary containing profile information
        """
        # LinkedIn API doesn't allow fetching other users' profiles
        # So we just return the authenticated user's profile
        return self.get_detailed_profile(access_token)


def create_oauth_handler() -> LinkedInOAuth:
    """
    Factory function to create LinkedIn OAuth handler.

    Returns:
        LinkedInOAuth instance
    """
    return LinkedInOAuth()
