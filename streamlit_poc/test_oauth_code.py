"""
Test script to exchange LinkedIn authorization code for access token.
"""

from utils.linkedin_oauth import create_oauth_handler

# The authorization code from the URL provided by the user
auth_code = "AQSGT0i6VsvHn2s-SMKYnCtKWIC3M-9pWCLGcRUYPyfCUznwZWPdh-JLYUw0FL1oHtzQLIF5bkmlqBTg7mhEyqhfrJfvcJNrWY2jnI4c0kXsXXqmNaiK7sxqSB_cxruX8uDh4cIyln2GTPrCKDbIPFzUthvFaCGVQmEqPn1fUvdj16TOLmZBc3HkeEgmhAGD2BDxkpTsVrNjc9IKgb0"

print("Testing OAuth token exchange...")
print(f"Authorization code: {auth_code[:50]}...")
print()

try:
    oauth_handler = create_oauth_handler()

    print("Exchanging code for access token...")
    token_response = oauth_handler.exchange_code_for_token(auth_code)

    print("✅ Token exchange successful!")
    print()
    print("Token Response:")
    print(f"  Access Token: {token_response.get('access_token', 'N/A')[:50]}...")
    print(f"  Expires In: {token_response.get('expires_in', 'N/A')} seconds")
    print(f"  Scope: {token_response.get('scope', 'N/A')}")
    print()

    # Try to fetch profile
    access_token = token_response.get("access_token")
    if access_token:
        print("Fetching user profile...")
        profile = oauth_handler.get_user_profile(access_token)

        print("✅ Profile fetch successful!")
        print()
        print("Profile Data:")
        print(f"  Name: {profile.get('name', 'N/A')}")
        print(f"  Email: {profile.get('email', 'N/A')}")
        print(f"  Given Name: {profile.get('given_name', 'N/A')}")
        print(f"  Family Name: {profile.get('family_name', 'N/A')}")
        print(f"  User ID (sub): {profile.get('sub', 'N/A')}")
        print(f"  Locale: {profile.get('locale', 'N/A')}")

except Exception as e:
    print(f"❌ Error: {str(e)}")
    print()
    print("This could mean:")
    print("  - The authorization code has expired (they expire quickly)")
    print("  - The code was already used")
    print("  - There's an issue with the OAuth configuration")
    print()
    print("Solution: Get a fresh authorization code by:")
    print("  1. Going to the app: http://localhost:8501")
    print("  2. Clicking 'Connect LinkedIn Account'")
    print("  3. Authorizing on LinkedIn")
    print("  4. Copying the new code from the redirect URL")
