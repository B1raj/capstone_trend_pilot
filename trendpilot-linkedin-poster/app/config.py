import os
from dotenv import load_dotenv

load_dotenv()

LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI")

LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"

SCOPES = "openid profile w_member_social"
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_API_BASE = "https://api.linkedin.com/v2"

print("CLIENT ID:", LINKEDIN_CLIENT_ID)
print("REDIRECT URI:", LINKEDIN_REDIRECT_URI)