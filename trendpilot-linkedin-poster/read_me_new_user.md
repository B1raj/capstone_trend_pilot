
# -------------------------------------------------------------------
# 🧭 What changes when a NEW user wants to post?
# -------------------------------------------------------------------

Right now, your system assumes:

One fixed user

One fixed PERSON_URN

One fixed access token

For multi-user posting, you must:

Identify which user is posting

Store OAuth tokens per user

Derive PERSON_URN per user

Run the same pipeline with user-specific credentials

🔑 Key Architectural Shift
Before (single user)
.env → one access token → one person URN
After (multi-user)
User logs in → OAuth → tokens stored → pipeline runs for that user

So the pipeline stays the same, but:

Identity becomes dynamic

Tokens become user-scoped

✅ Minimal, Capstone-Ready Multi-User Design

We will:

Add a UserContext

Pass it through main.py

Avoid hardcoding anything user-specific

No databases required yet — we’ll use an in-memory / file-backed store (acceptable for capstone).

🧩 New Concept: UserContext

This represents one LinkedIn user session.

UserContext = {
    "user_id": "...",
    "access_token": "...",
    "person_urn": "..."
}
🗂️ File-by-file changes
1️⃣ NEW FILE: app/user_context.py
# app/user_context.py

class UserContext:
    """
    Represents an authenticated LinkedIn user.
    """

    def __init__(self, user_id, access_token, person_urn):
        self.user_id = user_id
        self.access_token = access_token
        self.person_urn = person_urn
2️⃣ Update linkedin_api.py (MOST IMPORTANT)
🔴 What changes?

Stop reading access token from .env

Accept it per function call

✅ Updated linkedin_api.py
# app/linkedin_api.py
import requests
from app.config import LINKEDIN_API_BASE


def register_image_upload(person_urn, access_token):
    url = f"{LINKEDIN_API_BASE}/assets?action=registerUpload"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
            "owner": person_urn,
            "serviceRelationships": [
                {
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent"
                }
            ]
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def upload_image(upload_url, image_path, access_token):
    with open(image_path, "rb") as image_file:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/octet-stream"
        }

        response = requests.put(upload_url, headers=headers, data=image_file)
        response.raise_for_status()


def create_linkedin_post(person_urn, media_urn, text, access_token):
    url = f"{LINKEDIN_API_BASE}/ugcPosts"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "author": person_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "IMAGE",
                "media": [
                    {"status": "READY", "media": media_urn}
                ]
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

✅ This is the correct multi-user API design.

3️⃣ Updated main.py (Clean, Multi-User Orchestrator)
🔴 What changed?

No hardcoded PERSON_URN

No global access token

Everything comes from UserContext

✅ Final Multi-User main.py
# app/main.py
import os
from app.user_context import UserContext
from app.linkedin_api import (
    register_image_upload,
    upload_image,
    create_linkedin_post
)

# -------------------------------------------------------------------
# STEP 1: Create User Context (normally after OAuth)
# -------------------------------------------------------------------

def create_user_context():
    """
    In production:
    - user_id comes from your system
    - access_token comes from OAuth storage
    - person_urn comes from id_token.sub
    """

    user_id = "user_123"  # example
    access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")  # per-user in prod
    person_urn = "urn:li:person:tVpS60143I"

    return UserContext(user_id, access_token, person_urn)


# -------------------------------------------------------------------
# STEP 2: Upload Image and get MEDIA URN
# -------------------------------------------------------------------

def create_media_urn(user_ctx):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(
        base_dir,
        "assets",
        "diagrams",
        "trendpilot_workflow.png"
    )

    registration = register_image_upload(
        user_ctx.person_urn,
        user_ctx.access_token
    )

    upload_url = registration["value"]["uploadMechanism"][
        "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
    ]["uploadUrl"]

    media_urn = registration["value"]["asset"]

    upload_image(upload_url, image_path, user_ctx.access_token)

    return media_urn


# -------------------------------------------------------------------
# STEP 3: Get Final Post Text
# -------------------------------------------------------------------

def get_final_post_text(user_ctx):
    return (
        f"🚀 TrendPilot posting on behalf of {user_ctx.user_id}\n\n"
        "This post was autonomously selected, generated, and ranked "
        "using TrendPilot’s multi-agent pipeline.\n\n"
        "#AI #LinkedInAutomation #TrendPilot"
    )


# -------------------------------------------------------------------
# STEP 4: Publish Post
# -------------------------------------------------------------------

def publish_post(user_ctx, media_urn, post_text):
    return create_linkedin_post(
        person_urn=user_ctx.person_urn,
        media_urn=media_urn,
        text=post_text,
        access_token=user_ctx.access_token
    )


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------

def main():
    user_ctx = create_user_context()
    media_urn = create_media_urn(user_ctx)
    post_text = get_final_post_text(user_ctx)

    response = publish_post(user_ctx, media_urn, post_text)

    print("✅ POST PUBLISHED FOR USER:", user_ctx.user_id)
    print("RESPONSE:", response)


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------
# To run the files in Terminal window
# -------------------------------------------------------------------
cd trendpilot-linkedin-poster
python -m app.main

 python -m app.main_pipeline

 python -m app.main_multiuser
