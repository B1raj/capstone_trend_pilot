# app/main.py
import os
from app.user_context import UserContext
from app.linkedin_api_new_user import (
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