import os
from app.linkedin_api import (
    register_image_upload,
    upload_image,
    create_linkedin_post
)

# -------------------------------------------------------------------
# STEP 1: Resolve PERSON URN (OpenID-based, fixed for single-user MVP)
# -------------------------------------------------------------------

def create_person_urn():
    """
    In OpenID Connect flow, the LinkedIn member ID is obtained from
    the `sub` field of the id_token during OAuth.
    For this MVP / capstone, we fix it as a constant.
    """
    return "urn:li:person:tVpS60143I"


# -------------------------------------------------------------------
# STEP 2: Upload Image and get MEDIA URN
# -------------------------------------------------------------------

def create_media_urn(person_urn):
    """
    Registers and uploads an image to LinkedIn,
    returning the digital media asset URN.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(
        base_dir,
        "assets",
        "diagrams",
        "trendpilot_workflow.png"
    )

    registration = register_image_upload(person_urn)

    upload_url = registration["value"]["uploadMechanism"][
        "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
    ]["uploadUrl"]

    media_urn = registration["value"]["asset"]

    upload_image(upload_url, image_path)

    return media_urn


# -------------------------------------------------------------------
# STEP 3: Generate / Fetch Final Post Text
# -------------------------------------------------------------------

def get_final_post_text():
    """
    This text would normally come from TrendPilot's
    ranking + selection module.
    """
    return (
        "🚀 How we built an autonomous trend discovery engine\n\n"
        "At TrendPilot, we designed a system that:\n"
        "• Scans 30,000+ news articles daily\n"
        "• Extracts validated, domain-specific trends\n"
        "• Predicts engagement *before* posting\n\n"
        "The workflow below shows how multiple agents collaborate "
        "to turn raw news into high-performing LinkedIn content 👇\n\n"
        "#AI #ContentStrategy #TrendDiscovery #DataEngineering"
    )


# -------------------------------------------------------------------
# STEP 4: Publish Post to LinkedIn
# -------------------------------------------------------------------

def publish_post(person_urn, media_urn, post_text):
    """
    Publishes the final LinkedIn post with attached media.
    """
    response = create_linkedin_post(
        person_urn=person_urn,
        media_urn=media_urn,
        text=post_text
    )
    return response


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------

def main():
    # 1. Identify author
    person_urn = create_person_urn()

    # 2. Upload visual and get media URN
    media_urn = create_media_urn(person_urn)

    # 3. Get final post text
    post_text = get_final_post_text()

    # 4. Publish post
    response = publish_post(person_urn, media_urn, post_text)

    print("✅ LINKEDIN POST PUBLISHED SUCCESSFULLY")
    print("POST RESPONSE:", response)


if __name__ == "__main__":
    main()