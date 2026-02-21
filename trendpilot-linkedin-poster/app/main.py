from app.formatter import format_linkedin_post
from app.linkedin_api import get_authorization_url
from app.linkedin_api import register_image_upload, upload_image
from app.linkedin_api import create_linkedin_post
import os

PERSON_URN = "urn:li:person:tVpS60143I"
IMAGE_PATH = "assets/diagrams/trendpilot_workflow.png"
MEDIA_URN = "urn:li:digitalmediaAsset:D4D22AQHXoquWvFNTJw"


# def main():
#     # This comes from TrendPilot ranking system
#     selected_post = {
#         "topic": "Autonomous Trend Discovery",
#         "raw_text": """
#         We built an autonomous system that scans thousands of news articles,
#         extracts validated trends, and predicts engagement before posting.
#         """,
#         "predicted_engagement": 0.82,
#         "visual_path": "assets/diagrams/trendpilot_workflow.png"
#     }
#
#     formatted_post = format_linkedin_post(selected_post)
#
#     print("FINAL LINKEDIN POST:")
#     print(formatted_post)
#
# if __name__ == "__main__":
#     main()

# def main():
#     auth_url = get_authorization_url()
#     print("Open this URL in your browser:")
#     print(auth_url)
#
# if __name__ == "__main__":
#     main()


from app.linkedin_api import exchange_code_for_token

# def main():
#     auth_code = input("Paste authorization code here: ").strip()
#     token_data = exchange_code_for_token(auth_code)
#     print("ACCESS TOKEN RESPONSE:")
#     print(token_data)
#
# if __name__ == "__main__":
#     main()

from app.linkedin_api import (
    get_user_urn,
    register_image_upload,
    upload_image
)

# def main():
#     image_path = "assets/diagrams/trendpilot_workflow.png"
#
#     person_urn = get_user_urn()
#     print("PERSON URN:", person_urn)
#
#     registration = register_image_upload(person_urn)
#
#     upload_url = registration["value"]["uploadMechanism"][
#         "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
#     ]["uploadUrl"]
#
#     asset_urn = registration["value"]["asset"]
#
#     upload_image(upload_url, image_path)
#
#     print("IMAGE UPLOADED SUCCESSFULLY")
#     print("MEDIA URN:", asset_urn)
#
#
# if __name__ == "__main__":
#     main()


# def main():
#     auth_code = input("Paste NEW authorization code here: ").strip()
#     token_data = exchange_code_for_token(auth_code)
#
#     print("\nTOKEN RESPONSE:")
#     print(token_data)
#
#
# if __name__ == "__main__":
#     main()

# def main():
#     registration = register_image_upload(PERSON_URN)
#
#     upload_url = registration["value"]["uploadMechanism"][
#         "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
#     ]["uploadUrl"]
#
#     asset_urn = registration["value"]["asset"]
#
#     upload_image(upload_url, IMAGE_PATH)
#
#     print("✅ IMAGE UPLOADED SUCCESSFULLY")
#     print("MEDIA URN:", asset_urn)


POST_TEXT = (
    "🚀 How we built an autonomous trend discovery engine\n\n"
    "At TrendPilot, we designed a system that:\n"
    "• Scans 30,000+ news articles daily\n"
    "• Extracts validated, domain-specific trends\n"
    "• Predicts engagement *before* posting\n\n"
    "The workflow below shows how multiple agents collaborate "
    "to turn raw news into high-performing LinkedIn content 👇\n\n"
    "#AI #ContentStrategy #TrendDiscovery #DataEngineering"
)

def main():
    response = create_linkedin_post(
        PERSON_URN,
        MEDIA_URN,
        POST_TEXT
    )

    print("🎉 POST PUBLISHED SUCCESSFULLY")
    print(response)

if __name__ == "__main__":
    main()

