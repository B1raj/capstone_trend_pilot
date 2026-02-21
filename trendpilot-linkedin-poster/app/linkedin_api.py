import requests
from urllib.parse import urlencode
from app.config import (
    LINKEDIN_CLIENT_ID,
    LINKEDIN_CLIENT_SECRET,
    LINKEDIN_REDIRECT_URI,
    LINKEDIN_TOKEN_URL,
    SCOPES,
    LINKEDIN_AUTH_URL,
    LINKEDIN_API_BASE,
    LINKEDIN_ACCESS_TOKEN,
)

def get_authorization_url():
    params = {
        "response_type": "code",
        "client_id": LINKEDIN_CLIENT_ID,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "scope": SCOPES,
    }

    return f"{LINKEDIN_AUTH_URL}?{urlencode(params)}"

def exchange_code_for_token(auth_code):
    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "client_id": LINKEDIN_CLIENT_ID,
        "client_secret": LINKEDIN_CLIENT_SECRET,
    }

    response = requests.post(LINKEDIN_TOKEN_URL, data=data)
    response.raise_for_status()
    # 👇 TEMP DEBUG
    print("STATUS:", response.status_code)
    print("RESPONSE TEXT:", response.text)

    response.raise_for_status()
    return response.json()


# def get_user_urn():
#     url = f"{LINKEDIN_API_BASE}/me"
#     headers = {
#         "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}"
#     }
#
#     response = requests.get(url, headers=headers)
#     response.raise_for_status()
#
#     data = response.json()
#     return f"urn:li:person:{data['id']}"

def get_user_urn():
    url = "https://api.linkedin.com/v2/userinfo"

    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)

    # TEMP DEBUG (keep for now)
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)

    response.raise_for_status()

    data = response.json()
    user_id = data["sub"]

    return f"urn:li:person:{user_id}"


def register_image_upload(person_urn):
    url = f"{LINKEDIN_API_BASE}/assets?action=registerUpload"

    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
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


def upload_image(upload_url, image_path):
    with open(image_path, "rb") as image_file:
        headers = {
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "Content-Type": "application/octet-stream"
        }

        response = requests.put(
            upload_url,
            headers=headers,
            data=image_file
        )

        response.raise_for_status()


def create_linkedin_post(person_urn, media_urn, text):
    url = f"{LINKEDIN_API_BASE}/ugcPosts"

    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "author": person_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": text
                },
                "shareMediaCategory": "IMAGE",
                "media": [
                    {
                        "status": "READY",
                        "media": media_urn
                    }
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