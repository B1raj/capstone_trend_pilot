import json
import csv
from datetime import date, datetime
import os

INPUT_FILE = os.path.join(os.path.dirname(__file__), "sd_mll1a4381szw98stg7.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "linkedin_posts.csv")

COLUMNS = [
    "name", "headline", "location", "followers", "connections", "about",
    "time_spent", "content", "content_links", "media_type", "media_url",
    "num_hashtags", "hashtag_followers", "hashtags", "reactions", "comments",
    "views", "votes",
]


def compute_time_spent(date_posted_str):
    """Return number of days between today and date_posted."""
    if not date_posted_str:
        return ""
    try:
        posted = datetime.fromisoformat(date_posted_str.replace("Z", "+00:00"))
        delta = date.today() - posted.date()
        return delta.days
    except (ValueError, TypeError):
        return ""


def is_valid_post(record):
    """A post is valid if it has post_text that is non-empty."""
    post_text = record.get("post_text")
    return bool(post_text and str(post_text).strip())


def transform_record(record):
    """Map a single JSON record to the output CSV row dict."""
    hashtags = record.get("hashtags") or []
    embedded_links = record.get("embedded_links") or []

    return {
        "name": record.get("user_id", ""),
        "headline": record.get("headline", ""),
        "location": "",
        "followers": record.get("user_followers", ""),
        "connections": record.get("num_connections", ""),
        "about": "",
        "time_spent": compute_time_spent(record.get("date_posted")),
        "content": record.get("post_text", ""),
        "content_links": "; ".join(embedded_links) if embedded_links else "",
        "media_type": record.get("post_type", ""),
        "media_url": "",
        "num_hashtags": len(hashtags),
        "hashtag_followers": "",
        "hashtags": "; ".join(hashtags) if hashtags else "",
        "reactions": record.get("num_likes", ""),
        "comments": record.get("num_comments", ""),
        "views": "",
        "votes": "",
    }


def run(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for record in data:
        if is_valid_post(record):
            rows.append(transform_record(record))

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")
    return rows


if __name__ == "__main__":
    run()
