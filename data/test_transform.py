import csv
import json
import os
import tempfile
from datetime import date, timedelta

from transform import compute_time_spent, is_valid_post, transform_record, run, COLUMNS


# --- Unit tests for helpers ---

def test_compute_time_spent_valid():
    yesterday = (date.today() - timedelta(days=1)).isoformat() + "T00:00:00.000Z"
    assert compute_time_spent(yesterday) == 1


def test_compute_time_spent_empty():
    assert compute_time_spent("") == ""
    assert compute_time_spent(None) == ""


def test_is_valid_post_with_text():
    assert is_valid_post({"post_text": "Hello world"}) is True


def test_is_valid_post_empty_text():
    assert is_valid_post({"post_text": ""}) is False
    assert is_valid_post({"post_text": "   "}) is False


def test_is_valid_post_missing_key():
    assert is_valid_post({}) is False
    assert is_valid_post({"error": "dead_page"}) is False


def test_is_valid_post_none():
    assert is_valid_post({"post_text": None}) is False


# --- Unit tests for transform_record ---

def test_transform_record_full():
    record = {
        "user_id": "johndoe",
        "headline": "Software Engineer",
        "user_followers": 500,
        "num_connections": 300,
        "date_posted": (date.today() - timedelta(days=5)).isoformat() + "T00:00:00.000Z",
        "post_text": "Check out my new post!",
        "embedded_links": ["https://example.com"],
        "post_type": "article",
        "hashtags": ["#ai", "#ml"],
        "num_likes": 10,
        "num_comments": 3,
    }
    row = transform_record(record)
    assert row["name"] == "johndoe"
    assert row["headline"] == "Software Engineer"
    assert row["location"] == ""
    assert row["followers"] == 500
    assert row["connections"] == 300
    assert row["about"] == ""
    assert row["time_spent"] == 5
    assert row["content"] == "Check out my new post!"
    assert row["content_links"] == "https://example.com"
    assert row["media_type"] == "article"
    assert row["media_url"] == ""
    assert row["num_hashtags"] == 2
    assert row["hashtag_followers"] == ""
    assert row["hashtags"] == "#ai; #ml"
    assert row["reactions"] == 10
    assert row["comments"] == 3
    assert row["views"] == ""
    assert row["votes"] == ""


def test_transform_record_null_hashtags():
    record = {"user_id": "x", "post_text": "hi", "hashtags": None, "embedded_links": None}
    row = transform_record(record)
    assert row["num_hashtags"] == 0
    assert row["hashtags"] == ""
    assert row["content_links"] == ""


# --- Integration test for run() ---

def test_run_skips_errors_and_writes_csv():
    data = [
        {"error": "dead_page", "error_code": "dead_page"},
        {"post_text": "   "},
        {
            "user_id": "alice",
            "headline": "PM",
            "user_followers": 100,
            "num_connections": 50,
            "date_posted": date.today().isoformat() + "T00:00:00.000Z",
            "post_text": "Hello!",
            "embedded_links": [],
            "post_type": "text",
            "hashtags": ["#hello"],
            "num_likes": 5,
            "num_comments": 1,
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        output_path = os.path.join(tmpdir, "output.csv")
        with open(input_path, "w") as f:
            json.dump(data, f)

        rows = run(input_path, output_path)
        assert len(rows) == 1
        assert rows[0]["name"] == "alice"

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 1
        assert list(csv_rows[0].keys()) == COLUMNS
        assert csv_rows[0]["name"] == "alice"
        assert csv_rows[0]["num_hashtags"] == "1"
        assert csv_rows[0]["time_spent"] == "0"


def test_run_all_invalid_produces_empty_csv():
    data = [
        {"error": "dead_page"},
        {"post_text": ""},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        output_path = os.path.join(tmpdir, "output.csv")
        with open(input_path, "w") as f:
            json.dump(data, f)

        rows = run(input_path, output_path)
        assert len(rows) == 0

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 0
