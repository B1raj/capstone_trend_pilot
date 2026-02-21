def format_linkedin_post(post):
    """
    Deterministic formatter for LinkedIn-style posts
    """

    hook = "🚀 How we built an autonomous trend discovery engine"

    body = (
        "At TrendPilot, we designed a system that:\n"
        "• Scans 30,000+ articles daily\n"
        "• Extracts validated trends\n"
        "• Predicts engagement before posting\n\n"
        "Here’s the workflow 👇"
    )

    hashtags = "#AI #ContentStrategy #LinkedInAutomation #DataScience"

    return f"{hook}\n\n{body}\n\n{hashtags}"