"""Generate academic-style PDF progress report for Trend Finder module."""

from fpdf import FPDF


class AcademicReport(FPDF):
    MARGIN = 25
    ACCENT = (0, 51, 102)  # dark blue
    LIGHT_BG = (240, 245, 250)
    CODE_BG = (245, 245, 245)
    BLACK = (30, 30, 30)
    GRAY = (100, 100, 100)

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)
        self.set_margins(self.MARGIN, self.MARGIN, self.MARGIN)

    # -- header / footer ----------------------------------------------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.GRAY)
        self.cell(0, 8, "TrendPilot -- Trend Finder Progress Report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(self.MARGIN, 18, self.w - self.MARGIN, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.GRAY)
        if self.page_no() == 1:
            self.cell(0, 10, f"Page {self.page_no()}", align="C")
        else:
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # -- helpers -------------------------------------------------------
    def section_title(self, number, title):
        self.ln(6)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 10, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.ACCENT)
        self.line(self.MARGIN, self.get_y(), self.w - self.MARGIN, self.get_y())
        self.ln(4)
        self.set_text_color(*self.BLACK)

    def sub_title(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*self.BLACK)
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BLACK)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        x_start = self.MARGIN + indent
        self.set_x(x_start)
        self.cell(5, 5.5, "-")
        avail_w = self.w - self.get_x() - self.MARGIN
        self.multi_cell(avail_w, 5.5, text)

    def code_block(self, text):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(*self.CODE_BG)
        self.set_text_color(50, 50, 50)
        lines = text.strip().split("\n")
        self.set_x(self.MARGIN + 5)
        # top padding
        self.cell(self.w - 2 * self.MARGIN - 10, 3, "", fill=True, new_x="LMARGIN", new_y="NEXT")
        for line in lines:
            self.set_x(self.MARGIN + 5)
            self.cell(self.w - 2 * self.MARGIN - 10, 4.5, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
        # bottom padding
        self.set_x(self.MARGIN + 5)
        self.cell(self.w - 2 * self.MARGIN - 10, 3, "", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*self.BLACK)
        self.ln(3)

    def table(self, headers, rows, col_widths=None):
        w = self.w - 2 * self.MARGIN
        if col_widths is None:
            col_widths = [w / len(headers)] * len(headers)

        # header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*self.ACCENT)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # data rows
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*self.BLACK)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(*self.LIGHT_BG)
            else:
                self.set_fill_color(255, 255, 255)

            max_lines = 1
            for i, cell in enumerate(row):
                lines_needed = max(1, len(str(cell)) * self.get_string_width("x") / col_widths[i] + 0.5)
                max_lines = max(max_lines, int(lines_needed))
            row_h = max(7, max_lines * 5)

            x_start = self.get_x()
            y_start = self.get_y()

            if y_start + row_h > self.h - 25:
                self.add_page()
                # re-draw header
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(*self.ACCENT)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
                self.ln()
                self.set_font("Helvetica", "", 8.5)
                self.set_text_color(*self.BLACK)
                x_start = self.get_x()
                y_start = self.get_y()

            if fill:
                self.set_fill_color(*self.LIGHT_BG)
            else:
                self.set_fill_color(255, 255, 255)

            for i, cell in enumerate(row):
                self.set_xy(x_start + sum(col_widths[:i]), y_start)
                self.rect(x_start + sum(col_widths[:i]), y_start, col_widths[i], row_h, "DF")
                self.set_xy(x_start + sum(col_widths[:i]) + 1, y_start + 1)
                self.multi_cell(col_widths[i] - 2, 5, str(cell))

            self.set_xy(x_start, y_start + row_h)
            fill = not fill
        self.ln(4)

    def callout(self, label, text):
        self.set_fill_color(255, 248, 230)
        self.set_draw_color(220, 180, 50)
        x = self.get_x()
        y = self.get_y()
        self.rect(x + 5, y, self.w - 2 * self.MARGIN - 10, 18, "DF")
        self.set_xy(x + 8, y + 2)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 5, label)
        self.set_xy(x + 8, y + 8)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(self.w - 2 * self.MARGIN - 20, 5, text)
        self.ln(4)


def build_report():
    pdf = AcademicReport()

    # ==================================================================
    # TITLE PAGE
    # ==================================================================
    pdf.add_page()
    pdf.ln(45)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*pdf.ACCENT)
    pdf.cell(0, 14, "TrendPilot", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*pdf.GRAY)
    pdf.cell(0, 10, "AI-Powered LinkedIn Content Strategy Tool", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(*pdf.ACCENT)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*pdf.BLACK)
    pdf.cell(0, 10, "Trend Finder Module -- Progress Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Iterative Development of Trend Identification Pipeline", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*pdf.GRAY)
    pdf.cell(0, 7, "Author: Biraj Mishra", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Date: February 16, 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Course: MS Capstone Project", align="C", new_x="LMARGIN", new_y="NEXT")

    # ==================================================================
    # TABLE OF CONTENTS
    # ==================================================================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*pdf.ACCENT)
    pdf.cell(0, 10, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    toc_items = [
        ("1", "Objective"),
        ("2", "Iteration Summary"),
        ("3", "Iteration 1: BuzzSumo + News API"),
        ("4", "Iteration 2: Google Trends + spaCy NLP"),
        ("5", "Iteration 3: Google Trends + GPT-4o Prompt Engineering"),
        ("6", "Iteration 4: Interactive Loop + Version Safety + Query Fix"),
        ("7", "System Architecture (Current)"),
        ("8", "Key Learnings"),
        ("9", "File Inventory"),
        ("10", "Next Steps"),
    ]
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*pdf.BLACK)
    for num, title in toc_items:
        pdf.cell(10, 7, num + ".")
        pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")

    # ==================================================================
    # 1. OBJECTIVE
    # ==================================================================
    pdf.add_page()
    pdf.section_title("1", "Objective")
    pdf.body(
        "Build a system that takes a LinkedIn professional's profile bio and automatically identifies "
        "timely, specific, trending topics relevant to their expertise -- then recommends a concrete "
        "LinkedIn post angle backed by real-time search data from Google Trends."
    )
    pdf.body(
        "The system must produce post recommendations that are specific enough to be immediately actionable "
        "(e.g., referencing a new product launch or rising search trend), not generic thought-leadership "
        'angles (e.g., "Why AI is the future"). The recommendations must be grounded in verifiable, '
        "real-time data."
    )

    # ==================================================================
    # 2. ITERATION SUMMARY
    # ==================================================================
    pdf.section_title("2", "Iteration Summary")
    pdf.table(
        ["Iter", "Approach", "Topic Extraction", "Trend Source", "Outcome"],
        [
            ["1", "BuzzSumo + News API", "Keyword matching", "News articles", "Generic, noisy"],
            ["2", "Google Trends + spaCy", "Named Entity Recognition", "related_queries", "Ambiguous entities"],
            ["3", "Google Trends + GPT-4o", "LLM prompt engineering", "interest_over_time + related_queries", "Specific, actionable"],
            ["4", "Iter 3 + Interactive", "Same as Iter 3", "Same + merged API calls", "User control, factual"],
        ],
        col_widths=[12, 35, 35, 45, 33],
    )

    # ==================================================================
    # 3. ITERATION 1
    # ==================================================================
    pdf.section_title("3", "Iteration 1: BuzzSumo + News API")

    pdf.sub_title("Approach")
    pdf.body(
        "Used BuzzSumo API to find popular content and News API to find trending articles. Keywords were "
        "extracted from the user's LinkedIn bio using simple keyword matching, then matched against news "
        "headlines and BuzzSumo trending content."
    )

    pdf.sub_title("Problems Encountered")
    pdf.bullet("Generic topics: Results were dominated by broad news stories (e.g., \"AI is transforming business\", \"Cloud computing market growth\") with no specific angle for a LinkedIn post.")
    pdf.bullet("Noisy results: News API returned articles about politics, sports, entertainment -- not filtered to the user's professional domain.")
    pdf.bullet("Low relevance: No mechanism to score how relevant a trending topic was to the user's specific skill set.")

    pdf.sub_title("Example Output")
    pdf.code_block(
        'Topics found:\n'
        '- "AI Revolution in Healthcare" (news article, not user\'s domain)\n'
        '- "Cloud Computing Market Worth $1.5T by 2030" (generic market report)\n'
        '- "Top 10 Programming Languages 2026" (listicle, not actionable)'
    )

    pdf.sub_title("Decision")
    pdf.body(
        "Abandon news-based approach. Shift to Google Trends for real-time search interest data that "
        "better reflects what professionals are actually searching for."
    )

    # ==================================================================
    # 4. ITERATION 2
    # ==================================================================
    pdf.add_page()
    pdf.section_title("4", "Iteration 2: Google Trends + spaCy NLP")
    pdf.body("File: trend_finder.py (596 lines)")

    pdf.sub_title("Approach")
    pdf.body(
        "Used spaCy's en_core_web_sm NLP model to extract named entities (ORG, PRODUCT, PROPN, etc.) "
        "from the user's LinkedIn bio, then queried Google Trends related_queries API for each extracted entity."
    )

    pdf.sub_title("Key Components")
    pdf.bullet("Entity extraction using spaCy NER with 11 entity labels (PERSON, ORG, GPE, PRODUCT, EVENT, etc.)")
    pdf.bullet("Proper noun extraction for tokens tagged as PROPN")
    pdf.bullet("Noun phrase extraction for multi-word technical terms")
    pdf.bullet("TrendCache class with 7-day TTL and composite cache keys (keyword|geo|timeframe)")
    pdf.bullet("Retry logic with exponential backoff for Google Trends rate limiting (HTTP 429 errors)")

    pdf.sub_title("Entity Extraction Code (spaCy NER)")
    pdf.code_block(
        'ENTITY_LABELS = {\n'
        '    "PERSON", "ORG", "GPE", "PRODUCT", "EVENT",\n'
        '    "WORK_OF_ART", "LAW", "LANGUAGE", "NORP", "FAC", "LOC"\n'
        '}\n'
        '\n'
        'doc = nlp(text)  # spaCy processes text\n'
        'for ent in doc.ents:\n'
        '    if ent.label_ in ENTITY_LABELS:\n'
        '        entities.append(ent.text)  # e.g., "Cloud", "Spring", "Kong"\n'
        '\n'
        'for token in doc:\n'
        '    if token.pos_ == "PROPN":\n'
        '        entities.append(token.text)  # proper nouns'
    )

    pdf.sub_title("Problem 1: The \"Cloud\" Ambiguity")
    pdf.body(
        "spaCy has no domain awareness. When a bio mentioned \"cloud architecture\" or \"Spring Cloud\", "
        "spaCy extracted \"Cloud\" as a generic proper noun. Google Trends then returned results for "
        "cloud weather -- not cloud computing:"
    )
    pdf.code_block(
        'Entity extracted: "Cloud" (PROPN)\n'
        'Google Trends top queries:\n'
        '  - "cloud weather today" (value: 100)\n'
        '  - "cloud types" (value: 65)\n'
        '  - "cumulus cloud" (value: 37)'
    )

    pdf.sub_title("Problem 2: The \"Spring\" Ambiguity")
    pdf.body(
        "Similarly, \"Spring\" from \"Spring Cloud\" or \"Spring Boot\" was extracted standalone. "
        "Google Trends returned seasonal results:"
    )
    pdf.code_block(
        'Entity extracted: "Spring" (PROPN)\n'
        'Google Trends top queries:\n'
        '  - "spring 2026" (value: 100)\n'
        '  - "spring fever" (value: 74)\n'
        '  - "hot spring" (value: 49)\n'
        'Rising queries:\n'
        '  - "spring fever episodes" (+51500%)\n'
        '  - "best places to visit in spring" (+44000%)'
    )

    pdf.sub_title("Problem 3: The \"Kong\" Ambiguity")
    pdf.body(
        "The API gateway tool \"Kong\" was extracted correctly, but Google Trends returned "
        "entertainment results:"
    )
    pdf.code_block(
        'Entity extracted: "Kong" (ORG)\n'
        'Google Trends top queries:\n'
        '  - "king kong" (value: 100)\n'
        '  - "hong kong china" (value: 54)\n'
        '  - "donkey kong switch" (value: 17)\n'
        '  - "godzilla x kong" (value: 14)'
    )

    pdf.sub_title("Problem 4: Rate Limiting")
    pdf.body("The error log (trend_finder.log) shows aggressive 429 errors from Google Trends API:")
    pdf.code_block(
        '2026-02-07 13:45:37 - WARNING - Retry 1/2 for \'java programming\':\n'
        '  Google returned a response with code 429\n'
        '2026-02-07 13:45:40 - WARNING - Retry 2/2 for \'java programming\':\n'
        '  Google returned a response with code 429\n'
        '2026-02-07 13:45:45 - ERROR - Could not fetch trends:\n'
        '  TooManyRequestsError: code 429'
    )

    pdf.sub_title("Decision")
    pdf.body(
        "spaCy NER is not suitable for technical topic extraction from professional bios. The model "
        "lacks domain context and cannot distinguish \"Cloud (technology)\" from \"Cloud (weather)\". "
        "Replace NLP entity extraction with LLM-based prompt engineering."
    )

    # ==================================================================
    # 5. ITERATION 3
    # ==================================================================
    pdf.add_page()
    pdf.section_title("5", "Iteration 3: Google Trends + GPT-4o Prompt Engineering")
    pdf.body("File: trend_identification_v2.py (initial version)")

    pdf.sub_title("Approach")
    pdf.body(
        "Replaced spaCy NER entirely with a GPT-4o prompt that understands professional context. "
        "Added interest_over_time API for trend scoring alongside related_queries (top + rising) for "
        "search context. The LLM both extracts topics AND recommends a post angle."
    )

    pdf.sub_title("Key Changes from Iteration 2")
    pdf.bullet("LLM-driven topic extraction instead of spaCy NER")
    pdf.bullet("Trend scoring via interest_over_time (mean score over 3 months)")
    pdf.bullet("Top + rising queries fed into the post recommendation prompt")
    pdf.bullet("Two-stage LLM pipeline: extract topics -> score via Google Trends -> recommend post")

    pdf.sub_title("Topic Extraction Prompt (GPT-4o)")
    pdf.code_block(
        'You are analyzing a LinkedIn professional bio.\n'
        '\n'
        'Extract 8-12 specific, post-worthy technical topics\n'
        'suitable for LinkedIn.\n'
        'Only include:\n'
        '- Technologies, Platforms, Tools, Frameworks,\n'
        '  Engineering practices\n'
        '\n'
        'Exclude:\n'
        '- Job titles\n'
        '- Generic words (e.g., cloud, data, software)\n'
        '- Soft skills\n'
        '- Single generic nouns\n'
        '\n'
        'Return the result as a comma-separated list.'
    )

    pdf.sub_title("Why This Prompt Works")
    pdf.bullet("Explicitly excludes generic words like \"cloud\", \"data\", \"software\" -- solving the Iteration 2 ambiguity problem.")
    pdf.bullet("Asks for \"specific, post-worthy technical topics\" -- the LLM understands that \"Spring Cloud\" is a technology while \"Spring\" alone is ambiguous.")
    pdf.bullet("Filters out job titles and soft skills that polluted Iteration 2 results.")

    pdf.sub_title("Post Recommendation Prompt (Key Instructions)")
    pdf.code_block(
        'Instructions:\n'
        '1. Focus on SPECIFICITY over generality. Look at the\n'
        '   rising/breakout searches - these reveal new releases,\n'
        '   announcements, hot debates, or breakthroughs.\n'
        '2. The post title MUST reference a specific thing (a new\n'
        '   feature, release, tool, comparison, migration,\n'
        '   controversy, or real-world use case).\n'
        '3. If rising queries mention a specific product launch,\n'
        '   version, integration, or comparison - use that as\n'
        '   the post hook.\n'
        '4. The post should feel like the author is reacting to\n'
        '   something happening NOW.'
    )

    pdf.sub_title("Example Output (Real Cached Data)")
    pdf.body("For a LinkedIn bio mentioning API gateways, microservices, AWS, and integration architecture:")
    pdf.code_block(
        'Topic Extraction (LLM):\n'
        '  AWS, OpenShift, Spring Cloud, Microservices, Apigee,\n'
        '  Kong, REST APIs, Integration Architecture, API Gateways,\n'
        '  Zuul, WSO2, Axway\n'
        '\n'
        'Top Trending Topics (sorted by trend score):\n'
        '  1. AWS              (score: 55.2)\n'
        '  2. OpenShift        (score: 51.7)\n'
        '  3. Spring Cloud     (score: 50.8)\n'
        '  4. Integration Arch (score: 39.2)\n'
        '  5. Microservices    (score: 31.2)\n'
        '\n'
        'AWS - Rising Searches:\n'
        '  aws devops agent (+1600%)\n'
        '  aws transform custom (+1300%)\n'
        '  aws security agent (+300%)\n'
        '\n'
        'Apigee - Rising Searches:\n'
        '  apigee news (+250%)\n'
        '  apigee pricing (+200%)\n'
        '  kong (+190%)'
    )

    pdf.sub_title("Remaining Issues")
    pdf.bullet("Blank related queries: Some topics (OpenShift, Spring Cloud, Zuul) returned empty top_queries and rising_queries. Root cause: the code called build_payload() twice per keyword -- once for interest_over_time, again for related_queries -- and the second call was rate-limited.")
    pdf.ln(2)
    pdf.bullet("LLM hallucinated version numbers: When recommending a post about OpenShift, the LLM produced:")
    pdf.code_block(
        'TOPIC: OpenShift\n'
        'POST TITLE: "OpenShift 4.12: What Its New GitOps Features\n'
        '  Mean for Streamlined Cloud-Native Deployments"\n'
        '\n'
        'Problem: OpenShift 4.12 is outdated (current is 4.21).\n'
        'The LLM fabricated the version from its training data -\n'
        'it was NOT in the search data.'
    )

    # ==================================================================
    # 6. ITERATION 4
    # ==================================================================
    pdf.add_page()
    pdf.section_title("6", "Iteration 4: Interactive Loop + Version Safety + Query Fix")
    pdf.body("File: trend_identification_v2.py (current version, 272 lines)")

    pdf.sub_title("Fix 1: Merged API Calls (Blank Queries)")
    pdf.body("Before (Iteration 3) -- Two separate functions, two build_payload calls per keyword:")
    pdf.code_block(
        'def fetch_trend_score(keyword):\n'
        '    pytrends.build_payload(...)  # 1st API call\n'
        '    df = pytrends.interest_over_time()\n'
        '    return float(df[keyword].mean())\n'
        '\n'
        'def fetch_related_queries(keyword):\n'
        '    pytrends.build_payload(...)  # 2nd call (RATE-LIMITED!)\n'
        '    related = pytrends.related_queries()\n'
        '    ...'
    )
    pdf.body("After (Iteration 4) -- Single function, one build_payload call:")
    pdf.code_block(
        'def fetch_trend_data(keyword):\n'
        '    pytrends.build_payload(...)  # Single API call\n'
        '\n'
        '    # Trend score\n'
        '    df = pytrends.interest_over_time()\n'
        '    score = float(df[keyword].mean())\n'
        '\n'
        '    # Related queries (same payload, no extra call)\n'
        '    related = pytrends.related_queries()\n'
        '    ...\n'
        '    return score, top_queries, rising_queries'
    )
    pdf.body(
        "Cache validation was also updated to re-fetch entries saved without query data:"
    )
    pdf.code_block(
        '# Old: accepted stale entries without queries\n'
        'if cache_entry and is_cache_valid(timestamp):\n'
        '\n'
        '# New: re-fetches if queries are missing\n'
        'if cache_entry and is_cache_valid(timestamp)\n'
        '   and cache_entry.get("top_queries") is not None:'
    )

    pdf.sub_title("Fix 2: Version Number Hallucination Guard")
    pdf.body("Added a critical rule to the post recommendation prompt to prevent fabricated version numbers:")
    pdf.code_block(
        'CRITICAL RULE - Version numbers and factual accuracy:\n'
        '- NEVER invent or guess version numbers, release names,\n'
        '  or feature names. Your training data may be outdated.\n'
        '- ONLY mention a specific version/release if it explicitly\n'
        '  appears in the search query data above.\n'
        '- If no version is mentioned in the search data, write\n'
        '  the post title WITHOUT a version number. Use phrasing\n'
        '  like "latest release", "new update", or focus on the\n'
        '  concept/trend instead.\n'
        '- NEVER fabricate announcements, launches, or features\n'
        '  that are not evidenced by the search data.'
    )
    pdf.body("Added explicit bad/good examples in the prompt:")
    pdf.code_block(
        'BAD:  "OpenShift 4.12: What Its New GitOps Features Mean..."\n'
        '       (version not from search data = hallucinated)\n'
        '\n'
        'GOOD: "OpenShift\'s latest update doubles down on GitOps -\n'
        '       here\'s what changed for cloud-native teams"'
    )

    pdf.sub_title("Fix 3: Interactive Topic Selection Loop")
    pdf.body(
        "Before (Iteration 3): Program ran once, LLM auto-picked a topic, then exited. "
        "After (Iteration 4): User sees top 5 topics and can interactively choose:"
    )
    pdf.code_block(
        '--- Top 5 Trending Topics (Profile-Driven) ---\n'
        '\n'
        '  1. AWS  (trend score: 55.2)\n'
        '     Top searches: what is aws, aws news, aws ai\n'
        '     Rising: aws devops agent (+1600%)\n'
        '\n'
        '  2. OpenShift  (trend score: 51.7)\n'
        '  3. Spring Cloud  (trend score: 50.8)\n'
        '  4. Integration Architecture  (trend score: 39.2)\n'
        '  5. Microservices  (trend score: 31.2)\n'
        '\n'
        '  0. Let AI pick the best topic automatically\n'
        '  q. Quit\n'
        '\n'
        'Choose a topic number (1-5), 0 for AI pick, or q to quit:'
    )
    pdf.body(
        "After each recommendation, the menu loops back -- the user can explore multiple "
        "topics without restarting the program."
    )

    pdf.sub_title("Fix 4: Error Visibility")
    pdf.code_block(
        '# Before: Silent exception swallowing\n'
        'except Exception:\n'
        '    score = 0.0\n'
        '\n'
        '# After: Errors are printed\n'
        'except Exception as e:\n'
        '    print(f"[warn] Failed to fetch: {e}")\n'
        '    score = 0.0'
    )

    pdf.sub_title("Fix 5: JSON Serialization Safety")
    pdf.body(
        "Added explicit type conversion for query values to prevent numpy int64 "
        "serialization errors when saving to the JSON cache:"
    )
    pdf.code_block(
        'top_queries = [\n'
        '    {"query": row["query"], "value": int(row["value"])}\n'
        '    for _, row in top_df.head(10).iterrows()\n'
        ']\n'
        'rising_queries = [\n'
        '    {"query": row["query"], "value": str(row["value"])}\n'
        '    for _, row in rising_df.head(10).iterrows()\n'
        ']'
    )

    # ==================================================================
    # 7. ARCHITECTURE
    # ==================================================================
    pdf.add_page()
    pdf.section_title("7", "System Architecture (Current -- Iteration 4)")

    pdf.body("The current pipeline follows a two-stage LLM architecture with Google Trends data enrichment:")
    pdf.ln(2)

    # Draw architecture diagram using lines and boxes
    pdf.set_draw_color(*pdf.ACCENT)
    pdf.set_fill_color(*pdf.LIGHT_BG)
    pdf.set_font("Helvetica", "B", 9)

    cx = pdf.w / 2  # center x
    bw = 90  # box width
    bh = 18  # box height
    gap = 8

    boxes = [
        ("LinkedIn Bio (User Input)", "User enters professional bio text"),
        ("GPT-4o Topic Extraction", "Extract 8-12 specific technical topics"),
        ("Google Trends API (pytrends)", "interest_over_time + related_queries"),
        ("Rank & Display Top 5 Topics", "Sort by trend score, show search context"),
        ("User Selection (Interactive)", "Pick 1-5, AI auto-pick, or quit"),
        ("GPT-4o Post Recommendation", "Specific, timely title (no hallucinated versions)"),
    ]

    y = pdf.get_y() + 5
    for i, (title, desc) in enumerate(boxes):
        x = cx - bw / 2
        pdf.set_fill_color(*pdf.LIGHT_BG)
        pdf.rect(x, y, bw, bh, "DF")
        pdf.set_xy(x + 2, y + 2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*pdf.ACCENT)
        pdf.cell(bw - 4, 5, title, align="C")
        pdf.set_xy(x + 2, y + 8)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*pdf.GRAY)
        pdf.cell(bw - 4, 5, desc, align="C")

        if i < len(boxes) - 1:
            # arrow
            pdf.set_draw_color(*pdf.ACCENT)
            arrow_y = y + bh
            pdf.line(cx, arrow_y, cx, arrow_y + gap)
            # arrowhead
            pdf.line(cx - 2, arrow_y + gap - 3, cx, arrow_y + gap)
            pdf.line(cx + 2, arrow_y + gap - 3, cx, arrow_y + gap)

        # loop-back arrow for interactive selection (box index 4 -> 3)
        if i == 4:
            loop_x = cx + bw / 2 + 3
            pdf.set_draw_color(150, 150, 150)
            top_y = y - gap - bh  # top of box 4 which is the "Rank & Display" box
            pdf.line(loop_x, y + bh / 2, loop_x + 10, y + bh / 2)
            pdf.line(loop_x + 10, y + bh / 2, loop_x + 10, top_y + bh / 2)
            pdf.line(loop_x + 10, top_y + bh / 2, cx + bw / 2, top_y + bh / 2)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(150, 150, 150)
            pdf.text(loop_x + 12, y - 2, "loop back")

        y = y + bh + gap

    pdf.set_text_color(*pdf.BLACK)
    pdf.set_y(y + 5)

    # ==================================================================
    # 8. KEY LEARNINGS
    # ==================================================================
    pdf.section_title("8", "Key Learnings")
    pdf.table(
        ["Problem", "Root Cause", "Solution"],
        [
            ["Generic topics (Iter 1)", "News APIs return broad content, not professional context", "Switched to Google Trends for real search interest"],
            ["\"Cloud\" = weather (Iter 2)", "spaCy NER has no domain awareness", "Replaced with GPT-4o prompt that excludes generic words"],
            ["\"Spring\" = season (Iter 2)", "NLP model lacks tech context", "LLM extracts \"Spring Cloud\" as a whole term"],
            ["Blank related queries (Iter 3)", "Double build_payload() caused rate limiting", "Merged into single fetch_trend_data() call"],
            ["Hallucinated versions (Iter 3)", "LLM invents versions from stale training data", "Prompt rule: only cite versions from search data"],
            ["One-shot exit (Iter 3)", "Program terminated after one recommendation", "Added interactive loop with re-selection"],
            ["Silent failures (Iter 3)", "except Exception: pass hid errors", "Now prints [warn] with error message"],
        ],
        col_widths=[45, 50, 65],
    )

    # ==================================================================
    # 9. FILE INVENTORY
    # ==================================================================
    pdf.section_title("9", "File Inventory")
    pdf.table(
        ["File", "Lines", "Purpose"],
        [
            ["trend_finder.py", "596", "Iteration 2 - spaCy NER + Google Trends (deprecated)"],
            ["trend_identification_v2.py", "272", "Iteration 3 & 4 - GPT-4o + Google Trends (current)"],
            ["trend_cache.json", "~785", "Shared cache file (7-day TTL)"],
            ["trend_finder.log", "79", "Error logs from Iteration 2 (429 rate-limit errors)"],
            ["requirements.txt", "3", "Dependencies: spacy, pytrends, pandas"],
        ],
        col_widths=[50, 20, 90],
    )

    # ==================================================================
    # 10. NEXT STEPS
    # ==================================================================
    pdf.section_title("10", "Next Steps")
    pdf.bullet("Integrate trend finder output with the post generation agent (streamlit_poc/agents/post_generator.py)")
    pdf.bullet("Add geographic customization (currently hardcoded to US)")
    pdf.bullet("Explore caching related queries more aggressively to reduce API calls")
    pdf.bullet("Add support for multi-profile batch analysis")
    pdf.bullet("Evaluate prompt variations for topic extraction quality")

    # -- save ----------------------------------------------------------
    output_path = "/Users/BirajMishra/work/playground/ms/capstone/capstone_trend_pilot/trend_finder/PROGRESS_REPORT.pdf"
    pdf.output(output_path)
    print(f"\nPDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_report()
