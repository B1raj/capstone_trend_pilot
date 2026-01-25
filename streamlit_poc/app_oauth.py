"""
Streamlit UI for LinkedIn Post Generator with OAuth Support - Enhanced Version
"""

import streamlit as st
import streamlit.components.v1 as components
import urllib.parse
from agents.orchestrator import create_orchestrator
from utils.linkedin_oauth import create_oauth_handler
from utils.linkedin_scraper import extract_profile_data_oauth
import config


# Page configuration
st.set_page_config(
    page_title="AI LinkedIn Post Generator",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #0077B5;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .oauth-button {
        display: inline-block;
        padding: 12px 24px;
        background-color: #0077B5;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .oauth-button:hover {
        background-color: #005582;
    }
    </style>
""", unsafe_allow_html=True)


def handle_oauth_callback():
    """Handle OAuth callback from LinkedIn."""
    query_params = st.query_params

    # Show what we received (for debugging)
    if "code" in query_params:
        # Always show this when code is present
        if not st.session_state.get("callback_processed", False):
            st.info(f"🔍 Received authorization code from LinkedIn. Processing...")

    # Check if we've already processed this callback
    if "code" in query_params and not st.session_state.get("callback_processed", False):
        auth_code = query_params["code"]

        # Mark as being processed to prevent loops
        st.session_state.callback_processed = True

        # Show a processing message
        processing_placeholder = st.empty()
        with processing_placeholder:
            st.warning("🔄 Processing LinkedIn authentication... Please wait...")

        try:
                print(f"\n{'='*70}")
                print(f"🔐 OAUTH CALLBACK HANDLER")
                print(f"{'='*70}")
                print(f"Processing OAuth callback with code: {auth_code[:30]}...")
                print(f"Query params: {dict(query_params)}")
                print(f"{'='*70}\n")

                oauth_handler = create_oauth_handler()

                # Validate handler has required credentials
                if not oauth_handler.client_id or not oauth_handler.client_secret:
                    raise Exception(
                        "LinkedIn credentials not configured! "
                        "Please set LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET in Streamlit Cloud Secrets."
                    )

                # Exchange code for token
                token_response = oauth_handler.exchange_code_for_token(auth_code)

                if not token_response.get("access_token"):
                    raise Exception("No access token received from LinkedIn")

                # Store access token in session state
                st.session_state.access_token = token_response.get("access_token")
                st.session_state.authenticated = True
                st.session_state.auth_error = None  # Clear any previous errors

                print(f"✅ Successfully authenticated! Access token received.")
                print(f"{'='*70}\n")

                # Clear query parameters to prevent reprocessing
                st.query_params.clear()

                st.success("✅ Successfully authenticated with LinkedIn!")
                st.balloons()
                st.rerun()

            except Exception as e:
                error_msg = str(e)
                print(f"\n{'='*70}")
                print(f"❌ OAUTH CALLBACK ERROR")
                print(f"{'='*70}")
                print(f"Error: {error_msg}")
                print(f"{'='*70}\n")

                st.session_state.authenticated = False
                st.session_state.callback_processed = False  # Reset on error
                st.session_state.auth_error = error_msg  # Store error for display

                # Clear the code parameter on error
                st.query_params.clear()

                # Show detailed error
                st.error(f"❌ Authentication failed!")
                st.error(error_msg)

                # Show helpful hints based on error type
                if "redirect_uri" in error_msg.lower():
                    st.warning("""
                    **Redirect URI Mismatch**

                    Make sure:
                    1. LinkedIn Developer Portal has the correct redirect URI
                    2. Streamlit Cloud Secrets has LINKEDIN_REDIRECT_URI set
                    3. They match EXACTLY (including https://, no trailing slash)
                    """)
                elif "credentials" in error_msg.lower() or "client" in error_msg.lower():
                    st.warning("""
                    **Invalid Credentials**

                    Make sure in Streamlit Cloud Secrets:
                    1. LINKEDIN_CLIENT_ID is set correctly
                    2. LINKEDIN_CLIENT_SECRET is set correctly
                    3. Values match those in LinkedIn Developer Portal
                    """)

                # Add a reset button
                if st.button("🔄 Reset and Try Again"):
                    st.session_state.callback_processed = False
                    st.session_state.auth_error = None
                    st.rerun()


def handle_manual_code_entry(auth_code: str):
    """Handle manual authorization code entry."""
    try:
        oauth_handler = create_oauth_handler()

        with st.spinner("Exchanging code for access token..."):
            token_response = oauth_handler.exchange_code_for_token(auth_code)

        # Store access token in session state
        st.session_state.access_token = token_response.get("access_token")
        st.session_state.authenticated = True

        st.success("✅ Successfully authenticated with LinkedIn!")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Authentication failed: {str(e)}")
        st.error("Please ensure you copied the complete authorization code from the URL.")
        return False

    return True


def main():
    """Main Streamlit application with OAuth."""

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'profile_data' not in st.session_state:
        st.session_state.profile_data = None
    if 'generated_result' not in st.session_state:
        st.session_state.generated_result = None
    if 'callback_processed' not in st.session_state:
        st.session_state.callback_processed = False
    if 'auth_error' not in st.session_state:
        st.session_state.auth_error = None

    # Handle OAuth callback
    handle_oauth_callback()

    # Debug section (only show in Streamlit Cloud or when needed)
    if st.query_params.get("debug") == "true":
        with st.expander("🔧 Debug Information", expanded=True):
            st.markdown("### Current State")
            st.write("**Query Parameters:**", dict(st.query_params))
            st.write("**Authenticated:**", st.session_state.authenticated)
            st.write("**Has Access Token:**", bool(st.session_state.access_token))
            st.write("**Callback Processed:**", st.session_state.callback_processed)
            if st.session_state.auth_error:
                st.error(f"**Auth Error:** {st.session_state.auth_error}")

            st.markdown("### Configuration")
            st.write("**REDIRECT_URI:**", config.LINKEDIN_REDIRECT_URI)
            st.write("**Has CLIENT_ID:**", bool(config.LINKEDIN_CLIENT_ID))
            st.write("**Has CLIENT_SECRET:**", bool(config.LINKEDIN_CLIENT_SECRET))
            if config.LINKEDIN_CLIENT_ID:
                st.write("**CLIENT_ID (first 10 chars):**", config.LINKEDIN_CLIENT_ID[:10] + "...")

            st.markdown("### Actions")
            if st.button("🗑️ Clear All Session State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Session cleared! Refreshing...")
                st.rerun()

    # Header
    st.markdown('<div class="main-header">AI-Powered LinkedIn Post Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Generate engaging LinkedIn posts with AI-created diagrams based on trending topics</div>',
        unsafe_allow_html=True
    )

    # Check for LLM API keys
    if not config.OPENAI_API_KEY or not config.ANTHROPIC_API_KEY:
        st.error(
            "⚠️ API keys not configured. Please set OPENAI_API_KEY and ANTHROPIC_API_KEY."
        )
        return

    # Sidebar with information
    with st.sidebar:
        st.header("How It Works")
        st.markdown("""
        1. **Authenticate** with LinkedIn
        2. **Add profile URL** (optional)
        3. **AI analyzes** trending topics
        4. **Generates** multiple post variations
        5. **Predicts** engagement scores
        6. **Creates** visual diagrams
        7. **Delivers** ready-to-post content
        """)

        st.divider()

        st.header("Settings")
        st.metric("Engagement Threshold", f"{config.ENGAGEMENT_SCORE_THRESHOLD}/100")
        st.metric("Max Regeneration Attempts", config.MAX_REGENERATION_ATTEMPTS)

        st.divider()

        # Authentication status
        st.header("Authentication")
        if st.session_state.authenticated:
            st.success("✅ Authenticated")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.access_token = None
                st.session_state.profile_data = None
                st.session_state.generated_result = None
                st.rerun()
        else:
            st.warning("⚠️ Not authenticated")

        st.divider()

        st.markdown("""
        **Powered by:**
        - OpenAI GPT-4
        - Anthropic Claude
        - LangGraph
        - LinkedIn OAuth
        """)

    # Main content
    if not st.session_state.authenticated:
        # Show OAuth login
        st.header("1. Authenticate with LinkedIn")

        st.info("""
        To generate personalized posts, we need access to your LinkedIn profile.
        Click the button below to authenticate with LinkedIn OAuth.
        """)

        oauth_handler = create_oauth_handler()
        auth_url = oauth_handler.get_authorization_url(state="streamlit_app")

        st.markdown(
            f'<a href="{auth_url}" target="_blank" class="oauth-button">🔗 Connect LinkedIn Account</a>',
            unsafe_allow_html=True
        )

        st.divider()

        st.caption("""
        **Note**: Click the button above to authorize on LinkedIn.
        After authorization, you'll be redirected back with an authorization code.
        """)

        # Manual code entry option
        with st.expander("📋 OR: Manually Enter Authorization Code", expanded=False):
            st.markdown("""
            **If the automatic redirect doesn't work:**
            1. Click the "Connect LinkedIn Account" button above
            2. Authorize the application on LinkedIn
            3. Copy the **entire URL** from the browser address bar after redirect
            4. Paste it below or extract just the `code` parameter
            """)

            manual_code_input = st.text_area(
                "Paste the redirect URL or just the authorization code:",
                placeholder="Either paste the full URL:\nhttp://localhost:8501/oauth/callback?code=AQS...\n\nOr just the code:\nAQS...",
                height=100,
                key="manual_code_input"
            )

            if st.button("Submit Authorization Code", type="primary"):
                if manual_code_input:
                    # Extract code from URL if full URL was pasted
                    auth_code = manual_code_input.strip()

                    # Check if it's a full URL
                    if "code=" in auth_code:
                        # Extract code parameter from URL
                        if "?" in auth_code:
                            query_string = auth_code.split("?")[1]
                            params = urllib.parse.parse_qs(query_string)
                            if "code" in params:
                                auth_code = params["code"][0]

                    # Clean up the code (remove any whitespace or newlines)
                    auth_code = auth_code.strip()

                    if auth_code:
                        handle_manual_code_entry(auth_code)
                    else:
                        st.error("Could not extract authorization code. Please try again.")
                else:
                    st.warning("Please paste the authorization code or URL first.")

    else:
        # Authenticated - show post generator
        st.header("2. Profile & Settings")

        # Fetch basic profile if not already fetched
        if not st.session_state.profile_data:
            with st.spinner("Fetching your LinkedIn profile..."):
                try:
                    profile_data = extract_profile_data_oauth(st.session_state.access_token)
                    st.session_state.profile_data = profile_data
                    st.success(f"✅ Profile loaded: {profile_data.get('name', 'User')}")
                except Exception as e:
                    st.error(f"❌ Failed to fetch profile: {str(e)}")
                    if st.button("Retry Authentication"):
                        st.session_state.authenticated = False
                        st.session_state.access_token = None
                        st.rerun()
                    return

        # LinkedIn Profile URL input (optional - for refreshing data)
        st.subheader("📝 Refresh Profile Data")
        st.caption("Click below to fetch the latest data from your LinkedIn profile using OAuth")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Refresh Profile", key="refresh_profile"):
                with st.spinner("Fetching latest profile data from LinkedIn..."):
                    try:
                        oauth_handler = create_oauth_handler()
                        refreshed_profile = oauth_handler.get_detailed_profile(st.session_state.access_token)

                        # Update profile data
                        st.session_state.profile_data.update(refreshed_profile)
                        st.success("✅ Profile data refreshed successfully!")
                    except Exception as e:
                        st.warning(f"⚠️ Could not refresh profile data: {str(e)}")
                        st.info("Continuing with existing profile data.")

        # Show profile info
        with st.expander("📋 Your LinkedIn Profile", expanded=False):
            profile = st.session_state.profile_data

            st.info("💡 **Tip**: All detailed LinkedIn API responses are being printed to your terminal/console for debugging.")

            st.markdown("### 👤 Profile Information")

            # Create columns for better layout
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Name:**")
                st.info(profile.get('name', 'N/A'))

                st.write("**Email:**")
                st.info(profile.get('email', 'N/A'))

            with col2:
                if profile.get('given_name'):
                    st.write("**First Name:**")
                    st.info(profile.get('given_name', 'N/A'))

                if profile.get('family_name'):
                    st.write("**Last Name:**")
                    st.info(profile.get('family_name', 'N/A'))

            st.divider()

            # Show headline with edit option
            st.markdown("### 💼 Professional Information")
            headline = profile.get('headline', 'N/A')
            st.write("**Headline:**")
            st.info(headline)

            # If headline appears to be auto-generated, show info message
            if headline in ["LinkedIn Professional", "N/A"] or "Professional at" in headline:
                st.warning("💡 **Headline not available from LinkedIn API**. You can edit it below to personalize your posts.")

                # Allow manual headline input
                custom_headline = st.text_input(
                    "Enter your professional headline:",
                    value=headline if headline != "N/A" else "",
                    placeholder="e.g., Senior Software Engineer | AI & Machine Learning Enthusiast",
                    key="custom_headline"
                )

                if custom_headline and custom_headline != headline:
                    if st.button("Update Headline", key="update_headline"):
                        st.session_state.profile_data['headline'] = custom_headline
                        st.success("✅ Headline updated!")
                        st.rerun()

            # Current role
            if profile.get('current_role'):
                st.write("**Current Role:**")
                st.info(profile.get('current_role'))

            # About section
            if profile.get('about'):
                st.divider()
                st.markdown("### 📝 About")
                about_text = profile.get('about')
                if len(about_text) > 300:
                    st.text_area("", value=about_text, height=150, disabled=True, key="about_display")
                else:
                    st.info(about_text)

            # Skills
            if profile.get('skills') and len(profile.get('skills', [])) > 0:
                st.divider()
                st.markdown("### 🛠️ Skills")
                skills = profile.get('skills', [])
                if skills:
                    # Display skills as tags
                    skills_html = " ".join([f'<span style="background-color: #0077B5; color: white; padding: 4px 12px; margin: 4px; border-radius: 12px; display: inline-block;">{skill}</span>' for skill in skills[:10]])
                    st.markdown(skills_html, unsafe_allow_html=True)

            # Additional metadata
            st.divider()
            st.markdown("### 🔍 Additional Data")

            # Show all other fields that might be present
            additional_fields = {
                'sub': 'LinkedIn User ID',
                'picture': 'Profile Picture URL',
                'locale': 'Locale/Language',
            }

            has_additional = False
            for key, label in additional_fields.items():
                if profile.get(key):
                    has_additional = True
                    st.write(f"**{label}:**")
                    if key == 'picture':
                        st.image(profile.get(key), width=100)
                    else:
                        st.caption(profile.get(key))

            if not has_additional:
                st.caption("No additional metadata available")

            # Raw data view for debugging
            st.divider()
            with st.expander("🔧 View Raw Profile Data (Debug)"):
                st.json(profile)

        st.divider()

        # Generate post button
        st.header("3. Generate LinkedIn Post")

        st.info("💡 **Debug Info**: Detailed trend analysis and profile matching are being logged to your terminal/console.")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col3:
            generate_button = st.button(
                "🚀 Generate Post",
                type="primary",
                use_container_width=True,
                key="generate_button"
            )

        # Processing and results
        if generate_button:
            # Create placeholder for live updates
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            # Progress display with real-time updates
            with progress_placeholder.container():
                st.header("⚙️ AI Agents Working...")

                # Create status container for live updates
                status_container = st.status("Processing your request...", expanded=True)

                with status_container:
                    from agents.trend_finder import TrendFinderAgent
                    from agents.post_generator import PostGeneratorAgent
                    from agents.engagement_predictor import EngagementPredictorAgent
                    from agents.image_generator import ImageGeneratorAgent

                    try:
                        # Step 1: Find Trends
                        st.write("🔍 **Step 1/5**: Finding relevant trending topics...")
                        st.caption("   Analyzing your profile to find the most relevant trending topics...")
                        trend_finder = TrendFinderAgent()
                        relevant_topics = trend_finder.find_relevant_trends(st.session_state.profile_data)
                        selected_topic = relevant_topics[0] if relevant_topics else None
                        st.write(f"✅ Found {len(relevant_topics)} relevant topics")
                        st.write(f"   📌 Selected: **{selected_topic.get('topic', 'N/A')}**")
                        st.write(f"   🎯 Relevance Score: **{selected_topic.get('relevance_score', 0)}/100**")
                        st.write(f"   📊 Source: {selected_topic.get('source', 'Unknown')}")

                        # Show topic matching details
                        with st.expander("🔍 View Topic Analysis Details"):
                            st.markdown("**Profile-Topic Matching:**")
                            st.caption(f"Rationale: {selected_topic.get('rationale', 'N/A')}")

                            st.markdown("**All Analyzed Topics:**")
                            for i, topic in enumerate(relevant_topics, 1):
                                score = topic.get('relevance_score', 0)
                                # Create color based on score
                                if score >= 70:
                                    emoji = "🟢"
                                elif score >= 50:
                                    emoji = "🟡"
                                else:
                                    emoji = "🔴"

                                st.write(f"{emoji} **{i}. {topic.get('topic', 'N/A')}** - {score}/100")
                                st.caption(f"   Source: {topic.get('source', 'Unknown')}")
                                if i == 1:
                                    st.caption(f"   ✅ Selected for post generation")
                                st.progress(score / 100)

                        if not selected_topic:
                            st.error("No relevant topics found")
                            return

                        # Step 2: Generate Posts
                        st.write("✍️ **Step 2/5**: Generating post variations...")
                        st.write("   - Creating 3 variations with OpenAI GPT-4...")
                        st.write("   - Creating 3 variations with Anthropic Claude...")
                        post_generator = PostGeneratorAgent()
                        posts = post_generator.generate_posts(selected_topic, st.session_state.profile_data)
                        selected_post = post_generator.select_best_post(posts)
                        st.write(f"✅ Generated {len(posts)} post variations")

                        # Step 3: Predict Engagement
                        st.write("📊 **Step 3/5**: Evaluating engagement potential...")
                        engagement_predictor = EngagementPredictorAgent()
                        score, dim_scores, feedback, is_approved = engagement_predictor.predict_engagement(
                            selected_post["text"], selected_topic["topic"]
                        )
                        st.write(f"✅ Engagement score: **{score:.1f}/100**")

                        # Regenerate if needed
                        iteration = 1
                        while not is_approved and iteration < config.MAX_REGENERATION_ATTEMPTS:
                            st.write(f"🔄 **Regeneration #{iteration}**: Score below threshold, improving...")
                            posts = post_generator.generate_posts(selected_topic, st.session_state.profile_data, feedback=feedback)
                            selected_post = post_generator.select_best_post(posts)
                            score, dim_scores, feedback, is_approved = engagement_predictor.predict_engagement(
                                selected_post["text"], selected_topic["topic"]
                            )
                            st.write(f"   New score: **{score:.1f}/100**")
                            iteration += 1

                        # Step 4: Generate Diagram
                        st.write("🎨 **Step 4/5**: Creating visual diagram...")
                        image_generator = ImageGeneratorAgent()
                        diagram_code, diagram_type = image_generator.generate_diagram(
                            selected_post["text"], selected_topic["topic"]
                        )
                        if diagram_code:
                            st.write(f"✅ Generated {diagram_type} diagram")
                        else:
                            st.write("ℹ️ No diagram needed for this post type")

                        # Step 5: Complete
                        st.write("✅ **Step 5/5**: Finalizing results...")

                        # Store result in session state
                        st.session_state.generated_result = {
                            "profile_data": st.session_state.profile_data,
                            "selected_topic": selected_topic,
                            "relevant_topics": relevant_topics,
                            "generated_posts": posts,
                            "selected_post": selected_post,
                            "engagement_score": score,
                            "dimension_scores": dim_scores,
                            "engagement_feedback": feedback,
                            "diagram_code": diagram_code,
                            "diagram_type": diagram_type,
                            "status": "completed",
                            "error": None
                        }

                        st.write("🎉 **All done!** Scroll down to see your results.")
                        status_container.update(label="✅ Processing complete!", state="complete", expanded=False)

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.session_state.generated_result = {"error": str(e), "status": "error"}
                        status_container.update(label="❌ Processing failed", state="error", expanded=True)

        # Display results from session state
        if st.session_state.generated_result:
            result = st.session_state.generated_result

            st.divider()

            if result.get("error"):
                st.error(f"Error: {result['error']}")
            elif result.get("status") == "completed":
                st.success("🎉 Post generated successfully!")

                # Results section
                st.header("📄 Your LinkedIn Post")

                # Selected topic
                st.subheader("🎯 Selected Topic")
                topic = result.get("selected_topic", {})
                topic_source = topic.get('source', 'Unknown')

                st.info(f"**{topic.get('topic', 'N/A')}**\n\n{topic.get('description', '')}")
                st.caption(f"📊 Trend Source: {topic_source}")

                # Engagement score
                st.subheader("📊 Engagement Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    score = result.get("engagement_score", 0)
                    st.metric(
                        "Overall Engagement Score",
                        f"{score:.1f}/100",
                        delta="Approved" if score >= config.ENGAGEMENT_SCORE_THRESHOLD else "Below threshold"
                    )

                # Dimension scores
                if result.get("dimension_scores"):
                    with st.expander("📈 Detailed Score Breakdown"):
                        dim_scores = result["dimension_scores"]

                        # Define dimension descriptions and weights
                        dimension_info = {
                            "clarity": {
                                "description": "How clear and easy to understand is the message?",
                                "weight": "20%",
                                "details": "Measures if the post has a clear central message, uses simple language, and avoids jargon or ambiguity."
                            },
                            "relevance": {
                                "description": "How relevant is the content to the topic and target audience?",
                                "weight": "25% (Highest)",
                                "details": "Evaluates alignment with the trending topic, connection to the target audience's interests and professional background, timeliness, and whether the content addresses current challenges or opportunities in the field."
                            },
                            "call_to_action": {
                                "description": "Does the post encourage engagement and discussion?",
                                "weight": "15%",
                                "details": "Checks for questions, invitations to share opinions, or prompts that encourage comments, shares, and meaningful interactions."
                            },
                            "professional_tone": {
                                "description": "Is the tone appropriate for LinkedIn's professional audience?",
                                "weight": "15%",
                                "details": "Assesses if the tone is professional yet conversational, respectful, and suitable for a business networking platform."
                            },
                            "value_proposition": {
                                "description": "Does the post provide genuine value to readers?",
                                "weight": "20%",
                                "details": "Measures whether readers gain insights, actionable advice, new perspectives, or useful information they can apply."
                            },
                            "formatting": {
                                "description": "Is the post well-structured and easy to read?",
                                "weight": "5%",
                                "details": "Evaluates use of line breaks, bullet points, emojis (when appropriate), hashtags, and overall visual structure."
                            }
                        }

                        st.markdown("### Score Components")
                        st.caption("Each dimension contributes to the overall engagement score with different weights")

                        for dimension, score in dim_scores.items():
                            info = dimension_info.get(dimension, {
                                "description": dimension.replace("_", " ").title(),
                                "weight": "N/A",
                                "details": "No additional details available."
                            })

                            # Display dimension name and score
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{dimension.replace('_', ' ').title()}** (Weight: {info['weight']})")
                            with col2:
                                st.metric("", f"{score:.0f}/100", delta=None)

                            # Progress bar
                            st.progress(score / 100)

                            # Description and details
                            st.caption(f"📝 {info['description']}")

                            # Show detailed explanation in a nested expander for Relevance
                            if dimension == "relevance":
                                with st.expander("ℹ️ What makes content relevant?"):
                                    st.markdown(f"""
                                    {info['details']}

                                    **Relevance is scored based on:**
                                    - **Topic alignment**: How well the post connects to the trending topic
                                    - **Audience fit**: Relevance to your professional background and network
                                    - **Timeliness**: Whether the content is current and timely
                                    - **Professional value**: Does it address challenges or opportunities your audience cares about?

                                    **Why it matters**: Relevance has the highest weight (25%) because LinkedIn's algorithm prioritizes content that resonates with your specific professional network.
                                    """)
                            else:
                                with st.expander("ℹ️ Learn more"):
                                    st.caption(info['details'])

                            st.divider()

                # Generated post
                st.subheader("✍️ Generated LinkedIn Post")
                post = result.get("selected_post", {})
                post_text = post.get("text", "No post generated")

                # Display post in a code block for easy copying
                st.code(post_text, language="text")

                st.info("👆 Click anywhere in the text box above, then press Ctrl+A (or Cmd+A) to select all, and Ctrl+C (or Cmd+C) to copy!")

                # Post metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"📝 Style: {post.get('style', 'N/A').title()}")
                with col2:
                    st.caption(f"🤖 Generated by: {post.get('provider', 'N/A').upper()}")

                # Diagram
                if result.get("diagram_code"):
                    st.subheader("🎨 Visual Diagram")
                    diagram_type = result.get('diagram_type', 'N/A').title()
                    st.caption(f"Diagram Type: {diagram_type}")

                    # Render mermaid diagram using HTML/JavaScript
                    mermaid_code = result["diagram_code"]

                    # Escape the mermaid code for safe HTML embedding
                    import html
                    escaped_mermaid = html.escape(mermaid_code)

                    # Create complete HTML document with mermaid rendering
                    mermaid_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="UTF-8">
                        <style>
                            body {{
                                margin: 0;
                                padding: 20px;
                                background: white;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                min-height: 400px;
                            }}
                            #mermaid-diagram {{
                                max-width: 100%;
                                overflow: auto;
                            }}
                        </style>
                    </head>
                    <body>
                        <div id="mermaid-diagram"></div>
                        <script type="module">
                            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
                            mermaid.initialize({{
                                startOnLoad: true,
                                theme: 'default',
                                securityLevel: 'loose'
                            }});

                            const mermaidCode = `{mermaid_code}`;

                            try {{
                                const {{ svg }} = await mermaid.render('mermaid-svg', mermaidCode);
                                document.getElementById('mermaid-diagram').innerHTML = svg;
                            }} catch (error) {{
                                document.getElementById('mermaid-diagram').innerHTML =
                                    '<div style="color: red; padding: 20px;">Error rendering diagram: ' + error.message + '</div>';
                            }}
                        </script>
                    </body>
                    </html>
                    """

                    # Render using Streamlit components
                    components.html(mermaid_html, height=500, scrolling=True)

                    # Also show the code in an expander for copying
                    with st.expander("📋 View Mermaid Code"):
                        st.code(mermaid_code, language="mermaid")
                        st.caption(
                            "💡 You can copy this code and paste it into [mermaid.live](https://mermaid.live) "
                            "to edit or export as an image for your LinkedIn post."
                        )
                else:
                    st.info("ℹ️ No diagram was generated for this post (not needed for this content type)")

                # Alternative topics
                with st.expander("🔄 Alternative Topics (Click to Explore)"):
                    for i, alt_topic in enumerate(result.get("relevant_topics", [])[1:3], 1):
                        st.write(f"**{i}. {alt_topic.get('topic')}**")
                        st.write(f"   Relevance: {alt_topic.get('relevance_score', 0)}/100")
                        st.write(f"   {alt_topic.get('description', '')}")
                        if alt_topic.get('source'):
                            st.caption(f"   📊 Source: {alt_topic.get('source')}")
                        if i < 2:
                            st.divider()

    # Footer
    st.divider()
    st.caption(
        "Built with Streamlit, LangGraph, OpenAI, Anthropic Claude, and LinkedIn OAuth | "
        "This tool uses AI to generate content - always review before posting"
    )


if __name__ == "__main__":
    main()
