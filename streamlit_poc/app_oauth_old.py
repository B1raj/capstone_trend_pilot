"""
Streamlit UI for LinkedIn Post Generator with OAuth Support
"""

import streamlit as st
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

    if "code" in query_params:
        auth_code = query_params["code"]

        try:
            oauth_handler = create_oauth_handler()
            token_response = oauth_handler.exchange_code_for_token(auth_code)

            # Store access token in session state
            st.session_state.access_token = token_response.get("access_token")
            st.session_state.authenticated = True

            # Clear query parameters
            st.query_params.clear()

            st.success("✅ Successfully authenticated with LinkedIn!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Authentication failed: {str(e)}")
            st.session_state.authenticated = False


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

    # Handle OAuth callback
    handle_oauth_callback()

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
        2. **AI analyzes** trending topics
        3. **Generates** multiple post variations
        4. **Predicts** engagement scores
        5. **Creates** visual diagrams
        6. **Delivers** ready-to-post content
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
                height=100
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
        st.header("2. Generate LinkedIn Post")

        # Fetch profile if not already fetched
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

        # Show profile info
        with st.expander("📋 Your LinkedIn Profile", expanded=False):
            profile = st.session_state.profile_data
            st.write(f"**Name:** {profile.get('name', 'N/A')}")
            st.write(f"**Headline:** {profile.get('headline', 'N/A')}")
            if profile.get('about'):
                st.write(f"**About:** {profile['about'][:200]}...")

        # Generate post button
        col1, col2 = st.columns([3, 1])

        with col2:
            generate_button = st.button(
                "Generate Post",
                type="primary",
                use_container_width=True
            )

        # Processing and results
        if generate_button:
            # Initialize session state for progress
            if 'progress_messages' not in st.session_state:
                st.session_state.progress_messages = []

            # Progress callback
            def update_progress(message: str, status: str = "running"):
                st.session_state.progress_messages.append((message, status))

            # Create orchestrator
            orchestrator = create_orchestrator(progress_callback=update_progress)

            # Progress display
            progress_container = st.container()
            with progress_container:
                st.header("Processing")

                with st.spinner("Running AI agents..."):
                    # Create a dummy state for orchestrator
                    # Since we already have profile data from OAuth
                    from agents.orchestrator import WorkflowState

                    initial_state = {
                        "linkedin_url": "",  # Not needed with OAuth
                        "profile_data": st.session_state.profile_data,
                        "relevant_topics": [],
                        "selected_topic": None,
                        "generated_posts": [],
                        "selected_post": None,
                        "engagement_score": 0.0,
                        "engagement_feedback": "",
                        "dimension_scores": None,
                        "diagram_code": None,
                        "diagram_type": "none",
                        "iteration_count": 0,
                        "error": None,
                        "status": "profile_scraped"  # Skip scraping step
                    }

                    # Skip scraping and start from trend finding
                    from agents.trend_finder import TrendFinderAgent
                    from agents.post_generator import PostGeneratorAgent
                    from agents.engagement_predictor import EngagementPredictorAgent
                    from agents.image_generator import ImageGeneratorAgent

                    try:
                        # Run agents manually since we're skipping scraping
                        update_progress("Finding relevant trends...", "running")
                        trend_finder = TrendFinderAgent()
                        relevant_topics = trend_finder.find_relevant_trends(st.session_state.profile_data)
                        selected_topic = relevant_topics[0] if relevant_topics else None
                        update_progress(f"Found {len(relevant_topics)} relevant topics", "success")

                        if not selected_topic:
                            st.error("No relevant topics found")
                            return

                        # Generate posts
                        update_progress("Generating posts...", "running")
                        post_generator = PostGeneratorAgent()
                        posts = post_generator.generate_posts(selected_topic, st.session_state.profile_data)
                        selected_post = post_generator.select_best_post(posts)
                        update_progress(f"Generated {len(posts)} post variations", "success")

                        # Predict engagement
                        update_progress("Evaluating post quality...", "running")
                        engagement_predictor = EngagementPredictorAgent()
                        score, dim_scores, feedback, is_approved = engagement_predictor.predict_engagement(
                            selected_post["text"], selected_topic["topic"]
                        )
                        update_progress(f"Engagement score: {score:.1f}/100", "success")

                        # Generate diagram
                        update_progress("Creating visual diagram...", "running")
                        image_generator = ImageGeneratorAgent()
                        diagram_code, diagram_type = image_generator.generate_diagram(
                            selected_post["text"], selected_topic["topic"]
                        )
                        if diagram_code:
                            update_progress(f"Generated {diagram_type} diagram", "success")
                        else:
                            update_progress("No diagram needed", "info")

                        # Create result
                        result = {
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

                    except Exception as e:
                        result = {"error": str(e), "status": "error"}
                        update_progress(f"Error: {str(e)}", "error")

                # Show progress messages
                for message, status in st.session_state.progress_messages:
                    status_class = f"{status}-box" if status in ["success", "info", "warning", "error"] else "info-box"
                    st.markdown(
                        f'<div class="status-box {status_class}">{message}</div>',
                        unsafe_allow_html=True
                    )

            # Clear progress messages for next run
            st.session_state.progress_messages = []

            # Display results
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            elif result.get("status") == "completed":
                st.success("Post generated successfully!")

                # Results section
                st.header("Results")

                # Selected topic
                st.subheader("Selected Topic")
                topic = result.get("selected_topic", {})
                st.info(f"**{topic.get('topic', 'N/A')}**\n\n{topic.get('description', '')}")

                # Engagement score
                st.subheader("Engagement Analysis")
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
                    with st.expander("Detailed Score Breakdown"):
                        dim_scores = result["dimension_scores"]
                        for dimension, score in dim_scores.items():
                            st.progress(score / 100, text=f"{dimension.replace('_', ' ').title()}: {score:.0f}/100")

                # Generated post
                st.subheader("Generated LinkedIn Post")
                post = result.get("selected_post", {})
                post_text = post.get("text", "No post generated")

                # Display post in a nice text area
                st.text_area(
                    "Your Post",
                    value=post_text,
                    height=300,
                    help="Copy this text to LinkedIn"
                )

                # Copy button
                if st.button("📋 Copy to Clipboard", use_container_width=True):
                    st.code(post_text)
                    st.success("Post text displayed above - copy manually")

                # Post metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Style: {post.get('style', 'N/A').title()}")
                with col2:
                    st.caption(f"Generated by: {post.get('provider', 'N/A').upper()}")

                # Diagram
                if result.get("diagram_code"):
                    st.subheader("Visual Diagram")
                    st.caption(f"Diagram Type: {result.get('diagram_type', 'N/A').title()}")

                    # Display mermaid diagram
                    st.code(result["diagram_code"], language="mermaid")

                    st.info(
                        "💡 Copy the mermaid code above and paste it into a mermaid diagram renderer "
                        "(e.g., https://mermaid.live) to visualize it, then take a screenshot to include with your LinkedIn post."
                    )
                else:
                    st.info("No diagram was generated for this post (not needed for this content type)")

                # Alternative topics
                with st.expander("Alternative Topics"):
                    for i, alt_topic in enumerate(result.get("relevant_topics", [])[1:3], 1):
                        st.write(f"**{i}. {alt_topic.get('topic')}**")
                        st.write(f"Relevance: {alt_topic.get('relevance_score', 0)}/100")
                        st.write(alt_topic.get('description', ''))
                        st.divider()

    # Footer
    st.divider()
    st.caption(
        "Built with Streamlit, LangGraph, OpenAI, Anthropic Claude, and LinkedIn OAuth | "
        "This tool uses AI to generate content - always review before posting"
    )


if __name__ == "__main__":
    main()
