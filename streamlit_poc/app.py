"""
Streamlit UI for LinkedIn Post Generator
"""

import streamlit as st
from agents.orchestrator import create_orchestrator
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
    </style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<div class="main-header">AI-Powered LinkedIn Post Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Generate engaging LinkedIn posts with AI-created diagrams based on trending topics</div>',
        unsafe_allow_html=True
    )

    # Check for API keys
    if not config.OPENAI_API_KEY or not config.ANTHROPIC_API_KEY:
        st.error(
            "⚠️ API keys not configured. Please set OPENAI_API_KEY and ANTHROPIC_API_KEY in your .env file."
        )
        st.info(
            "1. Copy .env.example to .env\n"
            "2. Add your API keys\n"
            "3. Restart the application"
        )
        return

    # Sidebar with information
    with st.sidebar:
        st.header("How It Works")
        st.markdown("""
        1. **Enter** your LinkedIn profile URL
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

        st.markdown("""
        **Powered by:**
        - OpenAI GPT-4
        - Anthropic Claude
        - LangGraph
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Input section
        st.header("1. Enter LinkedIn Profile")
        linkedin_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://linkedin.com/in/your-profile",
            help="Enter the URL of your public LinkedIn profile"
        )

        # Example profiles
        with st.expander("Need an example? Try these public profiles"):
            st.code("https://linkedin.com/in/satyanadella")
            st.code("https://linkedin.com/in/jeffweiner08")

    with col2:
        st.header("2. Generate")
        generate_button = st.button(
            "Generate Post",
            type="primary",
            use_container_width=True,
            disabled=not linkedin_url
        )

    # Processing and results
    if generate_button and linkedin_url:
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
            progress_placeholder = st.empty()

            with st.spinner("Running AI agents..."):
                # Run the workflow
                result = orchestrator.run(linkedin_url)

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

            # Profile summary
            with st.expander("LinkedIn Profile Summary", expanded=False):
                profile = result.get("profile_data", {})
                st.write(f"**Name:** {profile.get('name', 'N/A')}")
                st.write(f"**Headline:** {profile.get('headline', 'N/A')}")
                if profile.get('about'):
                    st.write(f"**About:** {profile['about'][:200]}...")

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

            with col2:
                iterations = result.get("iteration_count", 1)
                st.metric("Generation Attempts", iterations)

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

            # Additional options
            with st.expander("Alternative Topics"):
                for i, alt_topic in enumerate(result.get("relevant_topics", [])[1:3], 1):
                    st.write(f"**{i}. {alt_topic.get('topic')}**")
                    st.write(f"Relevance: {alt_topic.get('relevance_score', 0)}/100")
                    st.write(alt_topic.get('description', ''))
                    st.divider()

        else:
            st.warning("Processing incomplete. Please try again.")

    # Footer
    st.divider()
    st.caption(
        "Built with Streamlit, LangGraph, OpenAI, and Anthropic Claude | "
        "This tool uses AI to generate content - always review before posting"
    )


if __name__ == "__main__":
    main()
