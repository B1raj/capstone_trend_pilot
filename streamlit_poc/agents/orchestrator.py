"""
Orchestrator: LangGraph-based workflow for LinkedIn post generation.
"""

from typing import TypedDict, List, Dict, Optional, Callable
from langgraph.graph import StateGraph, END
from agents.trend_finder import TrendFinderAgent
from agents.post_generator import PostGeneratorAgent
from agents.engagement_predictor import EngagementPredictorAgent
from agents.image_generator import ImageGeneratorAgent
from utils.linkedin_scraper import extract_profile_data
import config


class WorkflowState(TypedDict):
    """State schema for the LangGraph workflow."""
    linkedin_url: str
    profile_data: Optional[Dict[str, str]]
    relevant_topics: List[Dict]
    selected_topic: Optional[Dict]
    generated_posts: List[Dict[str, str]]
    selected_post: Optional[Dict[str, str]]
    engagement_score: float
    engagement_feedback: str
    dimension_scores: Optional[Dict[str, float]]
    diagram_code: Optional[str]
    diagram_type: str
    iteration_count: int
    error: Optional[str]
    status: str


class LinkedInPostOrchestrator:
    """
    Orchestrator that coordinates all agents using LangGraph.
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize orchestrator with all agents.

        Args:
            progress_callback: Optional callback function for progress updates
        """
        self.trend_finder = TrendFinderAgent()
        self.post_generator = PostGeneratorAgent()
        self.engagement_predictor = EngagementPredictorAgent()
        self.image_generator = ImageGeneratorAgent()
        self.progress_callback = progress_callback
        self.workflow = self._build_workflow()

    def _update_progress(self, message: str, status: str = "running"):
        """Send progress update if callback is provided."""
        if self.progress_callback:
            self.progress_callback(message, status)

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph workflow
        """
        # Create the graph
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("scrape_profile", self._scrape_profile_node)
        workflow.add_node("find_trends", self._find_trends_node)
        workflow.add_node("generate_posts", self._generate_posts_node)
        workflow.add_node("predict_engagement", self._predict_engagement_node)
        workflow.add_node("generate_diagram", self._generate_diagram_node)

        # Define edges
        workflow.set_entry_point("scrape_profile")
        workflow.add_edge("scrape_profile", "find_trends")
        workflow.add_edge("find_trends", "generate_posts")
        workflow.add_edge("generate_posts", "predict_engagement")

        # Conditional edge: regenerate if score too low
        workflow.add_conditional_edges(
            "predict_engagement",
            self._should_regenerate,
            {
                "regenerate": "generate_posts",
                "continue": "generate_diagram"
            }
        )

        workflow.add_edge("generate_diagram", END)

        # Compile the graph
        return workflow.compile()

    def _scrape_profile_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Scrape LinkedIn profile."""
        self._update_progress("Scraping LinkedIn profile...")

        try:
            profile_data = extract_profile_data(state["linkedin_url"])
            state["profile_data"] = profile_data
            state["status"] = "profile_scraped"
            self._update_progress("Profile scraped successfully", "success")
        except Exception as e:
            state["error"] = f"Failed to scrape profile: {str(e)}"
            state["status"] = "error"
            self._update_progress(f"Error: {str(e)}", "error")

        return state

    def _find_trends_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Find relevant trends."""
        self._update_progress("Analyzing trending topics...")

        try:
            relevant_topics = self.trend_finder.find_relevant_trends(
                state["profile_data"]
            )
            state["relevant_topics"] = relevant_topics
            # Select the top trend
            state["selected_topic"] = relevant_topics[0] if relevant_topics else None
            state["status"] = "trends_found"
            self._update_progress(
                f"Found {len(relevant_topics)} relevant topics", "success"
            )
        except Exception as e:
            state["error"] = f"Failed to find trends: {str(e)}"
            state["status"] = "error"
            self._update_progress(f"Error: {str(e)}", "error")

        return state

    def _generate_posts_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Generate post variations."""
        iteration = state.get("iteration_count", 0)
        self._update_progress(
            f"Generating posts (attempt {iteration + 1}/{config.MAX_REGENERATION_ATTEMPTS})..."
        )

        try:
            # Get feedback if this is a regeneration
            feedback = state.get("engagement_feedback") if iteration > 0 else None

            # Generate posts
            posts = self.post_generator.generate_posts(
                topic=state["selected_topic"],
                profile_data=state["profile_data"],
                feedback=feedback
            )

            state["generated_posts"] = posts
            # Select the best post (for now, first valid one)
            state["selected_post"] = self.post_generator.select_best_post(posts)
            state["status"] = "posts_generated"
            self._update_progress(f"Generated {len(posts)} post variations", "success")

        except Exception as e:
            state["error"] = f"Failed to generate posts: {str(e)}"
            state["status"] = "error"
            self._update_progress(f"Error: {str(e)}", "error")

        return state

    def _predict_engagement_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Predict engagement score."""
        self._update_progress("Evaluating post quality...")

        try:
            post_text = state["selected_post"]["text"]
            topic_context = state["selected_topic"]["topic"]

            score, dim_scores, feedback, is_approved = (
                self.engagement_predictor.predict_engagement(
                    post_text, topic_context
                )
            )

            state["engagement_score"] = score
            state["dimension_scores"] = dim_scores
            state["engagement_feedback"] = feedback
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            state["status"] = "engagement_predicted"

            if is_approved:
                self._update_progress(
                    f"Post approved with score {score:.1f}/100", "success"
                )
            else:
                self._update_progress(
                    f"Score {score:.1f}/100 - may need improvement", "warning"
                )

        except Exception as e:
            state["error"] = f"Failed to predict engagement: {str(e)}"
            state["status"] = "error"
            self._update_progress(f"Error: {str(e)}", "error")

        return state

    def _generate_diagram_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Generate mermaid diagram."""
        self._update_progress("Creating visual diagram...")

        try:
            post_text = state["selected_post"]["text"]
            topic = state["selected_topic"]["topic"]

            diagram_code, diagram_type = self.image_generator.generate_diagram(
                post_text, topic
            )

            state["diagram_code"] = diagram_code
            state["diagram_type"] = diagram_type
            state["status"] = "completed"

            if diagram_code:
                self._update_progress(
                    f"Generated {diagram_type} diagram", "success"
                )
            else:
                self._update_progress("No diagram needed for this post", "info")

        except Exception as e:
            state["error"] = f"Failed to generate diagram: {str(e)}"
            state["status"] = "error"
            self._update_progress(f"Error: {str(e)}", "error")

        return state

    def _should_regenerate(self, state: WorkflowState) -> str:
        """
        Decide whether to regenerate posts or continue.

        Args:
            state: Current workflow state

        Returns:
            "regenerate" or "continue"
        """
        score = state.get("engagement_score", 0)
        iteration = state.get("iteration_count", 0)

        # Continue if score is good enough or we've hit max iterations
        if score >= config.ENGAGEMENT_SCORE_THRESHOLD:
            return "continue"

        if iteration >= config.MAX_REGENERATION_ATTEMPTS:
            # Accept the post even with low score after max attempts
            return "continue"

        # Otherwise, regenerate
        return "regenerate"

    def run(self, linkedin_url: str) -> WorkflowState:
        """
        Run the complete workflow.

        Args:
            linkedin_url: LinkedIn profile URL

        Returns:
            Final workflow state with generated post and diagram
        """
        # Initialize state
        initial_state: WorkflowState = {
            "linkedin_url": linkedin_url,
            "profile_data": None,
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
            "status": "started"
        }

        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            return final_state
        except Exception as e:
            initial_state["error"] = f"Workflow error: {str(e)}"
            initial_state["status"] = "error"
            return initial_state


def create_orchestrator(progress_callback: Optional[Callable] = None) -> LinkedInPostOrchestrator:
    """
    Factory function to create an orchestrator instance.

    Args:
        progress_callback: Optional callback for progress updates

    Returns:
        LinkedInPostOrchestrator instance
    """
    return LinkedInPostOrchestrator(progress_callback)
