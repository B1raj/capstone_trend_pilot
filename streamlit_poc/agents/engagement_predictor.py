"""
Engagement Predictor Agent: Scores posts and provides improvement feedback.
"""

import json
from typing import Dict, Tuple
from utils.llm_clients import call_llm
import config


class EngagementPredictorAgent:
    """
    Agent that predicts engagement potential of LinkedIn posts using LLM-based scoring.
    """

    def __init__(self):
        self.name = "EngagementPredictorAgent"
        self.threshold = config.ENGAGEMENT_SCORE_THRESHOLD

    def predict_engagement(
        self, post_text: str, topic_context: str
    ) -> Tuple[float, Dict[str, float], str, bool]:
        """
        Predict engagement score for a post.

        Args:
            post_text: The LinkedIn post text
            topic_context: Context about the topic

        Returns:
            Tuple of (overall_score, dimension_scores, feedback, is_approved)
        """
        dimension_scores = self._score_dimensions(post_text, topic_context)
        overall_score = self._calculate_overall_score(dimension_scores)
        is_approved = overall_score >= self.threshold

        if is_approved:
            feedback = self._generate_approval_message(dimension_scores)
        else:
            feedback = self._generate_improvement_feedback(dimension_scores, post_text)

        return overall_score, dimension_scores, feedback, is_approved

    def _score_dimensions(
        self, post_text: str, topic_context: str
    ) -> Dict[str, float]:
        """
        Score the post on multiple dimensions using LLM.

        Args:
            post_text: The post content
            topic_context: Topic information

        Returns:
            Dictionary of dimension scores (0-100)
        """
        system_prompt = """You are a LinkedIn engagement expert and content strategist.
Analyze the given LinkedIn post and score it on multiple dimensions.

Respond ONLY with a JSON object in this exact format:
{
    "clarity": <0-100>,
    "relevance": <0-100>,
    "call_to_action": <0-100>,
    "professional_tone": <0-100>,
    "value_proposition": <0-100>,
    "formatting": <0-100>
}

Scoring Guidelines:
- Clarity (0-100): Is the message clear and easy to understand?
- Relevance (0-100): How relevant is it to the topic and target audience?
- Call to Action (0-100): Does it encourage engagement (comments, shares, discussion)?
- Professional Tone (0-100): Appropriate tone for LinkedIn?
- Value Proposition (0-100): Does it provide genuine value to readers?
- Formatting (0-100): Good use of spacing, structure, hashtags?"""

        user_prompt = f"""Topic Context: {topic_context}

LinkedIn Post:
{post_text}

Score this post on all dimensions (0-100 for each):"""

        try:
            response = call_llm(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider="openai",
                temperature=0.2  # Low temperature for consistent scoring
            )

            # Parse JSON response
            scores = json.loads(response)

            # Ensure all scores are in valid range
            for key in scores:
                scores[key] = max(0, min(100, float(scores[key])))

            return scores

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback scores if parsing fails
            return {
                "clarity": 50,
                "relevance": 50,
                "call_to_action": 50,
                "professional_tone": 50,
                "value_proposition": 50,
                "formatting": 50
            }

    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall score from dimension scores.

        Args:
            dimension_scores: Dictionary of individual dimension scores

        Returns:
            Weighted overall score (0-100)
        """
        # Define weights for each dimension
        weights = {
            "clarity": 0.20,
            "relevance": 0.25,
            "call_to_action": 0.15,
            "professional_tone": 0.15,
            "value_proposition": 0.20,
            "formatting": 0.05
        }

        # Calculate weighted sum
        overall = sum(
            dimension_scores.get(dim, 0) * weight
            for dim, weight in weights.items()
        )

        return round(overall, 1)

    def _generate_improvement_feedback(
        self, dimension_scores: Dict[str, float], post_text: str
    ) -> str:
        """
        Generate specific feedback for improving the post.

        Args:
            dimension_scores: Scores for each dimension
            post_text: The original post text

        Returns:
            Detailed improvement feedback
        """
        # Find lowest scoring dimensions
        sorted_dims = sorted(
            dimension_scores.items(),
            key=lambda x: x[1]
        )
        weakest_dims = sorted_dims[:3]  # Top 3 areas to improve

        system_prompt = """You are a LinkedIn content coach providing constructive feedback.
Based on the low-scoring dimensions, provide specific, actionable advice to improve the post.

Be concise but specific. Focus on the 2-3 most important improvements."""

        user_prompt = f"""Post to improve:
{post_text}

Dimension Scores (0-100):
{json.dumps(dimension_scores, indent=2)}

Weakest areas: {', '.join([f"{dim} ({score:.0f}/100)" for dim, score in weakest_dims])}

Provide specific feedback on how to improve this post, focusing on the weakest dimensions:"""

        try:
            feedback = call_llm(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider="anthropic",  # Use Claude for feedback generation
                temperature=0.5
            )
            return feedback.strip()
        except:
            # Fallback generic feedback
            weak_areas = ", ".join([dim.replace("_", " ") for dim, _ in weakest_dims])
            return f"Consider improving: {weak_areas}. Try to be more specific, add clear value, and encourage discussion."

    def _generate_approval_message(self, dimension_scores: Dict[str, float]) -> str:
        """
        Generate a positive message when post is approved.

        Args:
            dimension_scores: The dimension scores

        Returns:
            Approval message
        """
        strongest_dim = max(dimension_scores.items(), key=lambda x: x[1])
        return (
            f"Post approved! Strong performance in {strongest_dim[0].replace('_', ' ')} "
            f"({strongest_dim[1]:.0f}/100). Ready for image generation."
        )

    def get_score_breakdown(self, dimension_scores: Dict[str, float]) -> str:
        """
        Create a formatted breakdown of scores.

        Args:
            dimension_scores: Dictionary of dimension scores

        Returns:
            Formatted string with score breakdown
        """
        lines = ["Score Breakdown:"]
        for dimension, score in dimension_scores.items():
            bar_length = int(score / 5)  # 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)
            dimension_name = dimension.replace("_", " ").title()
            lines.append(f"{dimension_name:.<25} {bar} {score:.0f}/100")

        return "\n".join(lines)
