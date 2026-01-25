"""
Post Generator Agent: Generates LinkedIn posts using OpenAI and Claude.
"""

from typing import Dict, List, Optional
from utils.llm_clients import call_openai, call_anthropic
import config


class PostGeneratorAgent:
    """
    Agent that generates LinkedIn post variations using multiple LLMs.
    """

    def __init__(self):
        self.name = "PostGeneratorAgent"
        self.post_styles = [
            "educational",
            "opinion-based",
            "storytelling"
        ]

    def generate_posts(
        self,
        topic: Dict[str, str],
        profile_data: Dict[str, str],
        feedback: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate multiple LinkedIn post variations.

        Args:
            topic: Dictionary with topic, description, and rationale
            profile_data: User's profile information
            feedback: Optional feedback from engagement prediction for regeneration

        Returns:
            List of post dictionaries with text, style, and provider
        """
        posts = []

        # Generate posts from OpenAI
        for style in self.post_styles:
            post_text = self._generate_post_openai(
                topic, profile_data, style, feedback
            )
            posts.append({
                "text": post_text,
                "style": style,
                "provider": "openai",
                "topic": topic["topic"]
            })

        # Generate posts from Anthropic
        for style in self.post_styles:
            post_text = self._generate_post_anthropic(
                topic, profile_data, style, feedback
            )
            posts.append({
                "text": post_text,
                "style": style,
                "provider": "anthropic",
                "topic": topic["topic"]
            })

        return posts

    def _generate_post_openai(
        self,
        topic: Dict[str, str],
        profile_data: Dict[str, str],
        style: str,
        feedback: Optional[str] = None
    ) -> str:
        """
        Generate a single post using OpenAI.

        Args:
            topic: Topic information
            profile_data: User profile
            style: Post style (educational, opinion-based, storytelling)
            feedback: Optional improvement feedback

        Returns:
            Generated post text
        """
        system_prompt = f"""You are an expert LinkedIn content creator.
Generate a professional, engaging LinkedIn post in a {style} style.

Requirements:
- Length: {config.POST_MIN_WORDS}-{config.POST_MAX_WORDS} words
- Include 3-5 relevant hashtags at the end
- Make it authentic and valuable
- Use proper LinkedIn formatting (short paragraphs, line breaks)
- Match the tone to the user's professional level

Style Guidelines:
- Educational: Share insights, tips, or lessons learned
- Opinion-based: Take a stance on the topic with supporting arguments
- Storytelling: Use narrative to illustrate points, include personal experience if relevant"""

        user_prompt = f"""Topic: {topic['topic']}
Description: {topic['description']}
Rationale for this user: {topic.get('rationale', 'Relevant to their background')}

User Profile:
- Headline: {profile_data.get('headline', 'Professional')}
- Background: {profile_data.get('about', 'Experienced professional')[:200]}
- Skills: {', '.join(profile_data.get('skills', [])[:5])}"""

        if feedback:
            user_prompt += f"\n\nIMPORTANT - Previous Feedback to Address:\n{feedback}"

        user_prompt += "\n\nGenerate an engaging LinkedIn post on this topic:"

        try:
            post = call_openai(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.8  # Higher creativity for content generation
            )
            return post.strip()
        except Exception as e:
            return f"[Error generating post with OpenAI: {str(e)}]"

    def _generate_post_anthropic(
        self,
        topic: Dict[str, str],
        profile_data: Dict[str, str],
        style: str,
        feedback: Optional[str] = None
    ) -> str:
        """
        Generate a single post using Anthropic Claude.

        Args:
            topic: Topic information
            profile_data: User profile
            style: Post style
            feedback: Optional improvement feedback

        Returns:
            Generated post text
        """
        system_prompt = f"""You are an expert LinkedIn content strategist.
Create a compelling LinkedIn post in a {style} style that will engage the author's network.

Guidelines:
- Word count: {config.POST_MIN_WORDS}-{config.POST_MAX_WORDS} words
- Add 3-5 strategic hashtags
- Be authentic and provide real value
- Use LinkedIn best practices (hooks, white space, clear structure)
- Align tone with user's professional brand

{style.title()} Style:
- Educational: Focus on actionable insights and learning
- Opinion-based: Present a clear viewpoint with evidence
- Storytelling: Weave a narrative that connects emotionally"""

        user_prompt = f"""Create a post about: {topic['topic']}

Context: {topic['description']}
Why relevant: {topic.get('rationale', 'Aligns with their expertise')}

Author's Profile:
- Professional Identity: {profile_data.get('headline', 'Industry Professional')}
- Background: {profile_data.get('about', 'Experienced in their field')[:200]}
- Key Skills: {', '.join(profile_data.get('skills', [])[:5])}"""

        if feedback:
            user_prompt += f"\n\nCRITICAL - Address this feedback:\n{feedback}"

        user_prompt += "\n\nWrite the LinkedIn post now:"

        try:
            post = call_anthropic(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.8
            )
            return post.strip()
        except Exception as e:
            return f"[Error generating post with Anthropic: {str(e)}]"

    def select_best_post(self, posts: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Select the best post from generated variations (for now, returns first valid one).
        In a production system, this could use more sophisticated selection logic.

        Args:
            posts: List of generated posts

        Returns:
            The selected post dictionary
        """
        # Filter out error posts
        valid_posts = [p for p in posts if not p["text"].startswith("[Error")]

        if not valid_posts:
            raise Exception("No valid posts were generated")

        # For now, return the first valid post
        # Could be enhanced with additional scoring logic
        return valid_posts[0]
