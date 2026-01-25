"""
Image Generator Agent: Creates mermaid diagrams for LinkedIn posts.
"""

from typing import Optional, Tuple
from utils.llm_clients import call_llm
import config


class ImageGeneratorAgent:
    """
    Agent that generates mermaid diagrams to accompany LinkedIn posts.
    """

    def __init__(self):
        self.name = "ImageGeneratorAgent"

    def generate_diagram(self, post_text: str, topic: str) -> Tuple[Optional[str], str]:
        """
        Generate a mermaid diagram for the post if appropriate.

        Args:
            post_text: The LinkedIn post text
            topic: The topic of the post

        Returns:
            Tuple of (mermaid_code, diagram_type) or (None, "none") if not suitable
        """
        # First, determine if the post would benefit from a diagram
        needs_diagram, diagram_type = self._should_create_diagram(post_text, topic)

        if not needs_diagram:
            return None, "none"

        # Generate the mermaid diagram
        mermaid_code = self._create_mermaid_diagram(
            post_text, topic, diagram_type
        )

        return mermaid_code, diagram_type

    def _should_create_diagram(self, post_text: str, topic: str) -> Tuple[bool, str]:
        """
        Determine if a diagram would enhance the post and what type.

        Args:
            post_text: The post content
            topic: The topic

        Returns:
            Tuple of (should_create, diagram_type)
        """
        system_prompt = """You are an expert at visual content strategy for LinkedIn.
Determine if a visual would enhance this post and which type would be best.

Respond with JSON in this format:
{
    "needs_visual": true/false,
    "visual_type": "flowchart" | "sequence" | "mindmap" | "architecture" | "image" | "none",
    "reason": "brief explanation"
}

Guidelines:
- Flowchart: For processes, decision trees, workflows, step-by-step guides
- Sequence: For interactions, API flows, communication patterns, system interactions
- Mindmap: For concepts, relationships, categorization, brainstorming
- Architecture: For system designs, component relationships, technical architecture
- Image: For creative concepts, metaphors, abstract ideas that need illustration
- None: For posts that are opinion-based, storytelling, or don't need visualization

Prefer mermaid diagrams (flowchart, sequence, mindmap, architecture) for technical content.
Only suggest "image" for creative, non-technical content that would benefit from illustration."""

        user_prompt = f"""Topic: {topic}

Post Content:
{post_text}

Would a mermaid diagram enhance this post? If yes, which type?"""

        try:
            response = call_llm(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider="openai",
                temperature=0.3
            )

            import json
            result = json.loads(response)

            needs_visual = result.get("needs_visual", False)
            visual_type = result.get("visual_type", "none")

            if visual_type == "none":
                needs_visual = False

            return needs_visual, visual_type

        except:
            # Default: create a flowchart for technical topics
            technical_keywords = [
                "process", "workflow", "architecture", "system", "pipeline",
                "api", "integration", "flow", "cycle", "framework"
            ]
            is_technical = any(
                keyword in post_text.lower() or keyword in topic.lower()
                for keyword in technical_keywords
            )

            if is_technical:
                return True, "architecture"
            return False, "none"

    def _create_mermaid_diagram(
        self, post_text: str, topic: str, diagram_type: str
    ) -> str:
        """
        Generate mermaid diagram code.

        Args:
            post_text: The post content
            topic: The topic
            diagram_type: Type of diagram to create

        Returns:
            Mermaid diagram code
        """
        system_prompt = f"""You are an expert at creating mermaid diagrams for technical content.
Generate a clean, professional mermaid {diagram_type} diagram that visualizes the key concepts from the post.

Requirements:
- Use valid mermaid.js syntax
- Keep it simple and clear (5-10 nodes maximum)
- Use descriptive, concise labels (avoid overly long text)
- Make it visually appealing and easy to understand
- Ensure it's directly relevant to the post content
- For architecture diagrams, show clear component relationships and data flow
- Use proper styling and grouping where appropriate

Return ONLY the mermaid code, starting with the diagram type declaration.
Do not include markdown code fences (```), comments, or explanations.
The diagram should be ready to render as-is."""

        # Provide examples based on diagram type
        examples = {
            "flowchart": """Example format:
flowchart TD
    A[Start] --> B[Process]
    B --> C{Decision}
    C -->|Yes| D[Action]
    C -->|No| E[Alternative]""",

            "sequence": """Example format:
sequenceDiagram
    participant A as User
    participant B as System
    A->>B: Request
    B->>A: Response""",

            "mindmap": """Example format:
mindmap
  root((Central Concept))
    Branch1
      SubBranch1
      SubBranch2
    Branch2
      SubBranch3""",

            "architecture": """Example format:
graph TB
    subgraph Frontend
        A[UI Layer]
        B[State Management]
    end
    subgraph Backend
        C[API Gateway]
        D[Business Logic]
    end
    A --> C
    B --> C
    C --> D"""
        }

        user_prompt = f"""Topic: {topic}

Post Content:
{post_text}

Create a {diagram_type} mermaid diagram that visualizes the main concepts.

{examples.get(diagram_type, '')}

Generate the mermaid code now:"""

        try:
            mermaid_code = call_llm(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider="anthropic",  # Claude is good at structured output
                temperature=0.5
            )

            # Clean up the response
            mermaid_code = mermaid_code.strip()

            # Remove markdown code fences if present
            if mermaid_code.startswith("```"):
                lines = mermaid_code.split("\n")
                mermaid_code = "\n".join(lines[1:-1])

            # Validate basic syntax
            if not self._validate_mermaid_syntax(mermaid_code, diagram_type):
                # Return a fallback diagram
                return self._create_fallback_diagram(topic, diagram_type)

            return mermaid_code

        except Exception as e:
            return self._create_fallback_diagram(topic, diagram_type)

    def _validate_mermaid_syntax(self, code: str, expected_type: str) -> bool:
        """
        Basic validation of mermaid syntax.

        Args:
            code: The mermaid code
            expected_type: Expected diagram type

        Returns:
            True if syntax looks valid
        """
        code_lower = code.lower()

        # Check if it starts with a diagram declaration
        valid_starts = ["flowchart", "graph", "sequencediagram", "mindmap"]
        has_valid_start = any(code_lower.startswith(start) for start in valid_starts)

        # Check if it has content (not just the declaration)
        has_content = len(code.split("\n")) > 1

        return has_valid_start and has_content

    def _create_fallback_diagram(self, topic: str, diagram_type: str) -> str:
        """
        Create a simple fallback diagram when generation fails.

        Args:
            topic: The topic
            diagram_type: Type of diagram

        Returns:
            Simple mermaid diagram code
        """
        if diagram_type == "mindmap":
            return f"""mindmap
  root(({topic}))
    Key Concepts
      Concept 1
      Concept 2
    Applications
      Use Case 1
      Use Case 2
    Future Trends
      Trend 1
      Trend 2"""

        elif diagram_type == "sequence":
            return f"""sequenceDiagram
    participant User
    participant System
    User->>System: Request
    System->>System: Process
    System->>User: Response"""

        else:  # flowchart or architecture
            return f"""flowchart TD
    A[{topic}] --> B[Core Concept]
    B --> C[Implementation]
    C --> D[Benefits]
    B --> E[Challenges]
    E --> F[Solutions]"""
