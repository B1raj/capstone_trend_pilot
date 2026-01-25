# AI-Powered LinkedIn Post Generator

A multi-agent system that automatically generates engaging LinkedIn posts with visual diagrams based on trending topics matched to your professional profile.

## Features

- **Trend Analysis**: Identifies trending technology topics relevant to your background
- **Multi-LLM Generation**: Creates post variations using both OpenAI GPT-4 and Anthropic Claude
- **Engagement Prediction**: LLM-based scoring with automatic regeneration for low-quality posts
- **Visual Diagrams**: Generates mermaid diagrams (flowcharts, sequences, mindmaps) to accompany posts
- **Smart Orchestration**: LangGraph-based workflow with conditional loops and state management
- **Beautiful UI**: Streamlit interface with real-time progress tracking

## Architecture

```mermaid
graph TD
    A[User Input: LinkedIn URL] --> B[Profile Scraper]
    B --> C[Trend Finder Agent]
    C --> D[Post Generator Agent]
    D --> E[Engagement Predictor]
    E -->|Score < 70| D
    E -->|Score >= 70| F[Image Generator Agent]
    F --> G[Final Output]

    style A fill:#e1f5ff
    style G fill:#d4edda
    style E fill:#fff3cd
```

## Multi-Agent System

### 1. Trend Finder Agent
- Fetches mock trending technology topics
- Compares trends against LinkedIn profile using LLM
- Scores relevance (0-100) for each trend
- Returns top 3 most relevant topics

### 2. Post Generator Agent
- Generates 3 variations from OpenAI (educational, opinion-based, storytelling)
- Generates 3 variations from Claude (same styles)
- Incorporates feedback from engagement predictor on regeneration
- Maintains 150-300 word count with hashtags

### 3. Engagement Predictor Agent
- Scores posts on 6 dimensions using LLM:
  - Clarity (20% weight)
  - Relevance (25% weight)
  - Call to Action (15% weight)
  - Professional Tone (15% weight)
  - Value Proposition (20% weight)
  - Formatting (5% weight)
- Triggers regeneration if overall score < 70
- Provides detailed improvement feedback

### 4. Image Generator Agent
- Analyzes post to determine if diagram is needed
- Generates mermaid code for:
  - Flowcharts (processes, workflows)
  - Sequence diagrams (interactions, APIs)
  - Mindmaps (concepts, relationships)
  - Architecture diagrams (systems, components)
- Validates syntax and provides fallback diagrams

### 5. Orchestrator
- LangGraph-based state machine
- Coordinates all agents in sequence
- Implements conditional loops (max 3 regeneration attempts)
- Manages state and error handling

## Project Structure

```
├── agents/
│   ├── __init__.py
│   ├── trend_finder.py          # Trend matching agent
│   ├── post_generator.py        # Post creation agent (OpenAI + Claude)
│   ├── engagement_predictor.py  # LLM-based scoring agent
│   ├── image_generator.py       # Mermaid diagram generator
│   └── orchestrator.py          # LangGraph workflow
├── utils/
│   ├── __init__.py
│   ├── linkedin_scraper.py      # Playwright-based scraper
│   ├── mock_trends.py           # Mock trending topics
│   └── llm_clients.py           # OpenAI & Anthropic wrappers
├── app.py                       # Streamlit UI
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Anthropic API key

### Installation

1. **Clone or download this project**

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Playwright browsers**
   ```bash
   playwright install chromium
   ```

5. **Configure API keys**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=sk-your-openai-key-here
   ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
   ```

### Getting API Keys

**OpenAI API Key:**
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key to your `.env` file

**Anthropic API Key:**
1. Go to https://console.anthropic.com/
2. Sign in or create an account
3. Navigate to API Keys
4. Create a new key
5. Copy the key to your `.env` file

## Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the UI

1. **Enter LinkedIn Profile URL**
   - Paste your public LinkedIn profile URL
   - Format: `https://linkedin.com/in/your-username`
   - Note: Profile must be public (not require login)

2. **Click "Generate Post"**
   - Watch the progress as agents work
   - Each step is shown in real-time

3. **Review Results**
   - View engagement score and breakdown
   - Read the generated post
   - Copy the mermaid diagram code

4. **Use Your Content**
   - Copy the post text to LinkedIn
   - Render the mermaid diagram at https://mermaid.live
   - Screenshot the diagram and attach to your post

### Example Workflow

```
Input: https://linkedin.com/in/satyanadella
  ↓
Profile Scraping: Extract name, headline, about, skills
  ↓
Trend Finding: "AI Agents and Autonomous Systems" (relevance: 92/100)
  ↓
Post Generation: 6 variations (3 from GPT-4, 3 from Claude)
  ↓
Engagement Prediction: Score 78/100 → Approved
  ↓
Diagram Generation: Flowchart showing AI agent architecture
  ↓
Output: Post + Mermaid diagram code
```

## Configuration

Edit `config.py` to customize:

```python
# Engagement threshold (0-100)
ENGAGEMENT_SCORE_THRESHOLD = 70

# Maximum regeneration attempts
MAX_REGENERATION_ATTEMPTS = 3

# Post length
POST_MIN_WORDS = 150
POST_MAX_WORDS = 300

# LLM models
OPENAI_MODEL = "gpt-4"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
```

## Troubleshooting

### "Profile is private or requires login"
- Ensure your LinkedIn profile is set to public
- Check Settings → Visibility → Public profile visibility

### "API key not found"
- Verify `.env` file exists in the project root
- Check that API keys are correctly formatted (no extra spaces)
- Restart the Streamlit app after adding keys

### "Playwright timeout"
- Some profiles may take longer to load
- Increase `SCRAPING_TIMEOUT` in `config.py`
- Check your internet connection

### "No posts generated"
- Check API keys are valid and have credits
- Look at error messages in the UI
- Try a different profile URL

### "LLM API errors"
- Verify API keys have sufficient credits
- Check API status pages:
  - OpenAI: https://status.openai.com
  - Anthropic: https://status.anthropic.com

## Cost Considerations

Each complete run makes approximately:
- 8-12 OpenAI API calls (GPT-4)
- 4-6 Anthropic API calls (Claude 3.5 Sonnet)

Estimated cost per run: $0.10 - $0.30 (varies by input/output length)

To reduce costs:
- Lower `MAX_REGENERATION_ATTEMPTS` in config
- Use GPT-3.5-turbo instead of GPT-4 (change in config.py)

## Technical Details

### LangGraph Workflow

The orchestrator uses LangGraph's `StateGraph` to manage workflow:

```python
scrape_profile → find_trends → generate_posts → predict_engagement
                                        ↑              |
                                        |              ↓
                                        |         (conditional)
                                        |              |
                                        |-- regenerate if score < 70
                                        |              |
                                        |              ↓
                                    generate_diagram → END
```

### State Schema

```python
{
    "linkedin_url": str,
    "profile_data": dict,
    "relevant_topics": list,
    "selected_topic": dict,
    "generated_posts": list,
    "selected_post": dict,
    "engagement_score": float,
    "engagement_feedback": str,
    "dimension_scores": dict,
    "diagram_code": str,
    "diagram_type": str,
    "iteration_count": int,
    "error": str,
    "status": str
}
```

## Future Enhancements

Potential improvements:
- Real Twitter/X API integration
- Actual image generation with DALL-E
- Post scheduling to LinkedIn API
- A/B testing framework
- Historical post analysis
- Multi-platform support (Twitter, Medium, etc.)
- Custom LLM fine-tuning
- Analytics dashboard

## Dependencies

Key libraries:
- `streamlit` - Web UI framework
- `langgraph` - Multi-agent orchestration
- `langchain-openai` - OpenAI integration
- `langchain-anthropic` - Claude integration
- `playwright` - Web scraping
- `beautifulsoup4` - HTML parsing
- `python-dotenv` - Environment management

See `requirements.txt` for complete list.

## License

This project is for educational and personal use. Ensure compliance with:
- LinkedIn Terms of Service (web scraping)
- OpenAI Usage Policies
- Anthropic Acceptable Use Policy

## Contributing

This is a capstone project. Feel free to fork and extend!

Suggestions for improvements:
1. Add unit tests for each agent
2. Implement caching for API calls
3. Add user authentication
4. Create a FastAPI backend
5. Deploy to cloud (Streamlit Cloud, AWS, etc.)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review error messages in the UI
3. Check API service status pages
4. Verify environment setup

## Acknowledgments

Built with:
- OpenAI GPT-4
- Anthropic Claude 3.5 Sonnet
- LangChain & LangGraph
- Streamlit

---

**Built as part of an AI/ML Capstone Project**

Generate engaging LinkedIn content with the power of multi-agent AI systems!
