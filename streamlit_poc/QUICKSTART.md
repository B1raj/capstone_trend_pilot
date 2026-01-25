# Quick Start Guide

## ✅ Configuration Complete

The application is now configured to use API keys from your `.zshrc` shell environment.

### Environment Variables Used
- `OPENAI_API_KEY` - From your .zshrc
- `ANTHROPIC_API_KEY` - From your .zshrc

## Starting the Application

### Method 1: Using the Start Script (Recommended)
```bash
./start_app.sh
```

This script:
1. Sources your `.zshrc` to load environment variables
2. Starts Streamlit with those variables
3. Uses port 8502 by default

To use a different port:
```bash
./start_app.sh 8503
```

### Method 2: Manual Start
```bash
source ~/.zshrc
streamlit run app.py
```

## Accessing the Application

Once started, access the app at:
- **Local**: http://localhost:8501
- **Network**: http://192.168.1.137:8501
- **External**: http://151.192.165.83:8501

## How It Works

The `config.py` file now:
1. **First** checks for shell environment variables (from .zshrc)
2. **Then** falls back to .env file if shell variables aren't set
3. Uses `load_dotenv(override=False)` to prioritize shell variables

```python
# config.py prioritizes .zshrc environment variables
load_dotenv(override=False)  # Won't override shell variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
```

## Verifying API Keys

To check if your API keys are loaded:
```bash
source ~/.zshrc
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Anthropic:', bool(os.getenv('ANTHROPIC_API_KEY')))"
```

## Stopping the Application

```bash
# Find and kill Streamlit process
pkill -f streamlit

# Or kill by port
lsof -ti:8501 | xargs kill
```

## Current Status

✅ Streamlit running on port 8501
✅ Using OPENAI_API_KEY from .zshrc
✅ Using ANTHROPIC_API_KEY from .zshrc
✅ Ready to generate LinkedIn posts!

---

**Last Updated**: 2026-01-10
