#!/bin/zsh
# Load environment variables from .zshrc
source ~/.zshrc

# Start Streamlit OAuth app with environment variables
# Use port from argument or default to 8501
PORT=${1:-8501}
streamlit run app_oauth.py --server.headless true --server.port $PORT
