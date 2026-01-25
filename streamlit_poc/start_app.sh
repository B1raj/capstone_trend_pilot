#!/bin/zsh
# Load environment variables from .zshrc
source ~/.zshrc

# Start Streamlit with environment variables
# Use port from argument or default to 8502
PORT=${1:-8502}
streamlit run app.py --server.headless true --server.port $PORT
