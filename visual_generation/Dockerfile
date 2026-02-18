FROM python:3.10-slim

# Avoid Python buffering and set working directory
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies: Node.js and npm for Mermaid CLI
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Node package manifest and install Mermaid CLI locally
COPY package.json .
RUN npm install --omit=dev

# Copy the rest of the visual_generation source code into the image
COPY . .

# Dedicated output directory (mount this to the host; do NOT mount /app/service)
RUN mkdir -p /app/output

# Point the Python renderer at the locally installed Mermaid CLI binary
ENV MMDC_PATH=/app/node_modules/.bin/mmdc
ENV OUTPUT_DIR=/app/output

# Default environment variables for DashScope.
# IMPORTANT: override DASHSCOPE_API_KEY at runtime with your real key.
ENV DASHSCOPE_API_KEY="" \
    DASHSCOPE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1" \
    DASHSCOPE_CHAT_MODEL="qwen-plus" \
    DASHSCOPE_IMAGE_API_URL="https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation" \
    DASHSCOPE_DEFAULT_SIZE="1024*1024" \
    DASHSCOPE_IMAGE_MODEL="z-image-turbo"

# Entry point: run the unified visual service CLI.
# You can override the prompt by passing an argument to `docker run`.
ENTRYPOINT ["python", "-u", "service/visual_service.py"]

# Default prompt if none is provided
CMD ["Describe a simple login system with frontend, backend, and database"]

