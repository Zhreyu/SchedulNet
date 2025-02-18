"""Configuration for LLM agents."""

import os

# GitHub API configuration
GITHUB_API_KEY = "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"
GITHUB_API_URL = "https://api.github.com/graphql"

# OpenAI configuration
OPENAI_API_KEY = GITHUB_API_KEY  # Using GitHub token for OpenAI
OPENAI_MODEL = "gpt-4"
OPENAI_MAX_TOKENS = 1000
OPENAI_TEMPERATURE = 0.7

# System configuration
DEBUG = True
LOG_LEVEL = "INFO"

def setup_environment():
    """Setup environment variables for APIs."""
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["GITHUB_TOKEN"] = GITHUB_API_KEY
    
    # Initialize logging
    import logging
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
