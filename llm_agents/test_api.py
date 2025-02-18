"""Test script for LLM API configuration."""

import openai
from .config import setup_environment, OPENAI_MODEL
import logging

logger = logging.getLogger(__name__)

def test_openai_connection():
    """Test OpenAI API connection."""
    try:
        # Setup environment
        setup_environment()
        
        # Test API call
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you hear me?"}
            ]
        )
        
        if response and response.choices:
            logger.info("API Test Successful!")
            logger.info(f"Response: {response.choices[0].message.content}")
            return True
    except Exception as e:
        logger.error(f"API Test Failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_openai_connection()
