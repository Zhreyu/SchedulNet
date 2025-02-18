"""Base agent class for LLM-based OS optimization."""

from abc import ABC, abstractmethod
import openai
from typing import List, Dict, Any, Optional

class BaseAgent(ABC):
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.conversation_history = []
        
    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    *self.conversation_history,
                    {"role": "user", "content": prompt}
                ]
            )
            reply = response.choices[0].message.content
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""
            
    def _clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        
    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Optimize the given workload or resource."""
        pass
