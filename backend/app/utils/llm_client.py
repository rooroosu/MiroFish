"""
LLM client wrapper
Unified OpenAI-compatible API calls
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config
from ..model_routing import is_placeholder_secret, qwen_fallback_models


class LLMClient:
    """LLM client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.fallback_models = qwen_fallback_models()

        if is_placeholder_secret(self.api_key):
            raise ValueError("LLM_API_KEY is not configured")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send a chat request.

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            response_format: Response format (e.g. JSON mode)

        Returns:
            Model response text
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        models_to_try = [self.model]
        models_to_try.extend(model for model in self.fallback_models if model not in models_to_try)
        last_error = None

        for model_name in models_to_try:
            kwargs["model"] = model_name
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content is None:
                    last_error = ValueError(f"{model_name} returned empty content")
                    continue
                # Some models (e.g. MiniMax M2.5) include <think> blocks in content that must be removed
                content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                if not content:
                    last_error = ValueError(f"{model_name} returned only think-tags or whitespace")
                    continue
                return content
            except Exception as exc:
                last_error = exc

        raise last_error
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send a chat request and return parsed JSON.

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Strip markdown code-block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON: {cleaned_response}")
