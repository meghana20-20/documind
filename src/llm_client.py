"""
llm_client.py
A thin, pluggable wrapper so DocuMind can call OpenAI, Anthropic, or a
local Ollama model interchangeably, based on the LLM_PROVIDER env var.
"""

from __future__ import annotations
import os
import requests


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
        if self.provider == "openai":
            return self._openai(system_prompt, user_prompt, max_tokens)
        elif self.provider == "anthropic":
            return self._anthropic(system_prompt, user_prompt, max_tokens)
        elif self.provider == "ollama":
            return self._ollama(system_prompt, user_prompt, max_tokens)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def _openai(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def _anthropic(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _ollama(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
