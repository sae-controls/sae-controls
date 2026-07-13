"""Prompt formatting.

Do not modify: every reported number depends on this exact wrapper:

    "{SYSTEM_PREAMBLE}\\n\\n{question}"
    → wrapped as a single user-turn message
    → chat-templated with add_generation_prompt=True
"""
from __future__ import annotations

from .config import SYSTEM_PREAMBLE


def build_prompt(tokenizer, question: str) -> str:
    """Format a question into the chat-templated prompt the model expects."""
    combined = f"{SYSTEM_PREAMBLE}\n\n{question}"
    messages = [{"role": "user", "content": combined}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
