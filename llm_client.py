"""
LLM Client — Sends question + context to Claude and returns answer.

Uses Claude Haiku (cheapest, fastest) by default.
Maintains conversation history for multi-turn chat.
"""

import os
from typing import Dict, List

# Load .env file automatically
from dotenv import load_dotenv
load_dotenv()

# Try Claude first
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# NASA Expert System Prompt
# This tells Claude WHO it is and HOW to behave
NASA_SYSTEM_PROMPT = """You are an expert NASA mission analyst 
with deep knowledge of space exploration history.

You have access to official NASA mission transcripts and 
technical documents for Apollo 11, Apollo 13, and Challenger.

Rules you must follow:
1. Base answers STRICTLY on the provided context
2. Always mention which mission/document your answer comes from
3. If context doesn't contain the answer — say so clearly
4. Never make up mission details or technical facts
5. Be precise and helpful for researchers and historians"""


def generate_response(openai_key: str,
                      user_message: str,
                      context: str,
                      conversation_history: List[Dict],
                      model: str = "gpt-3.5-turbo") -> str:
    """
    Generate answer using Claude.
    
    Note: openai_key parameter kept for compatibility with chat.py
    We use ANTHROPIC_API_KEY from .env file instead.
    
    Args:
        openai_key           : ignored (kept for chat.py compatibility)
        user_message         : user's question
        context              : formatted NASA chunks from rag_client
        conversation_history : previous turns in conversation
        model                : ignored, we always use Claude Haiku
    
    Returns:
        Claude's answer as a string
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if CLAUDE_AVAILABLE and anthropic_key:
        return _ask_claude(anthropic_key, user_message,
                          context, conversation_history)

    return "Error: ANTHROPIC_API_KEY not found in .env file"


def _ask_claude(api_key: str,
                user_message: str,
                context: str,
                conversation_history: List[Dict]) -> str:
    """
    Internal function — builds messages and calls Claude API.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Build user message WITH context injected
    user_prompt = f"""Context from NASA Documents:
{context}

---

Question: {user_message}

Answer based on the context above. 
Cite which document/mission your information comes from."""

    # Build conversation history for Claude
    # Keep last 10 turns only — saves tokens + cost
    messages = []
    for turn in conversation_history[-10:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Add current question
    messages.append({"role": "user", "content": user_prompt})

    # Call Claude API
    response = client.messages.create(
        model="claude-haiku-4-5",   # fastest + cheapest Claude
        max_tokens=1024,
        system=NASA_SYSTEM_PROMPT,
        messages=messages
    )

    return response.content[0].text
