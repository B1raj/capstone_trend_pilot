"""
LLM client wrappers for OpenAI and Anthropic APIs.
"""

import time
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import config


def call_openai(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_retries: int = 3,
    temperature: float = 0.7
) -> str:
    """
    Call OpenAI API with retry logic.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        max_retries: Maximum number of retry attempts
        temperature: Temperature for generation (0-2)

    Returns:
        The generated text response

    Raises:
        Exception: If all retries fail
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY
    )

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"OpenAI API call failed after {max_retries} attempts: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff

    return ""


def call_anthropic(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_retries: int = 3,
    temperature: float = 0.7
) -> str:
    """
    Call Anthropic API with retry logic.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        max_retries: Maximum number of retry attempts
        temperature: Temperature for generation (0-1)

    Returns:
        The generated text response

    Raises:
        Exception: If all retries fail
    """
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    llm = ChatAnthropic(
        model=config.ANTHROPIC_MODEL,
        temperature=temperature,
        api_key=config.ANTHROPIC_API_KEY
    )

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Anthropic API call failed after {max_retries} attempts: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff

    return ""


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: str = "openai",
    temperature: float = 0.7
) -> str:
    """
    Generic LLM call that routes to the appropriate provider.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        provider: "openai" or "anthropic"
        temperature: Temperature for generation

    Returns:
        The generated text response
    """
    if provider.lower() == "openai":
        return call_openai(prompt, system_prompt, temperature=temperature)
    elif provider.lower() == "anthropic":
        return call_anthropic(prompt, system_prompt, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'openai' or 'anthropic'")
