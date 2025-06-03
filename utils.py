import tiktoken
from typing import Optional


def truncate_text(
    text: str, model: str = "gpt-4.1", max_tokens: Optional[int] = None
) -> str:
    """
    Truncate text to specified token count using tiktoken

    Args:
        text: Text to be truncated
        model: Model name to use, defaults to "gpt-4"
        max_tokens: Maximum token count, if None then no truncation

    Returns:
        Truncated text
    """
    if not max_tokens:
        return text

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # If encoder not found for specified model, use cl100k_base encoder
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def count_tokens(text: str, model: str = "gpt-4.1") -> int:
    """
    Count the number of tokens in a text using tiktoken

    Args:
        text: Text to count tokens
        model: Model name to use, defaults to "gpt-4"

    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # If encoder not found for specified model, use cl100k_base encoder
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)
