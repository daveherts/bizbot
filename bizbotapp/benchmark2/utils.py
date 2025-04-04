def clean_response(text: str) -> str:
    """
    Removes any system prompts or prefixes and returns the assistant's clean reply.
    """
    for prefix in ["Assistant:", "Response:", "Answer:"]:
        if prefix in text:
            return text.split(prefix)[-1].strip()
    return text.strip()
