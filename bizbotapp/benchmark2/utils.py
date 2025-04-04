def extract_assistant_reply(output: str, prompt: str = "") -> str:
    """
    Strips 'Assistant:' labels, removes echoed prompt, trims filler phrases.
    """
    # Remove assistant label if present
    if "Assistant:" in output:
        output = output.split("Assistant:")[-1]

    if "User:" in output:
        output = output.split("User:")[-1]

    output = output.strip()

    # Remove echoed question if present
    if prompt and output.lower().startswith(prompt.lower()):
        output = output[len(prompt):].strip()

    # Remove common filler phrases
    fillers = [
        "Sure, I can help with that.",
        "Of course, here's the information.",
        "Certainly.",
        "Let me help you with that.",
        "Here’s what you need to know.",
    ]
    for f in fillers:
        if output.startswith(f):
            output = output[len(f):].strip()

    return output
