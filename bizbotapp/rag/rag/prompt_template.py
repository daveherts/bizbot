def format_prompt(query: str, context: str = "") -> str:
    """
    Format the input prompt to include retrieved context, if available.
    If no context is retrieved, fall back to a generic answer style.
    """
    if context:
        return f"""
        Answer the following question using only the information from the context below.
        If the answer is not in the context, reply with "I'm sorry, I don't have that information.".

        Context:
        {context}

        Question: {query}
        Answer:
        """
    else:
        return f"""
        Answer the following question as best as you can.

        Question: {query}
        Answer:
        """
