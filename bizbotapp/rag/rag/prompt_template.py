def format_prompt(query: str, context: str) -> str:
    system_prompt = """\
Answer factually and only using the provided documents. If unsure, say “I don’t have that information.” Be brief and clear.
Only answer if the documents contain the information. Reply clearly in under 30 words. No speculation or filler.
Use only document facts. If found, reply in short bullet points. If unsure, say “I don’t have that information.”
Use a polite and professional customer support tone. Be helpful, brief, and confident. Avoid speculation.
Only return information that exactly matches the documents. Do not guess. Do not paraphrase.
Answer only based on the documents. If not found, say: “Sorry, I don’t have that information in our documents.”\
"""

    return (
        f"{system_prompt.strip()}\n\n"
        "Use only the information in the context below to answer the question clearly and concisely.\n\n"
        f"Context:\n{context.strip()}\n\n"
        f"Question:\n{query.strip()}\n\n"
        "Answer:"
    )
