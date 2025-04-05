import gradio as gr
from rag.bot import BizBot
from rag.telemetry import log_query

bot = BizBot()

def chat_with_bizbot(query):
    answer = bot.answer(query)
    log_query(query, "[context handled internally]", answer)
    return answer

gui = gr.Interface(
    fn=chat_with_bizbot,
    inputs=gr.Textbox(label="Ask BizBot a question", lines=2),
    outputs="text",
    title="BizBot RAG Chatbot",
    description="RAG-powered local chatbot using LLM + ChromaDB."
)

gui.launch(share=True)