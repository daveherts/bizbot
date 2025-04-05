# rag/admin_gui.py

import gradio as gr
from rag.bot import BizBot

bot = BizBot()

def chat_with_bizbot(query):
    return bot.answer(query)

with gr.Blocks() as admin_ui:
    gr.Markdown("## BizBot RAG Chatbot")
    gr.Markdown("RAG-powered local chatbot using LLM + ChromaDB + fine-tuned BizBot")

    with gr.Row():
        query_input = gr.Textbox(label="Ask BizBot a question", placeholder="e.g. What are your support hours")

    with gr.Row():
        clear_btn = gr.Button("Clear")
        submit_btn = gr.Button("Submit")

    output_text = gr.Textbox(label="Output", lines=8)

    submit_btn.click(
        fn=chat_with_bizbot,
        inputs=[query_input],
        outputs=output_text
    )

    clear_btn.click(fn=lambda: "", outputs=output_text)

if __name__ == "__main__":
    print("🚀 Launching Gradio UI...")
    admin_ui.launch(server_port=7860)
