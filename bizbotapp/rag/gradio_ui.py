import gradio as gr
from rag.bot import BizBot  # ✅ Corrected import based on folder structure

bot = BizBot()

def chat_with_bizbot(query):
    return bot.answer(query)

with gr.Blocks() as ui:
    gr.Markdown("## BizBot RAG Chatbot")
    gr.Markdown("RAG-powered local chatbot using fine-tuned BizBot")

    query_input = gr.Textbox(label="Ask a question", placeholder="e.g. What is your refund policy?")
    output_box = gr.Textbox(label="Answer", lines=6)

    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear")

    submit_btn.click(fn=chat_with_bizbot, inputs=query_input, outputs=output_box)
    clear_btn.click(fn=lambda: "", outputs=output_box)

if __name__ == "__main__":
    print("🚀 Gradio UI running...")
    ui.launch(server_port=7860)
