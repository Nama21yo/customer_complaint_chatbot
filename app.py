import gradio as gr
from rag_pipeline import rag_pipeline

def chatbot(question):
    answer, chunks, _ = rag_pipeline(question)
    sources = "\n".join([f"Source {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
    return answer, sources

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
    title="Complaint Answering Chatbot",
    description="Ask questions about customer complaints."
)

iface.launch()
