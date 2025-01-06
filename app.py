import gradio as gr

from train import main


def train_model():
    main()
    return "Training completed!"


# Create Gradio interface
interface = gr.Interface(
    fn=train_model,
    inputs=[],
    outputs="text",
    title="Pugmark Training",
    description="Click to start training the model",
)

# Launch the app
interface.launch()
