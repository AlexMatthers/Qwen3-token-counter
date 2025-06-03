import gradio as gr
from transformers import AutoTokenizer

# List of supported Qwen3 models
QWEN_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-4B"
]

# Cache tokenizers to avoid repeated downloads
tokenizer_cache = {}

def count_tokens(model_name, text_input, file_input):
    # Read text from uploaded file if provided
    if file_input is not None:
        text = file_input.read().decode("utf-8")
    else:
        text = text_input

    if not text.strip():
        return 0, []

    # Load tokenizer (with caching)
    if model_name not in tokenizer_cache:
        tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    tokenizer = tokenizer_cache[model_name]

    # Tokenization
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return len(token_ids), tokens

# Gradio UI
gr.Interface(
    fn=count_tokens,
    inputs=[
        gr.Dropdown(choices=QWEN_MODELS, label="Select Qwen Model", value=QWEN_MODELS[0]),
        gr.Textbox(lines=5, label="Input Text (ignored if file is uploaded)"),
        gr.File(label="Upload .txt File (optional)", file_types=[".txt"])
    ],
    outputs=[
        gr.Number(label="Token Count"),
        gr.JSON(label="Tokens")
    ],
    title="Qwen Token Counter",
    description="Select a Qwen model and input text or upload a .txt file to see token count and token list."
).launch()
