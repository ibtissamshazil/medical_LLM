import argparse
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_DIR = "smolvlm_radiology"
DEFAULT_INSTRUCTION = (
    "You are an expert radiographer and best amongst all. Describe accurately what you see in this image."
)

_MODEL: Any | None = None
_PROCESSOR: Any | None = None
_MODEL_DIR = DEFAULT_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Gradio demo for the fine-tuned SmolVLM model."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def resolve_dtype() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo.")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(model_dir: str) -> tuple[Any, Any]:
    global _MODEL, _PROCESSOR, _MODEL_DIR

    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory '{model_dir}' was not found. Train the model first."
        )

    _PROCESSOR = AutoProcessor.from_pretrained(model_dir)
    _MODEL = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=resolve_dtype(),
        _attn_implementation="sdpa",
    ).to("cuda")
    _MODEL.eval()
    _MODEL_DIR = model_dir
    return _MODEL, _PROCESSOR


def generate_report(
    image: Image.Image | None,
    instruction: str,
    max_new_tokens: int,
) -> str:
    if image is None:
        return "Upload an image first."

    model, processor = load_model(_MODEL_DIR)

    if image.mode != "RGB":
        image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction.strip() or DEFAULT_INSTRUCTION},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    result = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return result.strip()


def build_demo() -> gr.Interface:
    description = (
        "Upload a radiology image and the fine-tuned SmolVLM model will generate a report."
    )
    return gr.Interface(
        fn=generate_report,
        inputs=[
            gr.Image(type="pil", label="Radiology Image"),
            gr.Textbox(
                label="Instruction",
                value=DEFAULT_INSTRUCTION,
                lines=3,
            ),
            gr.Slider(
                label="Max New Tokens",
                minimum=32,
                maximum=256,
                step=16,
                value=128,
            ),
        ],
        outputs=gr.Textbox(label="Generated Report", lines=12),
        title="SmolVLM Radiology Demo",
        description=description,
        submit_btn="Generate Report",
        clear_btn="Clear",
    )


def main() -> None:
    args = parse_args()
    load_model(args.model_dir)
    demo = build_demo()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
