import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "smolvlm_radiology"
DATASET_ID = "unsloth/Radiology_mini"
DATASET_SPLIT = "test"
INSTRUCTION = (
    "You are an expert radiographer and best amongst all. Describe accurately what you see in this image."
)
MAX_NEW_TOKENS = 128


def main() -> None:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to("cuda")

    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    image = dataset[0]["image"].convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    result = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    print(result.strip())


if __name__ == "__main__":
    main()
