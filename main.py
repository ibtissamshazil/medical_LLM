import argparse
import importlib.util
import os
import sys
from typing import Any


DEFAULT_INSTRUCTION = (
    "You are an expert radiographer and best amongst all. Describe accurately what you see in this image."
)
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_DATASET_ID = "unsloth/Radiology_mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLM on the radiology mini dataset using a CUDA GPU."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="smolvlm_radiology")
    parser.add_argument("--hub-model-id", default=os.environ.get("HUB_MODEL_ID"))
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--skip-sample-generation",
        action="store_true",
        help="Skip before/after sample generation to reduce runtime.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the trained model and processor to Hugging Face Hub.",
    )
    parser.add_argument(
        "--skip-vram-check",
        action="store_true",
        help="Bypass the conservative VRAM compatibility check.",
    )
    parser.add_argument(
        "--stop-on-train-begin",
        action="store_true",
        help="Exit as soon as the trainer enters the training loop.",
    )
    return parser.parse_args()


def require_python_version() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(
            "Python 3.10 or newer is required. Install a real Python interpreter and "
            "make sure it is on PATH before running this script."
        )


def load_runtime_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "a required package"
        raise SystemExit(
            f"Missing dependency: {missing_name}. Install the packages from "
            "requirements.txt once a working Python environment is available."
        ) from exc

    return (
        torch,
        load_dataset,
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )


def resolve_hf_token(cli_token: str | None) -> str | None:
    return (
        cli_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )


def recommended_vram_gb(model_id: str) -> float:
    upper_model_id = model_id.upper()
    if "2.2B" in upper_model_id:
        return 12.0
    if "500M" in upper_model_id:
        return 8.0
    if "256M" in upper_model_id:
        return 6.0
    return 8.0


def validate_runtime(torch: Any, args: argparse.Namespace) -> tuple[Any, float]:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. This SmolVLM training script is configured to run on "
            "an NVIDIA CUDA GPU."
        )

    gpu_stats = torch.cuda.get_device_properties(0)
    total_vram_gb = gpu_stats.total_memory / 1024 / 1024 / 1024
    required_vram_gb = recommended_vram_gb(args.model_id)

    if not args.skip_vram_check and total_vram_gb < required_vram_gb:
        raise SystemExit(
            f"{args.model_id} is unlikely to fit for training on this machine. "
            f"Available VRAM: {total_vram_gb:.1f} GB. Recommended minimum: "
            f"{required_vram_gb:.1f} GB."
        )

    return gpu_stats, total_vram_gb


def resolve_compute_dtype(torch: Any) -> Any:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def resolve_attention_implementation() -> str:
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def resolve_image_token_id(processor: Any, model: Any) -> int | None:
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is not None:
        return image_token_id

    image_token = "<image>"
    token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    return None if token_id == processor.tokenizer.unk_token_id else token_id


def build_messages(instruction: str, answer: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


def build_collate_fn(processor: Any, image_token_id: int | None, instruction: str) -> Any:
    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        images: list[list[Any]] = []

        for example in examples:
            image = example["image"]
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")

            messages = build_messages(instruction=instruction, answer=example["caption"])
            text = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
            )
            texts.append(text.strip())
            images.append([image])

        batch = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The processor tokenizer does not define a pad or EOS token.")

        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = -100
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def run_sample_generation(
    torch: Any,
    model: Any,
    processor: Any,
    image: Any,
    instruction: str,
    max_new_tokens: int,
) -> None:
    if hasattr(image, "mode") and image.mode != "RGB":
        image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
    generated_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    print(generated_text.strip())


def build_stop_on_train_begin_callback(trainer_callback_cls: Any) -> Any:
    class StopOnTrainBeginCallback(trainer_callback_cls):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            print("Training loop entered. Stopping immediately as requested.")
            control.should_training_stop = True
            return control

    return StopOnTrainBeginCallback()


def configure_model_for_training(model: Any, processor: Any) -> None:
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
        if hasattr(model.config, "text_config") and model.config.text_config is not None:
            model.config.text_config.pad_token_id = pad_token_id

    model.config.use_cache = False
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.use_cache = False


def main() -> None:
    require_python_version()
    args = parse_args()
    hf_token = resolve_hf_token(args.hf_token)

    (
        torch,
        load_dataset,
        auto_model_cls,
        auto_processor_cls,
        trainer_cls,
        trainer_callback_cls,
        training_args_cls,
    ) = load_runtime_dependencies()

    gpu_stats, total_vram_gb = validate_runtime(torch, args)
    compute_dtype = resolve_compute_dtype(torch)
    attention_implementation = resolve_attention_implementation()
    print(
        f"GPU = {gpu_stats.name}. Total VRAM = {total_vram_gb:.1f} GB. "
        f"Attention backend = {attention_implementation}."
    )

    processor = auto_processor_cls.from_pretrained(args.model_id)
    model = auto_model_cls.from_pretrained(
        args.model_id,
        dtype=compute_dtype,
        _attn_implementation=attention_implementation,
    ).to("cuda")
    configure_model_for_training(model, processor)
    model.gradient_checkpointing_enable()

    dataset = load_dataset(args.dataset_id, split="train")
    if len(dataset) == 0:
        raise SystemExit("The dataset split is empty.")

    image_token_id = resolve_image_token_id(processor, model)
    collate_fn = build_collate_fn(
        processor=processor,
        image_token_id=image_token_id,
        instruction=args.instruction,
    )

    if not args.skip_sample_generation:
        print("\nBefore training:\n")
        model.eval()
        run_sample_generation(
            torch=torch,
            model=model,
            processor=processor,
            image=dataset[0]["image"],
            instruction=args.instruction,
            max_new_tokens=args.max_new_tokens,
        )
        torch.cuda.empty_cache()

    training_args = training_args_cls(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.max_steps,
        save_total_limit=1,
        optim="adamw_torch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset,
        callbacks=(
            [build_stop_on_train_begin_callback(trainer_callback_cls)]
            if args.stop_on_train_begin
            else None
        ),
    )

    start_gpu_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
    print(f"{start_gpu_memory:.3f} GB of memory reserved before training.")

    trainer_stats = trainer.train()

    used_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
    used_percentage = used_memory / total_vram_gb * 100
    print(f"{trainer_stats.metrics['train_runtime']:.2f} seconds used for training.")
    print(f"{trainer_stats.metrics['train_runtime'] / 60:.2f} minutes used for training.")
    print(f"Peak reserved memory = {used_memory:.3f} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage:.3f} %.")

    model.config.use_cache = True
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.use_cache = True

    if not args.skip_sample_generation:
        print("\nAfter training:\n")
        model.eval()
        run_sample_generation(
            torch=torch,
            model=model,
            processor=processor,
            image=dataset[0]["image"],
            instruction=args.instruction,
            max_new_tokens=args.max_new_tokens,
        )

    trainer.save_model(args.model_dir)
    processor.save_pretrained(args.model_dir)
    print(f"Saved trained model to {args.model_dir}.")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise SystemExit("--hub-model-id is required when --push-to-hub is set.")
        if not hf_token:
            raise SystemExit(
                "A Hugging Face token is required when --push-to-hub is set. "
                "Use --hf-token or set HF_TOKEN/HUGGINGFACEHUB_API_TOKEN."
            )
        model.push_to_hub(args.hub_model_id, token=hf_token)
        processor.push_to_hub(args.hub_model_id, token=hf_token)
        print(f"Pushed model and processor to {args.hub_model_id}.")


if __name__ == "__main__":
    main()
