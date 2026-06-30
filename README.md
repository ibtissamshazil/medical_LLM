# medical_LLM

This repository contains a CUDA-targeted SmolVLM fine-tuning script for the `unsloth/Radiology_mini` dataset.

## What Was Fixed

The previous script assumed:

- a Linux shell (`export`)
- a working Python install on `PATH`
- CUDA was always available
- an Unsloth/Llama vision checkpoint was the right architecture for this machine
- Hugging Face Hub push should always happen

The script now:

- validates Python, CUDA, package, and VRAM requirements up front
- switches the default model to `HuggingFaceTB/SmolVLM-256M-Instruct`
- uses the official Transformers SmolVLM CUDA path instead of the Unsloth `FastVisionModel` path
- uses a smaller default training footprint for an 8 GB GPU
- saves locally by default and only pushes when `--push-to-hub` is set
- accepts Hugging Face auth from `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, or `--hf-token`

## Observed Local Constraints

On the machine this repo was checked on:

- `python` on `PATH` is a Windows Store stub, not a real interpreter
- the NVIDIA GPU has 8 GB VRAM
- the previous default 11B model was too large for a safe default fine-tune on that GPU

## Setup

1. Install a real Python 3.10+ interpreter and ensure `python --version` works in PowerShell.
2. Install the project dependencies:

```powershell
C:\Users\ibtis\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. If you want to push to Hugging Face Hub, set your token in PowerShell:

```powershell
$env:HF_TOKEN = "your_huggingface_token"
```

## Run

Local training on the CUDA GPU without Hub push:

```powershell
C:\Users\ibtis\.venv\Scripts\python.exe main.py --skip-sample-generation
```

Use a larger SmolVLM checkpoint only if your GPU has the VRAM for it:

```powershell
C:\Users\ibtis\.venv\Scripts\python.exe main.py --model-id HuggingFaceTB/SmolVLM-500M-Instruct --skip-sample-generation
```

Push to the Hub only when you are ready:

```powershell
C:\Users\ibtis\.venv\Scripts\python.exe main.py --push-to-hub --hub-model-id your-username/your-model
```

Local app for testing the fine-tuned model:

```powershell
C:\Users\ibtis\.venv\Scripts\python.exe app.py
```

Then open `http://127.0.0.1:7860` if the browser does not open automatically.

## Notes

- The default model is now `HuggingFaceTB/SmolVLM-256M-Instruct`, which is a much better fit for the current 8 GB GPU than the previous 11B checkpoint.
- The script prefers `flash_attention_2` on CUDA when available and otherwise falls back to PyTorch `sdpa`, so it still runs on CUDA without requiring the flash-attn package.
