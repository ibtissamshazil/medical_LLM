# medical_LLM

This project fine-tunes `HuggingFaceTB/SmolVLM-256M-Instruct` on the `unsloth/Radiology_mini` dataset, saves the trained model locally, and includes a small Gradio app for testing predictions in the browser.

## About

This repository is a small medical vision-language project focused on radiology image understanding. It trains a compact SmolVLM model to generate text descriptions for medical images, provides a terminal test script for quick evaluation, and includes a local browser app for interactive testing on a Gradio server.

## Requirements

- Windows PowerShell
- Python `3.10+`
- NVIDIA GPU with CUDA-enabled PyTorch
- Around `8 GB` VRAM for the default setup

## 1. Clone and Set Up

Clone the repo, then create a virtual environment outside the project folder if you want to keep the repo clean:

```powershell
python -m venv C:\Users\USER\.venv
```

Install dependencies with the same interpreter you will use to run the project:

```powershell
C:\Users\USER\.venv\Scripts\python.exe -m pip install --upgrade pip
C:\Users\USER\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Quick check:

```powershell
C:\Users\USER\.venv\Scripts\python.exe -c "import gradio, torch; print('setup ok')"
```

Optional: set a Hugging Face token for higher rate limits or model upload:

```powershell
$env:HF_TOKEN = "your_huggingface_token"
```

## 2. Train the Model

Run training with the default SmolVLM checkpoint:

```powershell
C:\Users\USER\.venv\Scripts\python.exe main.py --skip-sample-generation
```

This downloads the base model and dataset on the first run, then saves the trained model to:

```text
smolvlm_radiology/
```

If you want to stop right when training starts for a quick check:

```powershell
C:\Users\USER\.venv\Scripts\python.exe main.py --skip-sample-generation --stop-on-train-begin
```

## 3. Test in the Terminal

After training, run:

```powershell
C:\Users\USER\.venv\Scripts\python.exe test.py
```

This loads `smolvlm_radiology`, takes the first sample from the dataset test split, and prints the generated report in the terminal.

## 4. Run the Local App

The repo includes a local Gradio app in `app.py`.

Start it with:

```powershell
C:\Users\USER\.venv\Scripts\python.exe app.py
```

By default it runs on local server port `7860`:

```text
http://127.0.0.1:7860
```

Use the app to upload an image, keep or edit the radiology instruction, and view the generated output in the browser.

To change the port:

```powershell
C:\Users\USER\.venv\Scripts\python.exe app.py --server-port 7861
```

## Notes

- Large checkpoints and model weights are intentionally ignored by Git.
- `test.py` and `app.py` both expect the trained model folder `smolvlm_radiology/` to exist.
- If you see `ModuleNotFoundError: No module named 'gradio'`, you are running a different Python than the one used for installation. Re-run the exact install command above and launch the app with `C:\Users\USER\.venv\Scripts\python.exe app.py`.
