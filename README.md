# Qwen3-TTS 1.7B CustomVoice

Text-to-speech generation with `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.

This project is configured to run on:
- Apple Silicon (`mps`)
- NVIDIA GPU (`cuda`)
- CPU fallback

It also supports a **local model cache workflow**:
- download once
- run offline from local files

## 1) Create environment

**Python 3.10+ is required** (qwen-tts depends on `accelerate`, which needs 3.10+). If your system `python3` is 3.9, use one of these:

**Option A — Anaconda Python (if you have it):**
```bash
deactivate  # if already in a venv
rm -rf .venv
/opt/anaconda3/bin/python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B — Homebrew Python 3.11:**
```bash
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or use the helper script (picks first available Python 3.10+):

```bash
./setup.sh
source .venv/bin/activate
```

## 2) (Optional) Pre-download model to local folder

If you want guaranteed local/offline availability:

```bash
python scripts/predownload_model.py
```

This stores model files under `models/Qwen3-TTS-12Hz-1.7B-CustomVoice`.

## 3) Generate audio

```bash
python scripts/generate_tts.py
```

Output is saved to:

`output/output.wav`

## Useful options

```bash
# Force local-only load (no network)
python scripts/generate_tts.py --offline

# Use a custom text file
python scripts/generate_tts.py --text-file input.txt

# Custom output
python scripts/generate_tts.py --output output/my_take.wav
```
