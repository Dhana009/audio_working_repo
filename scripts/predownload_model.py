#!/usr/bin/env python3
"""Download Qwen3-TTS model files into local project folder."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
LOCAL_MODEL_DIR = Path("models/Qwen3-TTS-12Hz-1.7B-CustomVoice")


def main() -> int:
    LOCAL_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from {MODEL_REPO} ...")
    path = snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(LOCAL_MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Model ready at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
