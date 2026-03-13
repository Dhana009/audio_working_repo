#!/usr/bin/env python3
"""
Qwen3-TTS 1.7B CustomVoice generation script.
Runs on CUDA, MPS (Apple Silicon), or CPU.

Local-cache-first behavior:
- If local model directory exists, load from it.
- Otherwise load from Hugging Face repo.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
LOCAL_MODEL_DIR = Path("models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_OUTPUT = Path("output/output.wav")

DEFAULT_TEXT = (
    "Here is something worth thinking about.\n"
    "What if the secret to long-term success... is almost embarrassingly small?\n"
    "That is exactly what James Clear argues in this chapter.\n"
    "Most of us believe that real change requires big, dramatic action. "
    "A complete overhaul. A life-changing decision.\n"
)

DEFAULT_INSTRUCT = (
    "Explain the idea in a normal conversational tone. "
    "Speak at a natural everyday speaking speed, similar to how people talk in a casual discussion. "
    "Do not exaggerate words or emotions. "
    "Avoid sounding dramatic and avoid speaking too slowly."
)

_script_start = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _script_start
    print(f"[{elapsed:6.1f}s] {msg}", flush=True)


def timer(label: str):
    class _Timer:
        def __enter__(self):
            self._t = time.time()
            return self

        def __exit__(self, *_):
            secs = int(time.time() - self._t)
            mins, s = divmod(secs, 60)
            log(f"  ✓ {label} done in {mins}m {s:02d}s")

    return _Timer()


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def estimate_runtime(device: str, text: str, instruct: str) -> None:
    secs_per_word = 1.1 if device == "mps" else 3.5
    text_words = len(text.split())
    instruct_words = len(instruct.split())
    total_words = text_words + instruct_words

    est_secs = total_words * secs_per_word
    est_low = int(est_secs * 0.7)
    est_high = int(est_secs * 1.4)
    lo_m, lo_s = divmod(est_low, 60)
    hi_m, hi_s = divmod(est_high, 60)

    speech_wpm = 110 if instruct_words > 60 else 130
    audio_secs = int((text_words / speech_wpm) * 60)
    audio_m, audio_s = divmod(audio_secs, 60)

    log(f"Text: {text_words} words + {instruct_words} instruct words = {total_words} total")
    log(f"Estimated audio output : ~{audio_m}m {audio_s:02d}s")
    log(f"Estimated generation   : {lo_m}m {lo_s:02d}s - {hi_m}m {hi_s:02d}s (on {device})")
    log("-" * 60)


def _flash_attn_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("flash_attn") is not None


def build_model_kwargs(device: str) -> dict:
    kwargs = {
        "device_map": device,
        "dtype": torch.bfloat16 if device != "cpu" else torch.float32,
    }
    if device.startswith("cuda") and torch.cuda.is_bf16_supported() and _flash_attn_available():
        kwargs["attn_implementation"] = "flash_attention_2"
    elif device.startswith("cuda") or device == "mps":
        kwargs["attn_implementation"] = "sdpa"
    else:
        kwargs["attn_implementation"] = "eager"
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate speech with Qwen3-TTS 1.7B CustomVoice")
    parser.add_argument("--text-file", type=Path, help="Text input file (utf-8). If omitted, built-in sample text is used.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output wav path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--speaker", default="Ryan", help="Speaker name")
    parser.add_argument("--language", default="English", help="Language name")
    parser.add_argument("--offline", action="store_true", help="Force local-only model loading")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier applied after generation (default: 1.0 = normal, "
             "1.1 = 10%% faster, 0.9 = 10%% slower). Uses librosa time-stretch, pitch is preserved.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    # Reduce CPU bloat before tensor operations.
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    args = parse_args()
    text = DEFAULT_TEXT if not args.text_file else args.text_file.read_text(encoding="utf-8")
    instruct = DEFAULT_INSTRUCT

    device = choose_device()
    log(f"Device: {device}")
    model_kwargs = build_model_kwargs(device)

    model_path = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_REPO
    if LOCAL_MODEL_DIR.exists():
        log(f"Using local model files: {LOCAL_MODEL_DIR}")
    else:
        log(f"Local model folder not found. Will load from repo: {MODEL_REPO}")

    if args.offline:
        if not LOCAL_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"--offline requested but local model folder missing: {LOCAL_MODEL_DIR}"
            )
        os.environ["HF_HUB_OFFLINE"] = "1"
        model_kwargs["local_files_only"] = True
        log("Offline mode enabled (HF_HUB_OFFLINE=1, local_files_only=True)")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log("Step 1/3 - Loading model...")
    with timer("Model load"):
        model = Qwen3TTSModel.from_pretrained(model_path, **model_kwargs)

    estimate_runtime(device=device, text=text, instruct=instruct)

    log("Step 2/3 - Generating speech...")
    with timer("Speech generation"):
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=args.language,
            speaker=args.speaker,
            instruct=instruct,
        )
        if device == "mps":
            torch.mps.synchronize()

    audio = wavs[0]
    if args.speed != 1.0:
        log(f"Step 3/4 - Adjusting speed x{args.speed} (pitch preserved)...")
        with timer("Speed adjust"):
            audio = librosa.effects.time_stretch(
                np.array(audio, dtype=np.float32), rate=args.speed
            )

    log(f"Step {'4' if args.speed != 1.0 else '3'}/{'4' if args.speed != 1.0 else '3'} - Saving audio file...")
    with timer("Save"):
        sf.write(str(args.output), audio, sr)

    total = int(time.time() - _script_start)
    m, s = divmod(total, 60)
    log(f"All done! Total time: {m}m {s:02d}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
