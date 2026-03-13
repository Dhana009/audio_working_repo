#!/usr/bin/env python3
"""
Parler-TTS Mini — Modal GPU runner.

Much faster than Qwen3-TTS: non-autoregressive, generates full audio
in ~15-30s on L4 vs ~218s.  Supports natural-language style descriptions.
Model cached in a Modal Volume after first download.

Usage:
    modal run scripts/generate_tts_parler_modal.py
    modal run scripts/generate_tts_parler_modal.py --speed 1.1
    modal run scripts/generate_tts_parler_modal.py --output output/parler_take.wav
    modal run scripts/generate_tts_parler_modal.py --text-file input.txt
"""

from __future__ import annotations

import modal

# ── Modal app ─────────────────────────────────────────────────────────────────

app = modal.App("parler-tts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.43.0",
        "parler-tts",
        "soundfile",
        "numpy",
        "huggingface_hub>=0.26.0",
    )
)

model_volume = modal.Volume.from_name("parler-tts-model-cache", create_if_missing=True)
VOLUME_PATH = "/model-cache"
MODEL_REPO  = "parler-tts/parler-tts-mini-v1"

# ── Content ───────────────────────────────────────────────────────────────────

DEFAULT_TEXT = (
    "Here is something worth thinking about. "
    "What if the secret to long-term success is almost embarrassingly small? "
    "That is exactly what James Clear argues in this chapter. "
    "Most of us believe that real change requires big, dramatic action. "
    "A complete overhaul. A life-changing decision. "
    "But Clear says the opposite is true. "
    "Real change, the kind that actually lasts, comes from tiny improvements, "
    "repeated every single day. "
    "Think about it this way. "
    "Imagine getting just one percent better every day. "
    "One percent. That is almost nothing. "
    "But over months and years, those tiny gains compound. "
    "They multiply into something you never thought possible. "
    "Habits, Clear says, work exactly like compound interest. "
    "The results are slow at first. Barely noticeable. "
    "And most people quit before they see them. "
    "But the ones who stay consistent? "
    "Eventually, they become unstoppable. "
    "The key lesson here is simple: stop waiting for the big breakthrough. "
    "Focus on getting just a little bit better every day, and trust the process."
)

# Parler-TTS uses a concise single-sentence description, not a multi-line instruct.
# Key phrases it responds to: pace, tone, warmth, recording quality, gender.
DEFAULT_DESCRIPTION = (
    "Ryan speaks with natural warmth and energy, like a podcast host sharing "
    "an insight that genuinely excites him. His pace is steady and conversational, "
    "confident but not rushed, with light emphasis on key ideas. "
    "The recording is high quality, close-mic, with no background noise."
)

# ── Remote GPU function ────────────────────────────────────────────────────────

@app.function(
    gpu="L4",
    image=image,
    volumes={VOLUME_PATH: model_volume},
    timeout=300,
)
def generate_audio(
    text: str,
    description: str,
    speed: float = 1.0,
) -> bytes:
    """Runs on Modal L4 GPU. Returns WAV bytes."""
    import io
    import os
    import time

    import numpy as np
    import soundfile as sf
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    os.environ["HF_HOME"] = VOLUME_PATH

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[Modal] GPU: {gpu_name}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    print(f"[Modal] Model loaded in {time.time() - t0:.1f}s")

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to("cuda")
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    t1 = time.time()
    with torch.inference_mode():
        generation = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_ids,
        )
    audio = generation.cpu().numpy().squeeze().astype(np.float32)
    sr = model.config.sampling_rate
    print(f"[Modal] Audio generated in {time.time() - t1:.1f}s  (sr={sr})")

    if speed != 1.0:
        # Simple resampling for speed change (preserves content, slight pitch shift)
        import scipy.signal
        target_len = int(len(audio) / speed)
        audio = scipy.signal.resample(audio, target_len).astype(np.float32)
        print(f"[Modal] Speed adjusted to {speed}x")

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


# ── Local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    text_file: str = "",
    output: str = "output/parler_output.wav",
    speed: float = 1.0,
):
    """
    Optional CLI flags:
      --text-file   Path to a UTF-8 text file
      --output      Output wav path (default: output/parler_output.wav)
      --speed       Speed multiplier (default: 1.0)
    """
    import time
    from pathlib import Path

    t0 = time.time()

    text = DEFAULT_TEXT
    if text_file:
        text = Path(text_file).read_text(encoding="utf-8")
        print(f"Using text from: {text_file} ({len(text.split())} words)")
    else:
        print(f"Using built-in sample text ({len(text.split())} words)")

    print(f"Sending to Modal L4 GPU (Parler-TTS Mini)...")
    print(f"  Speed: {speed}x")
    print("-" * 60)

    wav_bytes = generate_audio.remote(
        text=text,
        description=DEFAULT_DESCRIPTION,
        speed=speed,
    )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav_bytes)

    elapsed = int(time.time() - t0)
    m, s = divmod(elapsed, 60)
    size_kb = len(wav_bytes) // 1024
    print("-" * 60)
    print(f"Saved {size_kb} KB -> {out}")
    print(f"Total time (including Modal cold start): {m}m {s:02d}s")
