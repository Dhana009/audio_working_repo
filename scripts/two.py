#!/usr/bin/env python3
"""
Qwen3-TTS 1.7B CustomVoice — Modal GPU runner (optimized).

Optimizations applied:
  1. A10G GPU              — 2-3x faster than L4 for this model size
  2. Class-based container — model loaded ONCE, stays in memory across calls
     (scaledown_window=300 keeps it warm for 5 min between runs)

Usage:
    modal run scripts/generate_tts_modal.py
    modal run scripts/generate_tts_modal.py --speed 1.1
    modal run scripts/generate_tts_modal.py --output output/take2.wav
    modal run scripts/generate_tts_modal.py --text-file input.txt

    # Swap GPU: edit gpu="A10G" → gpu="A100" for ~4-6x vs L4
"""

from __future__ import annotations

import modal

# ── Modal App ─────────────────────────────────────────────────────────────────

app = modal.App("qwen3-tts")

# flash-attn requires CUDA dev headers not present in debian_slim.
# Using sdpa on A10G instead: still 2-3x faster than L4+sdpa overall.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.23,<2.0")           # numba compat (librosa dep)
    .pip_install("torch", "soundfile", "librosa")
    .pip_install("qwen-tts")
    .pip_install("huggingface_hub>=0.26.0,<1.0.0")
)

model_volume = modal.Volume.from_name("qwen3-tts-model-cache", create_if_missing=True)
VOLUME_PATH  = "/model-cache"
MODEL_REPO   = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

DEFAULT_TEXT = ( "The first chapter of Zero to One introduces a powerful way to think about progress.\n"

    "Most people believe progress means improving what already exists.\n"
    "We take something that works,\n"
    "and then we make it a little better.\n"

    "Peter Thiel calls this going from one to n.\n"
    "It means copying, scaling, or expanding something that already exists.\n"

    "But the book argues that the most important progress works differently.\n"
    "Real breakthroughs happen when someone creates something completely new.\n"

    "That is what Thiel calls going from zero to one.\n"
    "Instead of copying the past,\n"
    "you create something that did not exist before.\n"

    "For example,\n"
    "building another restaurant in a city is one to n.\n"
    "But inventing a completely new technology,\n"
    "like the first smartphone,\n"
    "is zero to one.\n"

    "This distinction changes how we think about innovation.\n"
    "Many industries focus on competition and incremental improvement.\n"
    "But the biggest value often comes from creating something entirely new.\n"

    "So the core idea of the chapter is simple.\n"
    "Progress is not just about doing more of the same.\n"
    "It is about discovering new ideas,\n"
    "and turning them into reality."
)
INSTRUCT_B = (
     "Explain the idea in a normal conversational tone. "
    "Speak at a natural everyday speaking speed, similar to how people talk in a casual discussion. "
    "Do not exaggerate words or emotions. "
    "Avoid sounding dramatic and avoid speaking too slowly. "
    "Sound slightly curious about the idea, as if you find it interesting and are sharing it naturally with someone. "
    "Keep the curiosity subtle and relaxed, not excited or exaggerated."
)

# ── Class-based container (model loaded once, reused across calls) ─────────────
#
# With @app.function the model reloads on every invocation (~27s overhead).
# With a @modal.cls the container stays warm: model loads in __enter__ once,
# then generate() can be called many times with zero reload cost.

@app.cls(
    gpu="A10G",                   # change to "A100" for max speed
    image=image,
    volumes={VOLUME_PATH: model_volume},
    timeout=600,
    scaledown_window=300,         # keep container warm for 5 min after last call
)
class TTSModel:

    @modal.enter()
    def load(self):
        """Runs once when the container starts. Model stays in memory."""
        import os
        import time

        import torch
        from qwen_tts import Qwen3TTSModel

        os.environ["HF_HOME"] = VOLUME_PATH

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print(f"[Modal] GPU: {gpu_name}")

        t0 = time.time()
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_REPO,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",  # sdpa on A10G is still 2-3x faster than L4
        )
        print(f"[Modal] Model loaded in {time.time() - t0:.1f}s  (warm for next calls)")

    @modal.method()
    def generate(
        self,
        text: str,
        instruct: str,
        speaker: str = "Ryan",
        language: str = "English",
        speed: float = 1.0,
    ) -> bytes:
        """Generate audio. Model is already loaded — no reload cost."""
        import io
        import time

        import numpy as np
        import soundfile as sf

        t1 = time.time()
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
        print(f"[Modal] Audio generated in {time.time() - t1:.1f}s  (sr={sr})")

        audio = np.array(wavs[0], dtype=np.float32)

        if speed != 1.0:
            import librosa
            print(f"[Modal] Applying speed x{speed} (pitch preserved)...")
            audio = librosa.effects.time_stretch(audio, rate=speed)

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()


# ── Local entrypoint (runs on your Mac) ───────────────────────────────────────

@app.local_entrypoint()
def main(
    text_file: str = "",
    output: str = "output/output2.wav",
    speaker: str = "Ryan",
    language: str = "English",
    speed: float = 1.0,
):
    """
    Optional CLI flags:
      --text-file   Path to a UTF-8 text file (default: built-in sample text)
      --output      Output wav path (default: output/output.wav)
      --speaker     Speaker name (default: Ryan)
      --language    Language (default: English)
      --speed       Speed multiplier, e.g. 1.1 = 10% faster (default: 1.0)
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

    print(f"Sending to Modal A10G GPU  (sdpa attention, warm container)...")
    print(f"  Speaker: {speaker}  |  Language: {language}  |  Speed: {speed}x")
    print("-" * 60)

    wav_bytes = TTSModel().generate.remote(
        text=text,
        instruct=INSTRUCT_B,
        speaker=speaker,
        language=language,
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
