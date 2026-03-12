#!/usr/bin/env python3
"""
Qwen3-TTS 1.7B — Modal GPU runner (FAST version)

Optimizations
-------------
✓ A100 GPU
✓ Model loaded once (warm container)
✓ HF cache stored in Modal volume
✓ Container stays warm 10 minutes
"""

from __future__ import annotations
import modal

app = modal.App("qwen3-tts")

MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOLUME_PATH = "/model-cache"

# ---------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("sox")
    .pip_install("numpy>=1.23,<2.0")
    .pip_install("torch", "soundfile", "librosa")
    .pip_install("qwen-tts")
    .pip_install("huggingface_hub>=0.26.0,<1.0.0")
)

# persistent cache
model_volume = modal.Volume.from_name(
    "qwen3-tts-model-cache",
    create_if_missing=True,
)

# ---------------------------------------------------------------------
# Text content
# ---------------------------------------------------------------------

DEFAULT_TEXT = """
Let’s explore an idea that sounds simple, but can completely change how we think about success.

Many people believe that real transformation requires dramatic action.
We wait for the big breakthrough, the perfect moment, or a sudden burst of motivation.

But what if progress does not actually work that way?

Author James Clear suggests something surprisingly different.
Instead of chasing massive change, we should focus on getting just one percent better every day.

At first, a one percent improvement feels almost meaningless.
You barely notice it.
But small gains have a powerful property: they compound over time.

Just like money growing with compound interest, tiny improvements begin to multiply.
Weeks turn into months, months turn into years, and suddenly those small steps become extraordinary progress.

The people who succeed are rarely the ones who make a single dramatic leap.
They are the ones who quietly improve, day after day, long after everyone else has stopped trying.

So the real lesson is simple.
Do not wait for the big moment.

Focus on becoming just a little better today than you were yesterday.
And trust that over time, those small improvements will transform everything.
"""

DEFAULT_INSTRUCT = """
You are an engaging podcast host explaining powerful ideas from nonfiction books.

Your goal is not to simply read the text, but to clearly explain the idea to a listener in a natural and compelling way.

Voice style guidelines:

• Speak conversationally, like you are sharing an insight with a curious friend.
• Maintain a confident and warm tone.
• Keep a steady pace similar to a thoughtful podcast.
• Emphasize important phrases and key ideas naturally.
• Add small pauses between major thoughts so the listener can absorb the message.
• Avoid sounding robotic or like you are reading from a script.
• Start the first sentence with energy and clarity.

Overall tone: insightful, conversational, and engaging.
"""

# ---------------------------------------------------------------------
# GPU container
# ---------------------------------------------------------------------

@app.cls(
    gpu="A100",                 # FASTEST option
    image=image,
    volumes={VOLUME_PATH: model_volume},
    scaledown_window=600,      # keep warm 10 minutes
    timeout=900,
)
class TTSModel:

    @modal.enter()
    def load_model(self):
        import os
        import torch
        import time
        from qwen_tts import Qwen3TTSModel

        os.environ["HF_HOME"] = VOLUME_PATH

        print("Loading model on GPU...")
        start = time.time()

        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_REPO,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        print(f"Model ready in {time.time()-start:.1f}s")

    @modal.method()
    def generate(self, text: str, instruct: str):
        import io
        import numpy as np
        import soundfile as sf
        import time

        start = time.time()

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language="English",
            speaker="Ryan",
            instruct=instruct,
        )

        print(f"Generation took {time.time()-start:.1f}s")

        audio = np.array(wavs[0], dtype=np.float32)

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")

        return buf.getvalue()

# ---------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------

@app.local_entrypoint()
def main(output: str = "output/output.wav"):

    import time
    from pathlib import Path

    start = time.time()

    tts = TTSModel()

    print("Sending request to Modal GPU...")
    print("-" * 50)

    wav_bytes = tts.generate.remote(
        text=DEFAULT_TEXT,
        instruct=DEFAULT_INSTRUCT,
    )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav_bytes)

    elapsed = int(time.time() - start)
    m, s = divmod(elapsed, 60)

    print("-" * 50)
    print(f"Saved audio → {output}")
    print(f"Total time: {m}m {s}s")