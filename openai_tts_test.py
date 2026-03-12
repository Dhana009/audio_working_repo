from openai import OpenAI

client = OpenAI()

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
Speak like an engaging podcast host explaining a powerful idea from a book.
Conversational tone, confident pacing, natural emphasis on key ideas.
"""

input_text = DEFAULT_INSTRUCT + "\n\n" + DEFAULT_TEXT

speech = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input=input_text
)

with open("openai_output.mp3", "wb") as f:
    f.write(speech.content)

print("Audio saved to openai_output.mp3")