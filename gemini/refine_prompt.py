#!/usr/bin/env python
"""Refine a rough scene description into a rich cinematic video-model prompt.

Uses Gemini 3.1 Flash-Lite (the fast/"instant" tier) so it can sit in a live
steering loop. Reads the prompt from argv[1] or stdin, prints ONLY the refined
prompt to stdout. On any error, prints the original text unchanged (never blocks
the demo). Used by audio_stream/web_live_cf.py via subprocess.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "gemini-3.1-flash-lite"   # Gemini-3 "instant" tier — lowest latency
SYS = (
    "You are a prompt engineer for a cinematic text-to-video model (Wan2.1-style, "
    "photorealistic). Rewrite the user's scene into ONE rich, vivid, highly detailed "
    "cinematic video prompt of 2-3 sentences. PRESERVE the user's subject and scene "
    "intent exactly — do not change what the scene is about. ENRICH it with: specific "
    "subject appearance and texture, concrete setting elements, lighting and time-of-day, "
    "explicit camera framing AND camera movement, motion/action, and quality tags "
    "(HDR, 4K, photorealistic, cinematic, sharp detail). Keep it under ~80 words. "
    "Output ONLY the final prompt text — no preamble, no quotes, no markdown."
)


def refine(text, timeout=None):
    text = (text or "").strip()
    if not text:
        return text
    try:
        from gemini_client import ask
        out = ask(text, model=MODEL, system=SYS, temperature=0.7,
                  thinking="minimal", max_output_tokens=220)
        out = " ".join(out.strip().strip('"').strip().split())
        return out if len(out) > len(text) else text   # never shorten/lose intent
    except Exception:
        return text


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    sys.stdout.write(refine(src))
