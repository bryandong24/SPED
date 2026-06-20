#!/usr/bin/env python3
"""Runnable examples. Run with the bundled venv:

    /mnt/data/SPED/gemini/.venv/bin/python /mnt/data/SPED/gemini/example.py
"""
import sys
sys.path.insert(0, "/mnt/data/SPED/gemini")

from gemini_client import ask, ask_json, describe_media, Chat  # noqa: E402


def main():
    # 1) plain text
    print("[text]    ", ask("Say hi in 3 words.", temperature=0))

    # 2) structured JSON output with a schema
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["pos", "neg", "neutral"]},
            "score": {"type": "integer"},
        },
        "required": ["sentiment", "score"],
    }
    print("[json]    ", ask_json("Classify: 'this render looks great!'", schema=schema))

    # 3) multimodal — judge a generated clip (uncomment with a real file)
    # print("[video]   ", ask(
    #     "Rate 1-10 how smooth the prompt swap looks, and name the worst artifact.",
    #     files=["/mnt/data/SPED/gino/out/stream/run.mp4"], thinking="high"))

    # 4) describe any media
    # print("[describe]", describe_media("/path/to/frame.png"))

    # 5) multi-turn chat
    c = Chat(system="You are a terse assistant.")
    print("[chat 1]  ", c.send("Pick a number 1-10."))
    print("[chat 2]  ", c.send("Add 5 to it. Show the math."))


if __name__ == "__main__":
    main()
