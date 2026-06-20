"""Gemini command planner: natural-language voice/text -> structured camera command.

Gemini never drives the model directly. It emits a constrained JSON `WorldCommand`; the
deterministic CameraController turns that into camera motion. A single Gemini call can take
either typed text or a recorded audio clip (transcribe + parse in one shot), since
`ask_json(..., files=[audio])` accepts inline audio.

Velocity conventions (must match camera.py):
  forward: + = move forward, - = backward
  strafe:  + = move right,    - = left
  turn:    + = turn right,    - = turn left   (yaw)
  pitch:   + = look up,       - = look down
  up:      + = move up,       - = move down
  speed:   0..2 global magnitude (1 = normal, 0 = hold still)
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
GEMINI_DIR = os.environ.get("GEMINI_DIR", "/mnt/data/SPED/gemini")
if GEMINI_DIR not in sys.path:
    sys.path.insert(0, GEMINI_DIR)

_AXES = ("forward", "strafe", "turn", "pitch", "up")

WORLD_COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "camera": {
            "type": "object",
            "properties": {
                "forward": {"type": "number"},
                "strafe": {"type": "number"},
                "turn": {"type": "number"},
                "pitch": {"type": "number"},
                "up": {"type": "number"},
                "speed": {"type": "number"},
            },
            "required": ["forward", "strafe", "turn", "pitch", "up", "speed"],
        },
        "prompt_delta": {"type": "string"},
        "intent": {"type": "string",
                   "enum": ["move", "look", "style", "scene", "stop", "none"]},
        "transcript": {"type": "string"},
    },
    "required": ["camera", "intent"],
}

SYSTEM = """You are the command router for a real-time, camera-controllable video world model.
Convert the user's spoken or typed instruction into a single JSON camera command.

The camera has five continuous velocity axes, each a float in [-1, 1], plus a speed
multiplier in [0, 2]:
  forward: +forward / -backward
  strafe:  +right   / -left
  turn:    +turn-right / -turn-left   (yaw)
  pitch:   +look-up  / -look-down
  up:      +move-up  / -move-down
  speed:   overall magnitude (1 = normal pace, 0 = hold still)

Rules:
- Output ONLY the JSON. Set unused axes to 0.0.
- "stop" / "hold" / "freeze" -> all axes 0 and speed 0, intent "stop".
- Combine axes when the instruction does ("turn left and go forward" -> turn -0.5, forward 0.7).
- Map vague intensity words: "slightly"~0.3, default~0.7, "fast/hard"~1.0.
- If the instruction is about the world's look/content rather than motion (e.g. "make it
  snowy", "drive into a city"), put a concise scene/style phrase in prompt_delta and set the
  matching intent ("style"/"scene"); still fill camera (0s if no motion implied).
- If you were given audio, also fill "transcript" with what you heard.

Examples:
"go forward" -> {"camera":{"forward":0.8,"strafe":0,"turn":0,"pitch":0,"up":0,"speed":1},"prompt_delta":"","intent":"move"}
"turn left and walk ahead" -> {"camera":{"forward":0.7,"strafe":0,"turn":-0.5,"pitch":0,"up":0,"speed":1},"prompt_delta":"","intent":"move"}
"look up at the sky" -> {"camera":{"forward":0,"strafe":0,"turn":0,"pitch":0.8,"up":0,"speed":1},"prompt_delta":"","intent":"look"}
"stop" -> {"camera":{"forward":0,"strafe":0,"turn":0,"pitch":0,"up":0,"speed":0},"prompt_delta":"","intent":"stop"}
"make it a snowy night" -> {"camera":{"forward":0,"strafe":0,"turn":0,"pitch":0,"up":0,"speed":1},"prompt_delta":"snowy winter night, cold moonlight, falling snow","intent":"style"}
"""

# Keyword fallback for offline dev / Gemini failures.
_KEYWORDS = [
    (("stop", "halt", "freeze", "hold"), {"speed": 0.0}),
    (("forward", "ahead", "go", "walk", "advance"), {"forward": 1.0}),
    (("back", "backward", "reverse"), {"forward": -1.0}),
    (("left",), {"turn": -1.0}),
    (("right",), {"turn": 1.0}),
    (("up", "sky"), {"pitch": 1.0}),
    (("down", "ground", "floor"), {"pitch": -1.0}),
]


def _zero_camera():
    return {"forward": 0.0, "strafe": 0.0, "turn": 0.0, "pitch": 0.0, "up": 0.0, "speed": 1.0}


class VoicePlanner:
    def __init__(self, model=None, temperature=0.0):
        self.model = model
        self.temperature = temperature
        self._ask_json = None  # lazy import so the worker can load without google-genai

    def _client(self):
        if self._ask_json is None:
            from gemini_client import ask_json, DEFAULT_MODEL
            self._ask_json = ask_json
            self.model = self.model or DEFAULT_MODEL
        return self._ask_json

    def plan(self, text=None, audio_path=None, state=None):
        """Return a normalized WorldCommand dict from text and/or an audio clip."""
        user = (text or "").strip()
        if state:
            user = (f"Current camera state: {json.dumps(state)}\n"
                    f"New instruction: {user}" if user else
                    f"Current camera state: {json.dumps(state)}\nTranscribe the audio and act on it.")
        files = [audio_path] if audio_path else None
        try:
            ask_json = self._client()
            cmd = ask_json(user or "Transcribe the audio and produce the camera command.",
                           files=files, schema=WORLD_COMMAND_SCHEMA, system=SYSTEM,
                           temperature=self.temperature, model=self.model)
            return self._normalize(cmd)
        except Exception as e:  # network/key/parse failure -> keyword fallback (text only)
            print(f"[planner] Gemini failed ({e}); using keyword fallback", file=sys.stderr)
            return self._fallback(text or "")

    @staticmethod
    def _normalize(cmd):
        cam = _zero_camera()
        src = (cmd or {}).get("camera", {}) or {}
        for k in _AXES:
            try:
                cam[k] = max(-1.0, min(1.0, float(src.get(k, 0.0))))
            except (TypeError, ValueError):
                cam[k] = 0.0
        try:
            cam["speed"] = max(0.0, min(2.0, float(src.get("speed", 1.0))))
        except (TypeError, ValueError):
            cam["speed"] = 1.0
        return {
            "camera": cam,
            "prompt_delta": str((cmd or {}).get("prompt_delta", "") or ""),
            "intent": (cmd or {}).get("intent", "none"),
            "transcript": str((cmd or {}).get("transcript", "") or ""),
        }

    def _fallback(self, text):
        cam = _zero_camera()
        low = text.lower()
        hit = False
        for words, delta in _KEYWORDS:
            if any(w in low for w in words):
                cam.update(delta)
                hit = True
        intent = "stop" if cam["speed"] == 0.0 else ("move" if hit else "none")
        return {"camera": cam, "prompt_delta": "", "intent": intent, "transcript": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=None)
    ap.add_argument("--audio", default=None, help="path to a recorded command clip")
    args = ap.parse_args()
    out = VoicePlanner().plan(text=args.text, audio_path=args.audio)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
