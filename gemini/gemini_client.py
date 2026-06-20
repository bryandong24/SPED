#!/usr/bin/env python3
"""
gemini_client.py — dead-simple wrapper around Google **Gemini 3.5 Flash**.

Gemini 3.5 Flash (`gemini-3.5-flash`) is a fast, cheap, frontier-level multimodal
model. It accepts TEXT, IMAGE, VIDEO, AUDIO and PDF input and returns text. In this
repo the obvious use is as a *judge / evaluator*: hand it a generated `.mp4` clip or
a frame `.png` and ask it to score motion quality, prompt adherence, the swap
transition, artifacts, etc. It is also a general-purpose LLM for any agentic step.

────────────────────────────────────────────────────────────────────────────────
HOW TO CALL IT  (read this first)
────────────────────────────────────────────────────────────────────────────────
The key is already configured (see .env) and the SDK lives in a dedicated venv,
so there is NOTHING to set up. Two ways to call:

1) SHELL / CLI  — recommended for agents, zero setup, works from any directory:

     /mnt/data/SPED/gemini/gem "Explain self-forcing in one sentence."

     # judge a generated video:
     /mnt/data/SPED/gemini/gem -f out/run.mp4 \
         "Rate 1-10: does the dog->night prompt swap look smooth? Why?"

     # several files + JSON output:
     /mnt/data/SPED/gemini/gem -f a.png -f b.png --json \
         "Return {\"sharper\": \"a\"|\"b\", \"reason\": str}"

   (`gem` is a thin wrapper for `.venv/bin/python gemini_client.py`.)

2) PYTHON  — use the bundled interpreter `/mnt/data/SPED/gemini/.venv/bin/python`
   (it has google-genai installed):

     import sys; sys.path.insert(0, "/mnt/data/SPED/gemini")
     from gemini_client import ask, ask_json, describe_media, Chat

     ask("Summarize the SWAP_PLAN.")                       # -> str
     describe_media("out/run.mp4")                          # -> str (video understanding)
     ask("Score 1-10 and explain", files=["out/run.mp4"])  # -> str
     ask_json("Return {ok: bool, score: int}", schema={...})# -> dict

See README.md / AGENTS.md for more. Run `gem --info` to confirm config.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-3.5-flash"          # stable id from the model card
PREVIEW_MODEL = "gemini-3-flash-preview"    # fallback / preview alias
# Media at or below this size is sent INLINE (base64 inside the request; the
# total request is capped ~20MB, and base64 inflates ~33%, so 13MB is the safe
# ceiling). Larger media falls back to the Files API (upload + processing wait).
# Inline avoids the Files API entirely — faster, and works even if the key isn't
# permitted to call FileService.
INLINE_MAX_BYTES = 13 * 1024 * 1024

_EXTRA_MIME = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo", ".m4v": "video/mp4",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".aac": "audio/aac", ".m4a": "audio/mp4",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".heic": "image/heic",
    ".pdf": "application/pdf", ".txt": "text/plain",
}


def _load_dotenv() -> dict:
    """Parse the .env file sitting next to this module (cwd-independent).

    Tiny zero-dependency parser: `KEY=value` per line, `#` comments and blank
    lines ignored, optional surrounding quotes stripped. Returns {} if absent.
    """
    ep = Path(__file__).resolve().with_name(".env")
    if not ep.exists():
        return {}
    out: dict = {}
    try:
        for raw in ep.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                out[k] = v
    except Exception:
        return {}
    return out


def _load_file_key() -> Optional[str]:
    """Read the API key from .env next to this module (GEMINI_API_KEY/GOOGLE_API_KEY)."""
    env = _load_dotenv()
    return env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")


def resolve_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolution order: explicit arg > env GEMINI_API_KEY/GOOGLE_API_KEY > .env file."""
    return (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or _load_file_key()
    )


# Public constant so callers can `from gemini_client import API_KEY`.
API_KEY = resolve_key()

_CLIENT: Optional[genai.Client] = None


def get_client(api_key: Optional[str] = None) -> genai.Client:
    """Return a cached genai.Client (fresh one if api_key is overridden)."""
    global _CLIENT
    key = resolve_key(api_key)
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Put it in gemini/.env as "
            "GEMINI_API_KEY=... or set env GEMINI_API_KEY."
        )
    if api_key:
        return genai.Client(api_key=key)
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


# ──────────────────────────────────────────────────────────────────────────────
# Media handling
# ──────────────────────────────────────────────────────────────────────────────
def _guess_mime(path: Path) -> str:
    # Prefer our canonical table for known media (e.g. .wav -> audio/wav, not the
    # audio/x-wav some stdlib installs return); fall back to mimetypes otherwise.
    ext = path.suffix.lower()
    if ext in _EXTRA_MIME:
        return _EXTRA_MIME[ext]
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def _upload_and_wait(client: genai.Client, path: Path,
                     timeout: float = 600.0, poll: float = 2.0) -> types.File:
    """Upload via the Files API and block until the file is ACTIVE."""
    try:
        f = client.files.upload(file=str(path))
    except Exception as e:  # noqa: BLE001
        if "API_KEY_SERVICE_BLOCKED" in str(e) or "FileService" in str(e):
            mb = INLINE_MAX_BYTES // (1024 * 1024)
            raise RuntimeError(
                f"Files API upload is blocked for this API key (needed for media "
                f">{mb}MB such as {path}). Either allow the Generative Language "
                f"API's file methods on the key, or use a file <={mb}MB (those are "
                f"sent inline and don't need the Files API)."
            ) from e
        raise
    waited = 0.0
    while True:
        state = getattr(f.state, "name", str(f.state)).upper()
        if state == "ACTIVE":
            return f
        if state == "FAILED":
            raise RuntimeError(f"Gemini failed to process upload: {path}")
        if waited >= timeout:
            raise TimeoutError(f"Timed out ({timeout}s) waiting for upload: {path}")
        time.sleep(poll)
        waited += poll
        f = client.files.get(name=f.name)


def _file_to_content(client: genai.Client, file: Union[str, Path]) -> Any:
    """Turn a local path into something `contents` accepts (Part or uploaded File)."""
    p = Path(file).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Media file not found: {p}")
    # Small media (incl. short video/audio) goes inline; large media uses the
    # Files API. Inline is faster and needs no FileService permission.
    if p.stat().st_size <= INLINE_MAX_BYTES:
        return types.Part.from_bytes(data=p.read_bytes(), mime_type=_guess_mime(p))
    return _upload_and_wait(client, p)


def _build_contents(prompt: Optional[str], files, client: genai.Client) -> Any:
    parts: list = [_file_to_content(client, f) for f in (files or [])]
    if prompt:
        parts.append(prompt)
    if not parts:
        raise ValueError("Nothing to send: provide a prompt and/or files.")
    if len(parts) == 1 and isinstance(parts[0], str):
        return parts[0]
    return parts


# ──────────────────────────────────────────────────────────────────────────────
# Config builder
# ──────────────────────────────────────────────────────────────────────────────
_SAFETY_OFF = [
    types.SafetySetting(category=c, threshold="BLOCK_NONE")
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


def _thinking_config(thinking) -> Optional[types.ThinkingConfig]:
    if thinking is None:
        return None
    if isinstance(thinking, bool):  # note: bool is a subclass of int, check first
        return types.ThinkingConfig(thinking_level="high" if thinking else "minimal")
    if isinstance(thinking, str):
        # levels: 'minimal' | 'low' | 'medium' | 'high' (coerced case-insensitively)
        return types.ThinkingConfig(thinking_level=thinking)
    if isinstance(thinking, int):
        return types.ThinkingConfig(thinking_budget=thinking)
    raise TypeError("thinking must be None, bool, a level str "
                    "('minimal'/'low'/'medium'/'high'), or an int token budget")


def _make_config(
    *,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    thinking: Any = None,
    response_mime_type: Optional[str] = None,
    response_schema: Any = None,
    response_json_schema: Any = None,
    seed: Optional[int] = None,
    safety_off: bool = False,
) -> Optional[types.GenerateContentConfig]:
    kw: dict = {}
    if system is not None:
        kw["system_instruction"] = system
    if temperature is not None:
        kw["temperature"] = temperature
    if top_p is not None:
        kw["top_p"] = top_p
    if max_output_tokens is not None:
        kw["max_output_tokens"] = max_output_tokens
    if response_mime_type is not None:
        kw["response_mime_type"] = response_mime_type
    if response_schema is not None:
        kw["response_schema"] = response_schema
    if response_json_schema is not None:
        kw["response_json_schema"] = response_json_schema
    if seed is not None:
        kw["seed"] = seed
    tc = _thinking_config(thinking)
    if tc is not None:
        kw["thinking_config"] = tc
    if safety_off:
        kw["safety_settings"] = _SAFETY_OFF
    return types.GenerateContentConfig(**kw) if kw else None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def generate(
    prompt: Optional[str] = None,
    *,
    files=None,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    **config_kwargs,
) -> types.GenerateContentResponse:
    """Low-level call: returns the full GenerateContentResponse.

    config_kwargs are forwarded to _make_config (system, temperature, top_p,
    max_output_tokens, thinking, response_mime_type, response_schema,
    response_json_schema, seed, safety_off).
    """
    client = get_client(api_key)
    contents = _build_contents(prompt, files, client)
    config = _make_config(**config_kwargs)
    return client.models.generate_content(model=model, contents=contents, config=config)


def ask(prompt: Optional[str] = None, *, files=None, model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None, **config_kwargs) -> str:
    """The everyday call. Text in (+ optional media files), text out.

    >>> ask("hello")
    >>> ask("Score 1-10 and explain", files=["out/run.mp4"], thinking="high")
    """
    resp = generate(prompt, files=files, model=model, api_key=api_key, **config_kwargs)
    return (resp.text or "").strip()


def ask_json(prompt: Optional[str] = None, *, files=None, schema: Any = None,
             model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
             **config_kwargs) -> Any:
    """Like ask() but returns parsed JSON (dict/list).

    `schema` may be a JSON-Schema dict, a Pydantic model, or a TypedDict; pass
    None to just force JSON output.
    """
    config_kwargs.setdefault("response_mime_type", "application/json")
    if isinstance(schema, dict):
        config_kwargs["response_json_schema"] = schema
    elif schema is not None:
        config_kwargs["response_schema"] = schema
    resp = generate(prompt, files=files, model=model, api_key=api_key, **config_kwargs)
    parsed = getattr(resp, "parsed", None)
    if parsed is not None:
        return parsed
    return json.loads(resp.text)


def describe_media(path: Union[str, Path], prompt: str = "Describe this in detail.",
                   **kwargs) -> str:
    """Convenience: understand a single image / video / audio / pdf file."""
    return ask(prompt, files=[path], **kwargs)


class Chat:
    """Lightweight multi-turn conversation that remembers history.

    >>> c = Chat(system="You are a terse film critic.")
    >>> c.send("Rate this clip.", files=["out/a.mp4"])
    >>> c.send("Now compare it to the previous one.", files=["out/b.mp4"])
    """

    def __init__(self, *, system: Optional[str] = None, model: str = DEFAULT_MODEL,
                 api_key: Optional[str] = None, **config_kwargs):
        self._client = get_client(api_key)
        config = _make_config(system=system, **config_kwargs)
        self._chat = self._client.chats.create(model=model, config=config)

    def send(self, prompt: Optional[str] = None, *, files=None) -> str:
        contents = _build_contents(prompt, files, self._client)
        resp = self._chat.send_message(contents)
        return (resp.text or "").strip()

    @property
    def history(self):
        return self._chat.get_history()


def selftest(model: str = DEFAULT_MODEL) -> str:
    """Tiny live round-trip to confirm the key + model + network all work."""
    return ask("Reply with exactly: OK", model=model, temperature=0)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_thinking(v: Optional[str]):
    if v is None:
        return None
    if v.isdigit():
        return int(v)
    return v  # 'low' / 'high'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="gemini",
        description="Call Gemini 3.5 Flash. Prompt comes from the positional arg "
                    "or stdin. Prints the reply to stdout.",
    )
    ap.add_argument("prompt", nargs="?", help="prompt text (omit to read from stdin)")
    ap.add_argument("-f", "--file", action="append", default=[], dest="files",
                    metavar="PATH", help="media file (image/video/audio/pdf); repeatable")
    ap.add_argument("-s", "--system", help="system instruction")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"model (default {DEFAULT_MODEL})")
    ap.add_argument("-t", "--temperature", type=float)
    ap.add_argument("--max-tokens", type=int, dest="max_output_tokens")
    ap.add_argument("--thinking",
                    help="'minimal' | 'low' | 'medium' | 'high' | integer token budget")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--json", action="store_true", help="force/parse JSON output")
    ap.add_argument("--schema-file", help="path to a JSON-Schema file (implies --json)")
    ap.add_argument("--safety-off", action="store_true",
                    help="disable content safety filters (BLOCK_NONE)")
    ap.add_argument("--info", action="store_true",
                    help="print resolved config (no API call) and exit")
    ap.add_argument("--selftest", action="store_true",
                    help="do a tiny live round-trip and exit")
    args = ap.parse_args(argv)

    if args.info:
        key = resolve_key()
        src = ("env" if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
               else ".env" if key else "MISSING")
        masked = (key[:6] + "…" + key[-4:]) if key else "(none)"
        import importlib.metadata as _m
        print(f"model            : {args.model}")
        print(f"sdk google-genai : {_m.version('google-genai')}")
        print(f"api key source   : {src}")
        print(f"api key          : {masked}")
        return 0

    if args.selftest:
        try:
            out = selftest(args.model)
            print(out)
            return 0 if "OK" in out else 1
        except Exception as e:  # noqa: BLE001
            print(f"selftest failed: {e}", file=sys.stderr)
            return 1

    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt and not args.files:
        ap.error("provide a prompt (arg or stdin) and/or --file")

    cfg = dict(
        system=args.system,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        thinking=_parse_thinking(args.thinking),
        seed=args.seed,
        safety_off=args.safety_off,
    )
    cfg = {k: v for k, v in cfg.items() if v is not None and v is not False}

    try:
        if args.json or args.schema_file:
            schema = None
            if args.schema_file:
                schema = json.loads(Path(args.schema_file).read_text())
            out = ask_json(prompt, files=args.files, schema=schema, model=args.model, **cfg)
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            out = ask(prompt, files=args.files, model=args.model, **cfg)
            print(out)
    except Exception as e:  # noqa: BLE001
        print(f"gemini error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
