# Gemini 3.5 Flash helper

A tiny, dependency-light wrapper around Google **Gemini 3.5 Flash**
(`gemini-3.5-flash`) — a fast, cheap, frontier multimodal model. It takes
**text, image, video, audio, and PDF** input and returns text. In this repo the
natural use is as a **judge / evaluator** for generated video (rate a swap
transition, spot artifacts, check prompt adherence), plus general LLM calls.

> **AI agents:** read [`AGENTS.md`](AGENTS.md) — it's the short, copy-paste version.

The API key and the SDK are already set up, so there is nothing to install.

## Quick start

```bash
# CLI — works from any directory, prints reply to stdout
/mnt/data/SPED/gemini/gem "Explain self-forcing in one sentence."

# Judge a generated clip
/mnt/data/SPED/gemini/gem -f out/run.mp4 "Rate 1-10 how smooth the swap looks. Why?"

# Health check (tiny live call → prints OK)
/mnt/data/SPED/gemini/gem --selftest
```

```python
# Python — use the bundled venv: /mnt/data/SPED/gemini/.venv/bin/python
import sys; sys.path.insert(0, "/mnt/data/SPED/gemini")
from gemini_client import ask, ask_json, describe_media, Chat

ask("hello")                                       # -> str
ask("score 1-10 + why", files=["out/run.mp4"])     # multimodal
ask_json("Return {score:int, reason:str}")          # -> dict
describe_media("frame.png")                          # -> str
```

## Layout

| file | purpose |
|---|---|
| `gemini_client.py` | the wrapper: `ask`, `ask_json`, `describe_media`, `generate`, `Chat`, and a CLI |
| `gem` | shell launcher → `.venv/bin/python gemini_client.py "$@"` |
| `.env` | holds `GEMINI_API_KEY` (git-ignored) |
| `.venv/` | venv with `google-genai` (git-ignored) |
| `AGENTS.md` | terse instructions for AI agents |
| `example.py` | runnable examples |
| `requirements.txt` | `google-genai>=2.9.0` |

## Python API

```python
ask(prompt, *, files=None, model="gemini-3.5-flash", system=None,
    temperature=None, max_output_tokens=None, thinking=None,
    seed=None, safety_off=False) -> str

ask_json(prompt, *, files=None, schema=None, **opts) -> dict|list
    # schema: a JSON-Schema dict, a Pydantic model, or a TypedDict (or None)

describe_media(path, prompt="Describe this in detail.", **opts) -> str

generate(prompt, *, files=None, **opts) -> GenerateContentResponse  # full object

class Chat(system=None, model=..., **opts):
    send(prompt, *, files=None) -> str        # remembers history
    history                                   # list of turns
```

- `files` is a list of local paths. Media **≤13 MB (any type) is sent inline**;
  larger files are uploaded through the Files API (which blocks until processed
  and requires the key to permit the file methods).
- `thinking`: `"minimal"|"low"|"medium"|"high"`, or an integer token budget, or
  `True`/`False`. Omit for the model default.
- `model`: defaults to `gemini-3.5-flash`. Preview alias: `gemini-3-flash-preview`
  (exposed as `PREVIEW_MODEL`).

## Configuration

The key is resolved in this order (first hit wins):

1. `api_key=...` passed in code
2. env `GEMINI_API_KEY` or `GOOGLE_API_KEY`
3. `.env` in this folder (`GEMINI_API_KEY=...`)

`gem --info` prints the model, SDK version, and which source the key came from
(without making an API call).

### Recreate the venv

```bash
cd /mnt/data/SPED/gemini
uv venv .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt
```

## Security

`.env` and `.venv/` are git-ignored, so the key is **not** committed. The key
is plaintext on disk; keep this folder private. To rotate, edit `.env` or set
`GEMINI_API_KEY`. The folder is world-readable so agents running as other Unix
users on this machine can call it — fine on a single-tenant box, but don't put a
shared box's key here if other users shouldn't have it.

## Model facts (from the model card)

| | |
|---|---|
| Model code | `gemini-3.5-flash` (stable) · `gemini-3-flash-preview` (preview) |
| Input | text, image, video, audio, PDF |
| Output | text |
| Input limit | 1,048,576 tokens |
| Output limit | 65,536 tokens |
| Supports | thinking, structured output, function calling, caching, code execution, search/URL grounding |
