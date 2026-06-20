# Gemini 3.5 Flash — instructions for AI agents

A ready-to-use **Gemini 3.5 Flash** (`gemini-3.5-flash`) helper. Key is already
configured; SDK is in a bundled venv. **No setup needed.** Multimodal: text, image,
**video**, audio, PDF in → text out. Great for *judging generated clips/frames*.

## The one call you need (CLI, works from any directory)

```bash
/mnt/data/SPED/gemini/gem "your prompt here"
```

`gem` runs the bundled venv automatically. The reply is printed to **stdout**;
errors go to **stderr** with a non-zero exit code. So you can capture it:

```bash
ANSWER=$(/mnt/data/SPED/gemini/gem "one-sentence summary of self-forcing")
```

### Common forms

```bash
# Judge / evaluate a generated video (auto-uploaded via the Files API):
/mnt/data/SPED/gemini/gem -f /mnt/data/SPED/gino/out/stream/run.mp4 \
    "Rate 1-10 how smooth the prompt swap looks and name the worst artifact."

# Multiple media files:
/mnt/data/SPED/gemini/gem -f a.png -f b.png "Which frame is sharper, a or b? Why?"

# Strict JSON output (parseable):
/mnt/data/SPED/gemini/gem --json "Return {\"score\": int 1-10, \"reason\": str} for: the render looks crisp"

# JSON constrained by a schema file:
/mnt/data/SPED/gemini/gem --json --schema-file schema.json -f run.mp4 "score this clip"

# Long prompt from stdin:
cat findings/FINDINGS.md | /mnt/data/SPED/gemini/gem "summarize the open questions"

# Knobs:
/mnt/data/SPED/gemini/gem -s "You are a terse film critic." --thinking high -t 0.2 "..."
```

### Flags

| flag | meaning |
|---|---|
| `-f, --file PATH` | media file (image/video/audio/pdf); repeat for several |
| `-s, --system STR` | system instruction |
| `-m, --model ID` | model id (default `gemini-3.5-flash`) |
| `-t, --temperature F` | sampling temperature |
| `--max-tokens N` | max output tokens |
| `--thinking LVL` | `minimal`\|`low`\|`medium`\|`high`, or an integer token budget |
| `--json` | force + parse JSON output |
| `--schema-file P` | JSON-Schema file to constrain output (implies `--json`) |
| `--safety-off` | disable safety filters (`BLOCK_NONE`) |
| `--info` | print resolved config, no API call |
| `--selftest` | tiny live round-trip; prints `OK` |

## From Python

Use the bundled interpreter (it has `google-genai`):
`/mnt/data/SPED/gemini/.venv/bin/python`

```python
import sys; sys.path.insert(0, "/mnt/data/SPED/gemini")
from gemini_client import ask, ask_json, describe_media, Chat

ask("hello")                                            # -> str
ask("score 1-10 + why", files=["out/run.mp4"], thinking="high")  # video judge
describe_media("frame.png")                             # -> str
ask_json("Return {ok: bool, score: int}",               # -> dict
         schema={"type":"object","properties":{"ok":{"type":"boolean"},
                 "score":{"type":"integer"}},"required":["ok","score"]})

c = Chat(system="terse critic"); c.send("rate", files=["a.mp4"]); c.send("vs this?", files=["b.mp4"])
```

## Rules of thumb

- Media **≤13 MB (any type, incl. short video/audio) is sent inline** — fast, no
  upload. Larger files use the Files API (upload + processing wait), which needs
  the API key to also permit the Generative Language *file* methods.
- Need machine-readable output? Use `--json` (CLI) or `ask_json(...)` (Python) —
  don't parse prose.
- Confirm it's live with `gem --selftest` (prints `OK`). `gem --info` shows the
  model + key without making a call.
- Input limit ~1M tokens; output limit 64k tokens.
- Key/model live in `gemini_client.py` / `.env`. Override the key per-run with
  env `GEMINI_API_KEY`.
