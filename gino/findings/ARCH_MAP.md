# ARCH_MAP.md — Self-Forcing autoregressive inference path

Repo: `/mnt/data/SPED/gino/Self-Forcing` (`guandeh17/Self-Forcing`, NeurIPS 2025).
Substrate: WAN 2.1-T2V-1.3B distilled to chunk-wise AR via DMD. All `file:line`
refs verified by reading the code (env still finishing flash-attn build at time
of writing; claims are static-analysis, to be re-confirmed by a live run).

## TL;DR — the three make-or-break answers

1. **Is the T5 text context per-chunk-overwritable mid-rollout? → YES, cleanly.**
   Text is encoded once (`causal_inference.py:84`) and its cross-attn **K/V is
   cached per block behind an `is_init` flag** (`wan/modules/model.py:174-186`).
   The cache is read-only after first fill *unless* `is_init` is reset to
   `False` — and the codebase already does exactly that reset to re-encode text
   (`causal_inference.py:124-126`). So a mid-rollout swap = (a) pass a new
   `conditional_dict` and (b) set `crossattn_cache[i]["is_init"]=False` for all
   30 blocks. This recomputes only the text K/V; it does **not** touch the
   self-attn KV cache of past frames (`kv_cache1`) or the diffusion latents.
   The brief's decoupling assumption is correct.

2. **Self-attn KV cache controls exist and are usable for Task 3** (window /
   sink / eviction) — see §3.

3. **Few-step headroom is small: 4 denoise steps/chunk** (distilled DMD,
   `configs/self_forcing_dmd.yaml: denoising_step_list: [1000,750,500,250]`).
   See `SPEEDUP_NOTES.md` for the implication.

## 0. Established facts — verification status

| Brief's claim | Verdict | Evidence |
|---|---|---|
| Chunk-wise, ~3 latent frames/chunk | ✅ | `num_frame_per_block: 3` (`configs/self_forcing_dmd.yaml`); loop `causal_inference.py:177-244` |
| A few distilled denoise steps/chunk | ✅ 4 steps | `denoising_step_list: [1000,750,500,250]`, `warp_denoising_step: true` |
| Block-causal sliding-window self-attn + KV cache of past frames | ✅ | `CausalWanSelfAttention.forward` `wan/modules/causal_model.py:86-240` |
| Text enters via cross-attn, retained from WAN | ✅ | `WanT2VCrossAttention` `wan/modules/model.py:159-194` |
| Text K/V has its own cache, decoupled from self-attn KV + latents | ✅ | `crossattn_cache` separate from `kv_cache1`; `causal_inference.py:300-312` |
| Rolling KV cache: sink tokens + eviction + local window | ✅ | `causal_model.py:202-235` |
| 14B variant exists | ✅ | `real_name: Wan2.1-T2V-14B` referenced; we run the 1.3B |

Latent shape per rollout: `[1, 21, 16, 60, 104]` = 21 latent frames, 16 ch,
60×104 latent grid (→ 480×832 px). 21 frames / 3-per-block = **7 chunks**.
`frame_seq_length = 1560` (= 60·104/4, patchified) — **fixed per frame**.

## 1. Chunk-wise autoregressive generation loop

`pipeline/causal_inference.py`, `CausalInferencePipeline.inference()`.

- **Outer AR loop over chunks**: `causal_inference.py:180` —
  `for current_num_frames in all_num_frames:` where
  `all_num_frames = [num_frame_per_block]*num_blocks` (`:177`). Each iteration
  = one chunk of 3 latent frames, denoised then committed to `output` (`:224`)
  and the KV cache (`:228-235`), with `current_start_frame += current_num_frames`
  (`:244`).
- **Per-chunk few-step denoise loop**: `causal_inference.py:188` —
  `for index, current_timestep in enumerate(self.denoising_step_list):`. 4
  iterations. Non-final steps re-noise the x0 prediction to the next timestep
  (`:206-211`); the final step yields the clean chunk (`:214-221`).
- **Post-chunk "clean context" pass**: `causal_inference.py:226-235` re-runs the
  model at `context_noise` timestep on the *clean* `denoised_pred` to write the
  past-frame K/V into `kv_cache1` (so future chunks attend to a denoised history,
  not a noisy one). This is the train/test-gap fix that names "Self-Forcing".

## 2. How the T5 context enters (the crux)

- **Encoded once per rollout**: `causal_inference.py:84-86`
  `conditional_dict = self.text_encoder(text_prompts=text_prompts)`.
  `WanTextEncoder.forward` (`utils/wan_wrapper.py:37-50`) tokenizes (umT5-xxl,
  `seq_len=512`) and returns `{"prompt_embeds": context}`.
- **Threaded by reference** into every model call as
  `conditional_dict=conditional_dict` (`causal_inference.py:197-204, 214-221,
  228-235`), unpacked to `context=prompt_embeds` in
  `WanDiffusionWrapper.forward` (`utils/wan_wrapper.py:230,243`).
- **Cross-attn caches the text K/V behind `is_init`** —
  `wan/modules/model.py:174-186`:
  ```python
  if crossattn_cache is not None:
      if not crossattn_cache["is_init"]:     # first call for this block
          crossattn_cache["is_init"] = True
          k = self.norm_k(self.k(context)).view(...)   # K from text
          v = self.v(context).view(...)                # V from text
          crossattn_cache["k"] = k; crossattn_cache["v"] = v
      else:
          k = crossattn_cache["k"]; v = crossattn_cache["v"]  # reuse; context IGNORED
  ```
  After the first chunk, `context` is ignored — the cached text K/V is reused.
- **Overwrite path already present**: `causal_inference.py:124-126` resets
  `crossattn_cache[block]["is_init"] = False` for all blocks (used when a fresh
  `inference()` call reuses caches). This is the exact hook a mid-rollout swap
  needs. **Make-or-break: the text context is NOT baked in at init — it is
  per-chunk overwritable.**
- Cross-attn cache shape: `[B, 512, 12, 128]` per block, 30 blocks
  (`causal_inference.py:300-312`) — 512 = umT5 token budget, 12 heads × 128 dim.

### Swap recipe (for Task 2)
Between chunk K and K+1, before the next denoise loop:
```python
new_cond = pipeline.text_encoder(text_prompts=[new_prompt])
for c in pipeline.crossattn_cache:
    c["is_init"] = False
conditional_dict = new_cond          # pass this into subsequent generator() calls
```
Self-attn `kv_cache1` (past frames under old prompt) is left intact → the new
chunk reconciles old-world momentum (self-attn) with new prompt (cross-attn).
This is the expected out-of-distribution regime to characterize in Task 2.

## 3. Self-attention KV cache of past frames + its controls

`CausalWanSelfAttention.forward`, `wan/modules/causal_model.py:86-240` (the
`kv_cache is not None` branch, `:193-235`).

- **Storage**: `kv_cache1`, a list of 30 dicts `{k, v, global_end_index,
  local_end_index}` (`causal_inference.py:278-298`). K/V shape `[B,
  kv_cache_size, 12, 128]`.
- **Window size (`local_attn_size`)**: cache size =
  `local_attn_size * frame_seq_length` if set, else default `32760`
  (`causal_inference.py:283-288`; `causal_model.py:76`
  `max_attention_size = 32760 if local_attn_size==-1 else local_attn_size*1560`).
  Attention reads only the trailing window
  `kv_cache["k"][local_end_index - max_attention_size : local_end_index]`
  (`causal_model.py:231-232`). Set at model init
  (`WanDiffusionWrapper.__init__`, `wan_wrapper.py:126-128`).
- **Sink tokens (`sink_size`)**: first `sink_size` frames
  (`sink_tokens = sink_size * frame_seqlen`, `causal_model.py:202`) are pinned —
  eviction shifts only the region *after* the sinks (`:211-216`).
- **Eviction**: when `num_new + local_end_index > kv_cache_size`, oldest
  non-sink tokens are dropped by a left-shift (`causal_model.py:206-222`).
- **RoPE indexing**: `causal_rope_apply(q, ..., start_frame=current_start//frame_seqlen)`
  (`causal_model.py:194-199`) — positions keyed to absolute frame index assuming
  a **fixed `frame_seqlen=1560` per frame**. (Relevant to `SPEEDUP_NOTES.md`.)

### Task-3 lever (post-swap window shrink/evict)
To make the new prompt re-anchor faster, after a swap manually reduce each
block's effective history: e.g. lower `local_end_index`/`global_end_index` to
drop old-prompt frames, or rebuild the pipeline with a smaller `local_attn_size`.
Both are reachable at the pipeline level via the cache dicts above.

## 4. Entry points / configs
- CLI: `inference.py` → builds `CausalInferencePipeline`, config
  `configs/self_forcing_dmd.yaml`, checkpoint `checkpoints/self_forcing_dmd.pt`,
  `--use_ema`.
- Streaming GUI: `demo.py` (Flask + socketio).
- Key knobs (config): `num_frame_per_block: 3`, `denoising_step_list`,
  `timestep_shift: 5.0`, `guidance_scale: 3.0`, `context_noise` (arg),
  `independent_first_frame`.

## Flagged discrepancies vs. the brief
- None material. The brief said "~3 latent frames per chunk, a few denoise
  steps" — confirmed exactly 3 frames, 4 steps. Cross-attn caching via `is_init`
  is *cleaner* than the brief implied ("recompute/overwrite the cross-attention
  K/V cache"): you don't manually overwrite K/V, you just flip `is_init=False`
  and pass new embeds; the model re-fills the cache itself.
