# minWM Training — HunyuanVideo 1.5 Backbone

Two model lines built on HunyuanVideo 1.5: **HY Action2V** (camera-controlled action-to-video) and **HY TI2V** (text+image to video).

Each line splits into two phases:
- **Phase 1 Bidirectional SFT** — bidirectional multi-step base.
- **Phase 2 Causal Forcing** — distillation to causal few-step.

Phase 2 has 4 stages: Stage 1 Teacher Forcing AR Diffusion, Stage 2(a) Causal ODE Distillation Initialization (Causal Forcing), Stage 2(b) Causal Consistency Distillation (Causal Forcing++), Stage 3 Asymmetric DMD with Self Rollout.

Every subsection follows the same structure: **(1) Model download**, **(2) Data preparation**, **(3) Training script**, **(4) Validation**.

> Wan-backbone training lives in [`training_wan.md`](training_wan.md). Quick Start / inference commands live in the main [README](README.md).

First install:

```bash
conda create -n minwm python=3.10 -y 
conda activate minwm
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:$PYTHONPATH"
```


---

## 1. HY Action2V

### 1.1 Phase 1: Bidirectional SFT

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**


```bash
hf download tencent/HunyuanVideo-1.5 \
    --local-dir ./ckpts/HunyuanVideo-1.5 \
    --include "vae/*" "scheduler/*" "transformer/480p_i2v/*"

hf download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/llm

hf download google/byt5-small \
    --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/byt5-small

modelscope download --model AI-ModelScope/Glyph-SDXL-v2 \
    --local_dir ./ckpts/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2

hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir ./ckpts/HunyuanVideo-1.5/vision_encoder/siglip \
    --token <your_hf_token>
```

**(2) Data preparation**

Pick one. Everything lands under `./dataset/`.

**Option A: download minWM-dataset**

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "preencode_input.json" "videos/**"
```

Resulting layout:

```
./dataset/
├── preencode_input.json
└── videos/
    ├── 000000_right8a11/gen.mp4
    ├── 000001_w10d9/gen.mp4
    └── ...
```

**Option B: bring your own videos and trajectories**

Match Option A's layout: provide your own `preencode_input.json` plus `videos/`. `preencode_input.json` is a list; each entry must contain at least `image_path` / `caption` / `pose_str`:

```json
[
    {
        "image_path": "/abs/path/to/image1.png",
        "caption": "A scenic mountain view",
        "pose_str": "right-8, a-11"
    }
]
```

Video directory naming rule: `{i:06d}_{slug(pose_str)}/gen.mp4`, where `i` is the JSON index and `slug` is `pose_str` lowercased with non-alphanumeric characters stripped.

**Final: encoding (shared by both options)**

Script defaults: `HUNYUAN_CHECKPOINT=./ckpts/HunyuanVideo-1.5`, `INPUT_DIR=./dataset`, `OUTPUT_DIR=./dataset/HY15/Action2V`:

```bash
bash HY15/scripts/data_preprocessing/run_preencode_downloaded_camera_video.sh
```

CFG negative prompt embeddings are downloaded separately:

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "others/HY/Action2V/**"
```

Final layout:

```
./dataset/
├── HY15/Action2V/                       # encoded outputs
│   ├── latents/
│   └── train_index.json
└── others/HY/Action2V/                  # HF neg prompts
    ├── hunyuan_neg_prompt.pt
    ├── hunyuan_neg_byt5_prompt.pt
    └── negative_prompt.pt
```

**(3) Training script**

Bidirectional + camera (ProPE) training. Model class `HunyuanTransformer3DARActionProPEModel`, camera dataloader carrying viewmats/Ks, flow matching MSE loss:

```bash
bash HY15/scripts/training/hy15_camera/run_bi_camera_multinode.sh
```

By default the script writes ckpts to `./ckpts/HY15/Action2V/bidirectional/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents (`diffusion_pytorch_model.safetensors` + `config.json` etc.) to `./ckpts/HY15/Action2V/bidirectional/`, matching the predownload layout used in §1.2 Stage 1 (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/Action2V/bidirectional/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/bidirectional/
```

**(4) Validation**

Reuses the README Quick Start HY Action2V inference script in 50-step bidirectional mode. The script defaults to `TRANSFORMER_DIR=./ckpts/HY15/Action2V/bidirectional` (this stage's output, also the §1.2 Stage 1 predownload path); override via env to point at another ckpt:

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/bidirectional \
OUTPUT_DIR=./outputs/eval_bidir_camera \
    bash HY15/scripts/inference/run_infer_bidirectional_camera.sh
```

</details>

### 1.2 Phase 2: Causal Forcing

#### Stage 1: Teacher Forcing AR Diffusion

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/bidirectional/**"
```


**(2) Data preparation**

Same as §1.1 (2). Reuses the same `./dataset/HY15/Action2V/{latents, train_index.json}` and `./dataset/others/HY/Action2V/`.

**(3) Training script**

Convert the bidirectional model from Phase 1 into causal + teacher-forcing AR. Same ProPE model class and camera dataloader; loss is still flow matching MSE:

```bash
bash HY15/scripts/training/hy15_camera/run_ar_hunyuan_mem_multinode.sh
```

By default ckpts land at `./ckpts/HY15/Action2V/ar_diffusion_tf/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/Action2V/ar_diffusion_tf/`, matching the predownload layout used in §1.2 Stage 2(a) (1):

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/ar_diffusion_tf/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/ar_diffusion_tf/
```

**(4) Validation**

50-step AR rollout mode. Script defaults `TRANSFORMER_DIR=./ckpts/HY15/Action2V/ar_diffusion_tf` (this stage's output, also the Stage 2(a) predownload path); other variables likewise overridable via env:

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/ar_diffusion_tf \
OUTPUT_DIR=./outputs/eval_ar_camera \
    bash HY15/scripts/inference/run_infer_ar_diffusion_camera.sh
```

</details>

#### Stage 2(a): Causal ODE Distillation Initialization (Causal Forcing)

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/ar_diffusion_tf/**"
```

**(2) Data preparation**

Pick one. Everything lands at `./dataset/HY15/Action2V_ode/`, parallel to and not overlapping the §1.1 SFT latents (`./dataset/HY15/Action2V/`). Negative prompts reuse the `./dataset/others/HY/Action2V/` already downloaded in §1.1.

**Option A: download minWM-dataset's pre-generated ODE latents**

```bash
# 1) Download from HF (the repo layout lands under ODE_data/)
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "ODE_data/HY15/Action2V/**"

# 2) Move to the unified layout ./dataset/HY15/Action2V_ode/
mkdir -p ./dataset/HY15/Action2V_ode
mv ./dataset/ODE_data/HY15/Action2V/latents ./dataset/HY15/Action2V_ode/latents


# 3) Regenerate the absolute-path index in the new location
#    (writes to ./dataset/HY15/Action2V_ode/train_index.json)
python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/Action2V_ode/latents \
    -o ./dataset/HY15/Action2V_ode/train_index.json
```

**Option B: sample yourself from the Stage 1 ckpt**

Requires the §1.2 Stage 1 ckpt (at `./ckpts/HY15/Action2V/ar_diffusion_tf/`); run 48-step CFG sampling on the SFT-encoded data (§1.1):

```bash
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY15/Action2V/ar_diffusion_tf/diffusion_pytorch_model.safetensors \
PREENCODED_DIR=./dataset/HY15/Action2V/train_index.json \
ODE_OUTPUT_PATH=./dataset/HY15/Action2V_ode/latents \
    bash HY15/scripts/training/hy15_camera/ode_sampling.sh

python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/Action2V_ode/latents \
    -o ./dataset/HY15/Action2V_ode/train_index.json
```

Final layout:

```
./dataset/
├── HY15/
│   ├── Action2V/                        # SFT latents (from §1.1)
│   │   ├── latents/
│   │   └── train_index.json
│   └── Action2V_ode/                    # ODE latents (this section)
│       ├── latents/
│       └── train_index.json
└── others/HY/Action2V/                  # neg prompts (reused from §1.1)
```

**(3) Training script**

ODE regression: on the ODE latents prepared in §1.2 Stage 2(a) (2) (`./dataset/HY15/Action2V_ode/`), have the model directly regress the ODE solver outputs at key timesteps [0, 12, 24, 36, -2, -1] (corresponding to [1000, 750, 500, 250] etc.):

```bash
bash HY15/scripts/training/hy15_camera/run_ar_causal_ode.sh
```

By default ckpts land at `./ckpts/HY15/Action2V/causal_ode/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/Action2V/causal_ode/`, matching the predownload layout used in §1.2 Stage 3 (1):

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/causal_ode/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/causal_ode/
```

**(4) Validation**

The 4-step DMD inference script (`run_infer_causal_camera.sh`) is shared by Stage 2(a) / 2(b) / 3, defaulting to `TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd` (the final output, matching the README Quick Start). For this stage, override via env to causal_ode:

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/causal_ode \
OUTPUT_DIR=./outputs/eval_causal_ode_camera \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```


</details>

#### Stage 2(b): Causal Consistency Distillation (Causal Forcing++)

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

Same as Stage 2(a).

**(2) Data preparation**

Same as §1.1 (2). Reuses the same encoded data.

**(3) Training script**

Consistency distillation: a frozen teacher and an EMA target network force the student to output a consistent prediction across adjacent timestep pairs (t, t_next). `trainer: consistency_distillation`:

```bash
bash HY15/scripts/training/hy15_camera/run_ar_causal_cd.sh
```

By default ckpts land at `./ckpts/HY15/Action2V/causal_cd/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/Action2V/causal_cd/`, matching the predownload layout used in §1.2 Stage 3 (1):

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/causal_cd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/causal_cd/
```

**(4) Validation**

Same as §1.2 Stage 2(a) (4); env-switch to this stage's output causal_cd:

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/causal_cd \
OUTPUT_DIR=./outputs/eval_causal_cd_camera \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```

</details>

#### Stage 3: Asymmetric DMD with Self Rollout

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/causal_ode/**" # Or causal_cd
```

**(2) Data preparation**

Same as §1.1 (2). Reuses the SFT latents encoded in §1.1 (`./dataset/HY15/Action2V/{latents, train_index.json}` and `./dataset/others/HY/Action2V/`). Stage 3 runs self rollout: DMD training only consumes the conditioning portion of `train_index.json` (first-frame image + caption + pose); it neither supervises against real video latents nor consumes the §1.2 Stage 2(a) ODE latents.

**(3) Training script**

Asymmetric DMD with self rollout: score distillation, conditioning only, no real-video supervision. The student aligns to the teacher's score field on its own rollouts:

```bash
bash HY15/scripts/training/hy15_camera/run_ar_hunyuan_dmd.sh
```

By default ckpts land at `./ckpts/HY15/Action2V/dmd/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/Action2V/dmd/`, matching the inference path used by the README Quick Start / §1.2 Stage 3 (4):

```bash
BEST_STEP=XXXX
mv ./ckpts/HY15/Action2V/dmd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/Action2V/dmd/
```

**(4) Validation**

Run the README Quick Start (HY Action2V) directly (the script defaults to `TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd`, this stage's output):

```bash
TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd \
OUTPUT_DIR=./outputs/eval_dmd_camera \
    bash HY15/scripts/inference/run_infer_causal_camera.sh
```

</details>

### 1.3 Using WorldPlay Checkpoints as Init (Discrete Action Mode)

You can download the official [tencent/HY-WorldPlay](https://huggingface.co/tencent/HY-WorldPlay) checkpoints and use them as initialization, skipping earlier stages. 
<details>
<summary><b>Expand</b> (download / training commands / parameter details)</summary>

**(1) Download**

```bash
# Bidirectional model (Phase 1 output → use as Stage 1 init)
hf download tencent/HY-WorldPlay --local-dir ./ckpts/HY-WorldPlay \
    --include "bidirectional_model/*"

# AR model (Stage 1 output → use as Stage 2 init)
hf download tencent/HY-WorldPlay --local-dir ./ckpts/HY-WorldPlay \
    --include "ar_model/*"
```

**(2) Training with discrete action**

Set `USE_DISCRETE_ACTION=True` and point `AR_ACTION_LOAD_FROM_DIR` to the downloaded checkpoint:

```bash
# Phase 1: Bidirectional camera + discrete action (for testing/debugging only)
USE_DISCRETE_ACTION=True \
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors \
    bash HY15/scripts/training/hy15_camera/run_bi_camera_multinode.sh

# Phase 2 Stage 1: AR teacher forcing + discrete action
USE_DISCRETE_ACTION=True \
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors \
    bash HY15/scripts/training/hy15_camera/run_ar_hunyuan_mem_multinode.sh

# Phase 2 Stage 2(a): Causal ODE + discrete action
USE_DISCRETE_ACTION=True \
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors \
    bash HY15/scripts/training/hy15_camera/run_ar_causal_ode.sh

# Phase 2 Stage 2(b): Causal CD + discrete action
USE_DISCRETE_ACTION=True \
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors \
    bash HY15/scripts/training/hy15_camera/run_ar_causal_cd.sh

# Phase 2 Stage 3: DMD + discrete action
USE_DISCRETE_ACTION=True \
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY15/Action2V/causal_ode/diffusion_pytorch_model.safetensors \
TEACHER_MODEL_PATH=./ckpts/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors \
    bash HY15/scripts/training/hy15_camera/run_ar_hunyuan_dmd.sh
```

Without `USE_DISCRETE_ACTION` (default `False`), behavior is identical to standard camera training (PRoPE only).

> Note: Phase 1 (bidirectional) is mainly for testing/debugging the discrete action pipeline. In production, start from Phase 2 Stage 1 with the official WorldPlay bidirectional checkpoint. The HY1.5 base model (`./ckpts/HunyuanVideo-1.5`) is still required for VAE / text encoder / scheduler — download it per §1.1.

**(3) `USE_DISCRETE_ACTION` parameter details**

When set to `True`, the training script passes `--use_discrete_action True` to the trainer. This triggers:
- FSDP loading stage: calls `add_discrete_action_parameters()` to create the `action_in` module (a `TimestepEmbedder` with 81 classes) on the transformer
- Training entry: verifies `action_in` exists on the loaded model; if missing, warns and adds as zero-init fallback
- Forward pass: discrete action labels (derived from viewmats) are embedded and added to the timestep vector

When `False` (default), if `action_in` is found on the model (e.g. loaded from a WorldPlay ckpt), it is removed with a warning to avoid unused parameters.

| | Standard camera | WorldPlay (`USE_DISCRETE_ACTION=True`) |
|---|---|---|
| Camera conditioning | PRoPE only | PRoPE + discrete action embedding |
| Extra parameters | `img_attn_prope_proj` per block | Same + `action_in` (TimestepEmbedder) |
| Data requirement | viewmats in dataset | Same (action labels derived automatically) |

</details>

---

## 2. HY TI2V

### 2.1 Phase 1: Bidirectional SFT

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

Same as §1.1 (1).

**(2) Data preparation**

Pick one. Everything lands under `./dataset/`.

**Option A: download minWM-dataset's pre-encoded latents**

> The HF remote directory is named `ODE_data/HY15/TI2V/` (toy data, ~5K, shared with the Causal ODE stage and named accordingly in the repo); after downloading, mv it to the local SFT layout `./dataset/HY15/TI2V/`.

```bash
# 1) Download from HF (the repo layout lands under ODE_data/)
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "ODE_data/HY15/TI2V/**"

# 2) Move to the unified layout ./dataset/HY15/TI2V/
mkdir -p ./dataset/HY15/TI2V
mv ./dataset/ODE_data/HY15/TI2V/latents ./dataset/HY15/TI2V/latents

# 3) Regenerate the absolute-path index in the new location
#    (writes to ./dataset/HY15/TI2V/train_index.json)
python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/TI2V/latents \
    -o ./dataset/HY15/TI2V/train_index.json
```

**Option B: bring your own videos + text (encode locally from raw video)**

Input `./dataset/videos.json` (replace paths with your own):

```json
[
    {"video_path": "/abs/path/to/video1.mp4", "caption": "A cat playing"},
    {"video_path": "/abs/path/to/video2.mp4", "caption": "A sunset"}
]
```

Script defaults: `HUNYUAN_CHECKPOINT=./ckpts/HunyuanVideo-1.5`, `INPUT_JSON=./dataset/videos.json`, `OUTPUT_DIR=./dataset/HY15/TI2V`:

```bash
bash HY15/scripts/data_preprocessing/run_preencode_video.sh
```

**Final: CFG negative prompt embeddings (shared by both options)**

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "others/HY/TI2V/**"
```

Final layout (unified at `./dataset/HY15/TI2V/`, identical for Option A/B):

```
./dataset/
├── HY15/TI2V/                           # encoded latents (A: moved from HF download; B: locally encoded)
│   ├── latents/
│   └── train_index.json
└── others/HY/TI2V/                      # HF neg prompts
    ├── hunyuan_neg_prompt.pt
    ├── hunyuan_neg_byt5_prompt.pt
    └── negative_prompt.pt
```

**(3) Training script**

Bidirectional TI2V SFT. Model class `HunyuanTransformer3DARActionModel`, global (non-causal) attention, flow matching MSE loss:

```bash
bash HY15/scripts/training/hyvideo15/run_bi_hunyuan_mem_multinode.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/bidirectional/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents (`diffusion_pytorch_model.safetensors` + `config.json` etc.) to `./ckpts/HY15/TI2V/bidirectional/`, matching the predownload layout used in §2.2 Stage 1 (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/bidirectional/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/bidirectional/
```

**(4) Validation**

Reuses the README Quick Start HY TI2V inference script in 50-step bidirectional mode. The script defaults to `TRANSFORMER_DIR=./ckpts/HY15/TI2V/bidirectional` (this stage's output, also the §2.2 Stage 1 predownload path); override via env to point at another ckpt:

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/bidirectional \
OUTPUT_DIR=./outputs/eval_bidir_ti2v \
    bash HY15/scripts/inference/run_infer_bidirectional.sh
```

</details>

### 2.2 Phase 2: Causal Forcing

#### Stage 1: Teacher Forcing AR Diffusion

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt (TI2V can also be initialized directly from the official bidirectional model; we still provide a fine-tuned version of ours for completeness):

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/TI2V/bidirectional/**"
```

**(2) Data preparation**

Same as §2.1 (2). Reuses the same `./dataset/HY15/TI2V/{latents, train_index.json}` and `./dataset/others/HY/TI2V/`.

**(3) Training script**

Convert the bidirectional TI2V model from Phase 1 into causal + teacher-forcing AR. Loss remains flow matching MSE:

```bash
bash HY15/scripts/training/hyvideo15/run_ar_hunyuan_mem_multinode.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/ar_diffusion_tf/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/TI2V/ar_diffusion_tf/`, matching the predownload layout used in §2.2 Stage 2(a) (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/ar_diffusion_tf/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/ar_diffusion_tf/
```

**(4) Validation**

50-step AR rollout mode. Script defaults `TRANSFORMER_DIR=./ckpts/HY15/TI2V/ar_diffusion_tf` (this stage's output, also the Stage 2(a) predownload path); other variables likewise overridable via env:

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/ar_diffusion_tf \
OUTPUT_DIR=./outputs/eval_ar_ti2v \
    bash HY15/scripts/inference/run_infer_ar_diffusion.sh
```

</details>

#### Stage 2(a): Causal ODE Distillation Initialization (Causal Forcing)

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/TI2V/ar_diffusion_tf/**"
```

**(2) Data preparation**

Pick one. Option A reuses the toy ODE data already downloaded in §2.1 (2); Option B samples ODE trajectories from the Stage 1 ckpt yourself (lands in a separate directory `./dataset/HY15/TI2V_ode/` to avoid mixing with §2.1). Negative prompts reuse the `./dataset/others/HY/TI2V/` already downloaded in §2.1.

**Option A: reuse the ODE data already downloaded in §2.1**

The `ODE_data/HY15/TI2V/` downloaded under §2.1 Option A (the HF repo names it that way; it is the toy ODE data shared with this stage) was moved to `./dataset/HY15/TI2V/{latents, train_index.json}`. This stage reuses it directly; no extra action needed.

> Difference vs. §1.2 Stage 2(a) (HY Action2V): in Action2V the SFT latents and the ODE latents are two independent datasets (in `./dataset/HY15/Action2V/` and `./dataset/HY15/Action2V_ode/` respectively); in TI2V the toy dataset is itself ODE data — §2.1 Option A already uses it as the SFT latents, so this stage simply shares that single copy.

**Option B: sample yourself from the Stage 1 ckpt**

Requires the §2.2 Stage 1 ckpt (at `./ckpts/HY15/TI2V/ar_diffusion_tf/`); run 48-step CFG sampling on the SFT-encoded data (§2.1):

```bash
AR_ACTION_LOAD_FROM_DIR=./ckpts/HY15/TI2V/ar_diffusion_tf/diffusion_pytorch_model.safetensors \
PREENCODED_DIR=./dataset/HY15/TI2V/train_index.json \
ODE_OUTPUT_PATH=./dataset/HY15/TI2V_ode/latents \
    bash HY15/scripts/ode_sampling/ode.sh

python HY15/scripts/data_preprocessing/create_train_index.py \
    ./dataset/HY15/TI2V_ode/latents \
    -o ./dataset/HY15/TI2V_ode/train_index.json
```

Final layout (Option B drop point):

```
./dataset/
├── HY15/
│   ├── TI2V/                            # SFT latents (from §2.1)
│   │   ├── latents/
│   │   └── train_index.json
│   └── TI2V_ode/                        # ODE latents (Option B output here)
│       ├── latents/
│       └── train_index.json
└── others/HY/TI2V/                      # neg prompts (reused from §2.1)
```

**(3) Training script**

ODE regression: on the ODE latents prepared in §2.2 Stage 2(a) (2), have the model directly regress the ODE solver outputs at key timesteps:

```bash
bash HY15/scripts/training/hyvideo15/run_ar_causal_ode.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/causal_ode/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/TI2V/causal_ode/`, matching the predownload layout used in §2.2 Stage 3 (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/causal_ode/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/causal_ode/
```

**(4) Validation**

The 4-step DMD inference script (`run_infer_causal.sh`) is shared by Stage 2(a) / 2(b) / 3, defaulting to `TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd` (the final output, matching the README Quick Start). For this stage, override via env to causal_ode:

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/causal_ode \
OUTPUT_DIR=./outputs/eval_causal_ode_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

</details>

#### Stage 2(b): Causal Consistency Distillation (Causal Forcing++)

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

Same as Stage 2(a).

**(2) Data preparation**

Same as §2.1 (2). Reuses the same encoded data.

**(3) Training script**

Consistency distillation. `trainer: consistency_distillation`; with a frozen teacher and an EMA target network, the student is forced to output a consistent prediction across adjacent timestep pairs (t, t_next):

```bash
bash HY15/scripts/training/hyvideo15/run_ar_causal_cd.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/causal_cd/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/TI2V/causal_cd/`, matching the predownload layout used in §2.2 Stage 3 (1):

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/causal_cd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/causal_cd/
```

**(4) Validation**

Same as §2.2 Stage 2(a) (4); env-switch to this stage's output causal_cd:

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/causal_cd \
OUTPUT_DIR=./outputs/eval_causal_cd_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

</details>

#### Stage 3: Asymmetric DMD with Self Rollout

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/TI2V/causal_ode/**" # or causal_cd
```

**(2) Data preparation**

Same as §2.1 (2). Reuses the SFT latents encoded in §2.1: `./dataset/HY15/TI2V/{latents, train_index.json}` and `./dataset/others/HY/TI2V/`. Stage 3 runs self rollout: DMD training only consumes the conditioning portion of `train_index.json` (first-frame image + caption); it does not supervise against real video latents.

**(3) Training script**

Asymmetric DMD with self rollout. Score distillation, consuming only the conditioning portion of `train_index.json` (first-frame image + caption); no real-video supervision:

```bash
bash HY15/scripts/training/hyvideo15/run_ar_hunyuan_dmd.sh
```

By default ckpts land at `./ckpts/HY15/TI2V/dmd/checkpoint-XXXX/transformer/`. After training, pick the best step and promote the entire `transformer/` contents to `./ckpts/HY15/TI2V/dmd/`, matching the inference path used by the README Quick Start:

```bash
BEST_STEP=XXXX  # selected by validation
mv ./ckpts/HY15/TI2V/dmd/checkpoint-${BEST_STEP}/transformer/* \
   ./ckpts/HY15/TI2V/dmd/
```

**(4) Validation**

Run the README Quick Start (HY TI2V) directly (the script defaults to `TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd`, this stage's output):

```bash
TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd \
OUTPUT_DIR=./outputs/eval_dmd_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh
```

</details>
