# minWM Training — Wan 2.1 Backbone

Two model lines built on Wan 2.1: **Wan Action2V** (camera-controlled action-to-video) and **Wan T2V** (text to video, not yet integrated in this repo).

Wan Action2V splits into two phases:
- **Phase 1 Bidirectional SFT** — bidirectional multi-step base.
- **Phase 2 Causal Forcing** — distillation to causal few-step.

Phase 2 has 4 stages: Stage 1 Teacher Forcing AR Diffusion, Stage 2(a) Causal ODE Distillation Initialization (Causal Forcing), Stage 2(b) Causal Consistency Distillation (Causal Forcing++), Stage 3 Asymmetric DMD with Self Rollout.

Every subsection follows the same structure: **(1) Model download**, **(2) Data preparation**, **(3) Training script**, **(4) Validation**.

> HunyuanVideo-backbone training lives in [`training_hunyuan.md`](training_hunyuan.md). Quick Start / inference commands live in the main [README](README.md).

First install:

```bash
conda create -n minwm python=3.10 -y 
conda activate minwm
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:$PYTHONPATH"
```

---

## 1. Wan Action2V

### 1.1 Phase 1: Bidirectional SFT

> By default, 16 machines ensure gbs=32. If there are fewer machines, you need to disable sp and enable ga. Same below.

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir ./ckpts/Wan2.1-T2V-1.3B \
    --include "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" "google/umt5-xxl/*" "diffusion_pytorch_model.safetensors" "config.json"
```

**(2) Data preparation**

Video source (Option A download / Option B bring your own) and disk layout are **the same as the HY Action2V Phase 1 (2)** ([`training_hunyuan.md` §1.1](training_hunyuan.md#11-phase-1-bidirectional-sft)), reusing the same `./dataset/preencode_input.json` + `./dataset/videos/`. The difference is only in encoding:

- HY encoding lands at `<OUTPUT_DIR>/latents/` (per-sample `.pt`),
- Wan encoding lands at `<OUTPUT_DIR>/data/` (a merged LMDB); the downstream training entry reads LMDB directly.

**Encoding (shared by both options)**

The VAE uses the Wan2.1 base model (already symlinked per the README Installation step to `Wan21/wan_models/Wan2.1-T2V-1.3B/`). Script defaults:

```
VAE_PATH=Wan21/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
INPUT_JSON=./dataset/preencode_input.json
VIDEO_DIR=./dataset/videos
OUTPUT_DIR=./dataset/Wan21/Action2V
```

Run directly:

```bash
bash Wan21/scripts/data_preprocessing/run_build_worldplaygen_lmdb.sh
```

The merged LMDB lands at `./dataset/Wan21/Action2V/data/`.

> Wan's CFG negative prompt is written into the config (`Wan21/configs/causal_forcing_dmd_camera.yaml:38`), so no `.pt` preencoding is needed; hence no equivalent of HY's `others/HY/Action2V/` download step.

Final layout:

```
./dataset/
├── preencode_input.json                 # same as training_hunyuan.md §1.1
├── videos/                              # same as training_hunyuan.md §1.1
└── Wan21/Action2V/
    └── data/                            # encoded LMDB
```

**(3) Training script**

Bidirectional + camera (PRoPE) SFT. `trainer: bidirectional_diffusion`, using `WanDiffusionWrapper(use_camera=True)` and `CameraLatentLMDBDataset` (carrying viewmats/Ks); flow matching loss:

```bash
bash Wan21/scripts/training/run_stage0_bidirectional_camera.sh
```

By default ckpts land at `logs/bidirectional_camera/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/bidirectional/`, matching the predownload layout used in §1.2 Stage 1 (1):

```bash
BEST_STEP=00x000  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/bidirectional
mv logs/bidirectional_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/bidirectional/model.pt
```

**(4) Validation**

50-step bidirectional + camera mode. Script defaults `CHECKPOINT_PATH=./ckpts/Wan21/Action2V/bidirectional/model.pt` (this stage's output, also the §1.2 Stage 1 predownload path); override via env to point at another ckpt:

```bash 
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/bidirectional/model.pt \
OUTPUT_FOLDER=./outputs/eval_bidir_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_bidirectional_camera.sh
```

</details>

### 1.2 Phase 2: Causal Forcing

#### Stage 1: Teacher Forcing AR Diffusion

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/bidirectional/**"
```

**(2) Data preparation**

Same as §1.1 (2). Reuses the same `./dataset/Wan21/Action2V/data/` (LMDB).

**(3) Training script**

Convert the bidirectional model from Phase 1 into causal + teacher-forcing AR. Switch the model to `CausalWanModel`, keep `use_camera: true` to retain PRoPE projection parameters; flow matching loss:

```bash
bash Wan21/scripts/training/run_stage1_ar_camera.sh
```

By default ckpts land at `logs/ar_camera_tf/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/ar_diffusion_tf/`, matching the predownload layout used in §1.2 Stage 2(a) (1):

```bash
BEST_STEP=00x000  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/ar_diffusion_tf
mv logs/ar_camera_tf/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```

**(4) Validation**

50-step AR + camera mode. Script defaults `CHECKPOINT_PATH=./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt` (this stage's output, also the Stage 2(a) predownload path):

```bash 
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt \
OUTPUT_FOLDER=./outputs/eval_ar_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_ar_camera.sh
```

</details>

#### Stage 2(a): Causal ODE Distillation Initialization (Causal Forcing)

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**


If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/ar_diffusion_tf/**"
```

**(2) Data preparation**

Pick one. Everything lands at `./dataset/Wan21/Action2V/ode_lmdb/`, parallel to and not overlapping the §1.1 SFT LMDB (`./dataset/Wan21/Action2V/data/`).

**Option A: download minWM-dataset's pre-generated ODE latents and merge LMDB locally**

What HF publishes is the **unmerged `.pt` latents** (the output of `get_causal_ode_data_prope.py`); after downloading, run a local merge:

```bash
# 1) Download the .pt latents (HF repo layout lands under ODE_data/)
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "ODE_data/Wan21/Action2V/**"

# 2) Merge into the training-ready LMDB
python Wan21/wan_utils/build_ode_prope_lmdb.py \
    --input_dir ./dataset/ODE_data/Wan21/Action2V \
    --output_dir ./dataset/Wan21/Action2V/ode_lmdb \
    --map_size_gb 10000
```

**Option B: sample yourself from the Stage 1 ckpt**

Requires the §1.2 Stage 1 ckpt (at `./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt`); run 48-step CFG sampling on the §1.1-encoded SFT LMDB (`./dataset/Wan21/Action2V/data/`), then merge into LMDB:

```bash
# 1) Run 48-step CFG sampling with the Stage 1 ckpt to get .pt latents
torchrun --nproc_per_node=8 Wan21/get_causal_ode_data_prope.py \
    --generator_ckpt ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt \
    --rawdata_path ./dataset/Wan21/Action2V/data \
    --output_folder ./dataset/Wan21/Action2V/ode_latents

# 2) Merge into the training-ready LMDB
python Wan21/wan_utils/build_ode_prope_lmdb.py \
    --input_dir ./dataset/Wan21/Action2V/ode_latents \
    --output_dir ./dataset/Wan21/Action2V/ode_lmdb \
    --map_size_gb 10000
```

> Difference vs. HY Action2V Stage 2(a) ([`training_hunyuan.md` §1.2 Stage 2(a)](training_hunyuan.md#stage-2a-causal-ode-distillation-initialization-causal-forcing)): HY's ODE outputs are per-sample `.pt`, so after HF download only `mv` + regenerating `train_index.json` is needed; for Wan, since the training entry consumes LMDB directly, **both options additionally require running `build_ode_prope_lmdb.py` to merge**.

Final layout:

```
./dataset/
├── preencode_input.json                # same as §1.1
├── videos/                             # same as §1.1
└── Wan21/Action2V/
    ├── data/                           # SFT LMDB (from §1.1)
    └── ode_lmdb/                       # ODE LMDB (output of this section, read directly by the training entry)
```

**(3) Training script**

ODE regression. `trainer: ode`; trains on the ODE LMDB prepared in §1.2 Stage 2(a) (2) (`./dataset/Wan21/Action2V/ode_lmdb/`), read by `CameraODERegressionLMDBDataset` and regressing the ODE solver outputs at key timesteps [1000, 750, 500, 250]:

```bash
bash Wan21/scripts/training/run_stage2_causal_ode_camera.sh
```

By default ckpts land at `logs/causal_ode_camera/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/causal_ode/`, matching the predownload layout used in §1.2 Stage 3 (1):

```bash
BEST_STEP=00x000  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/causal_ode
mv logs/causal_ode_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/causal_ode/model.pt
```

**(4) Validation**

`run_infer_causal_camera.sh` defaults to the 4-step DMD config + dmd ckpt (matching the README Quick Start). For this stage, env-switch both `CONFIG_PATH` and `CHECKPOINT_PATH` to the ODE version:

```bash 
CONFIG_PATH=Wan21/configs/causal_ode_camera.yaml \
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/causal_ode/model.pt \
OUTPUT_FOLDER=./outputs/eval_causal_ode_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
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

Consistency distillation. `trainer: consistency_distillation`; with a frozen teacher and an EMA target network, the student is forced to output a consistent prediction across adjacent timestep pairs (t, t_next):

```bash
bash Wan21/scripts/training/run_stage2_causal_cd_camera.sh
```

By default ckpts land at `logs/causal_cd_camera/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/causal_cd/`, matching the predownload layout used in §1.2 Stage 3 (1):

```bash
BEST_STEP=00x000  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/causal_cd
mv logs/causal_cd_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/causal_cd/model.pt
```

**(4) Validation**

Same as §1.2 Stage 2(a) (4); env-switch to the CD config + CD ckpt:

```bash 
CONFIG_PATH=Wan21/configs/causal_cd_camera.yaml \
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/causal_cd/model.pt \
OUTPUT_FOLDER=./outputs/eval_causal_cd_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

</details>

#### Stage 3: Asymmetric DMD with Self Rollout

<details>
<summary><b>Expand</b> (model download / data preparation / training script / validation)</summary>

**(1) Model download**

If skipping the previous stage, download the provided prior-stage ckpt:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/causal_ode/**"  # Or causal_cd
```

**(2) Data preparation**

Same as §1.1 (2). Reuses the SFT LMDB encoded in §1.1 (`./dataset/Wan21/Action2V/data/`). Stage 3 runs self rollout and no longer consumes the Stage 2(a) ODE LMDB.

**(3) Training script**

Asymmetric DMD with self rollout. `trainer: score_distillation`; conditioning only, no real-video supervision. The student aligns to the teacher's score field on its own rollouts:

```bash
bash Wan21/scripts/training/run_stage3_causal_dmd_camera.sh # 100~200 steps recommended
```

By default ckpts land at `logs/causal_dmd_camera/checkpoint_model_<step>/model.pt`. After training, pick the best step and promote `model.pt` to `./ckpts/Wan21/Action2V/dmd/`, matching the inference path used by the README Quick Start / §1.2 Stage 3 (4):

```bash
BEST_STEP=000x00  # selected by validation (zero-padded to 6 digits)
mkdir -p ./ckpts/Wan21/Action2V/dmd
mv logs/causal_dmd_camera/checkpoint_model_${BEST_STEP}/model.pt \
   ./ckpts/Wan21/Action2V/dmd/model.pt
```

**(4) Validation**

Run the README Quick Start (Wan Action2V) directly (the script defaults to `CONFIG_PATH=causal_forcing_dmd_camera.yaml` and `CHECKPOINT_PATH=./ckpts/Wan21/Action2V/dmd/model.pt`, this stage's output):


```bash 
CHECKPOINT_PATH=./ckpts/Wan21/Action2V/dmd/model.pt \
OUTPUT_FOLDER=./outputs/eval_dmd_wan \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

</details>

---

## 2. Wan T2V

Not yet integrated in this repo. See the [Causal-Forcing repo](https://github.com/thu-ml/Causal-Forcing) for training, inference, data, and models.
