# 🌍 minWM: Full-Stack Open-Source Video World Model Framework

>  ***A full-stack framework and tutorial for newcomers, rather than a specific model.***

<p align="center">
  <a href="https://arxiv.org/abs/2605.30263"><img src="https://img.shields.io/badge/Technical_Report-arXiv-b31b1b?logo=arxiv&logoColor=white" alt="Technical Report"></a>
  <a href="https://huggingface.co/MIN-Lab/minWM"><img src="https://img.shields.io/badge/Hugging_Face-Models-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

**minWM** is our contribution to the world-model community: a **full-stack open-source framework** that walks you end-to-end through turning a bidirectional T2V foundation model into an action-conditioned video world model — with example data, runnable scripts, **Claude skills** capturing our hands-on experience, and **onboarding knowledge** for newcomers. We hope more researchers and developers join us in growing the community together.

## 🎬 Demo

https://github.com/user-attachments/assets/99c25915-7fe7-4a20-a2c4-9d291502fccf

## 🔥 News

- **2026-05-29** 🚀 We release the [technical report](https://arxiv.org/pdf/2605.30263).
- **2026-05-17** 🚀 We release **minWM** — the first full-stack open-source world model framework.


## 📋 Table of Contents

- [🎬 Demo](#-demo)
- [🔥 News](#-news)
- [✨ Why minWM?](#-why-minwm)
  - [1. Full-Stack Framework](#1-full-stack-framework)
  - [2. Multi-Backbone Support](#2-multi-backbone-support)
  - [3. Multi-Condition Injection](#3-multi-condition-injection)
  - [4. Claude Skills — Modify the Framework with an LLM Assistant](#4-claude-skills--modify-the-framework-with-an-llm-assistant)
  - [5. Onboarding Knowledge — for Newcomers to World Models](#5-onboarding-knowledge--for-newcomers-to-world-models)
- [🛠️ Installation](#️-installation)
- [🧱 Model Checkpoints](#-model-checkpoints)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Data & Training & Reproduction](#️-data--training--reproduction)
- [📚 Citation](#-citation)
- [Contact](#contact)
- [🙏 Acknowledgements](#-acknowledgements)

## ✨ Why minWM?

### 1. Full-Stack Framework

The complete **data → training → inference** pipeline is open-sourced; every stage exposes input/output checkpoints so you can stop, swap, or fork anywhere.

**1.1 Data.** We walk you through how to construct training-ready datasets paired with camera poses, and the full data processing pipeline that turns them into latents.

**1.2 Training.** Including FSDP + sequence parallelism, single-/multi-node training, and the full distillation pipeline from a bidirectional diffusion model to a 4-step AR student:

```
Phase 1                            Phase 2 — Distillation to Causal Few-Step
─────────────────────              ────────────────────────────────────────────
Bidirectional SFT      ──▶   Stage 1   Teacher Forcing AR Diffusion
                             Stage 2a  Causal ODE  (proposed in [Causal Forcing](https://arxiv.org/abs/2602.02214))
                             Stage 2b  Causal CD   (proposed in [Causal Forcing++](https://arxiv.org/abs/2605.15141))
                             Stage 3   Asymmetric DMD with Self Rollout
                                                ▼
                                         4-step real-time
```

**1.3 Inference.**

- ✅ 4-step DMD inference for HY Action2V / HY TI2V / Wan Action2V, multi-GPU sequence parallelism, camera-trajectory control via pose strings (`"a*4,w*8,s*7"`) or JSON files
- 🚧 Inference acceleration [TBD]

### 2. Multi-Backbone Support

minWM supports two paths to arriving at an interactive world model.

#### 2.1 From Scratch: Bidirectional T2V Foundation → Real-Time World Model

The HunyuanVideo 1.5 and Wan 2.1 lines walk through the full 4-stage pipeline — starting from a bidirectional T2V foundation model and ending at a 4-step autoregressive world model.


| Backbone             | Architecture          | Params | Training       | Inference    |
| -------------------- | --------------------- | ------ | -------------- | ------------ |
| **Wan 2.1**          | Cross-attention + DiT | 1.3 B  | ✅ all 4 stages | ✅ 4-step DMD |
| **HunyuanVideo 1.5** | MMDiT                 | 8 B    | ✅ all 4 stages | ✅ 4-step DMD |



Both lines share the same trainer / loss / dataset abstractions, so adding a third backbone is structurally a wrapper-and-config exercise.

#### 2.2 Finetuning an Existing Video World Model 🚧 [TBD]

The forthcoming `worldplay-finetune` entry will let you start from an already-trained video world model and adapt it to new conditions, scenes, or resolutions — without rerunning the 4-stage pipeline from scratch.

### 3. Multi-Condition Injection

We aim to support both multiple condition types and multiple injection methods, mixable along either axis.

#### 3.1 Supported Conditions

- ✅ Camera pose
- 🚧 Human pose [TBD]

#### 3.2 Supported Injection Methods

- ✅ ProPE
- 🚧 Latent concat [TBD]
- 🚧 Cross-attention [TBD]

### 4. Claude Skills — Modify the Framework with an LLM Assistant
We are packaging our project experience across the CF / CF++ pipeline as Claude skills, so that an LLM assistant can help users debug failures and integrate new models without reverse-engineering the whole repo.

- 🐛 **`debug-world-model`** — collected failure modes from the training pipeline (loss NaN, frame-to-frame jitter, camera drift, memory attenuation, distillation collapse, …). Claude diagnoses likely root causes from your symptoms instead of guessing.
- 🔌 **`integrate-new-backbone`** — step-by-step recipe for plugging a new video DiT into minWM, grounded in the HunyuanVideo and Wan reference integrations — e.g. *"look at how HY does teacher forcing here, do the same for your model there"*.

### 5. Onboarding Knowledge — for Newcomers to World Models

- `onboarding-world-model`

A third Claude skill aimed at researchers entering the world-model space for the first time. Two parts:

- 🎓 **Foundations** — the minimal background to follow the pipeline: Teacher Forcing for AR diffusion training and Causal Forcing & Causal Forcing++ for AR diffusion distillation.
- 🪤 **Pitfalls** — the non-obvious mistakes we hit while building minWM, distilled so you don't repeat them.

Intended audience: graduate students, independent researchers, and junior labs that want to enter the world-model space without spending three months reverse-engineering existing repos.

## 🛠️ Installation

```bash
conda create -n minwm python=3.10 -y 
conda activate minwm
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:$PYTHONPATH"
```

<details> <summary> 🧱 Model Checkpoints (Click to expand) </summary> 

All weights live under `./ckpts/` after download.


| Checkpoint                                                                | Backbone | Stage                               | Use case                               | Download                                              |
| ------------------------------------------------------------------------- | -------- | ----------------------------------- | -------------------------------------- | ----------------------------------------------------- |
| `Wan21/Action2V/{bidirectional,ar_diffusion_tf,causal_ode,causal_cd,dmd}` | Wan 2.1  | Same 4 stages                       | Wan pipeline                           | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HunyuanVideo-1.5` (base)                                                 | HY 1.5   | —                                   | Required by both HY pipelines          | [HF](https://huggingface.co/tencent/HunyuanVideo-1.5) |
| `Wan2.1-T2V-1.3B` (base)                                                  | Wan 2.1  | —                                   | Required by Wan pipeline               | [HF](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)   |
| `HY15/Action2V/bidirectional`                                             | HY 1.5   | Phase 1 SFT                         | Starting point for HY Action2V Phase 2 | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HY15/Action2V/ar_diffusion_tf`                                           | HY 1.5   | Phase 2 Stage 1                     | Teacher Forcing AR diffusion           | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HY15/Action2V/causal_ode`                                                | HY 1.5   | Phase 2 Stage 2a (proposed in Causal Forcing)   | DMD initialization               | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HY15/Action2V/causal_cd`                                                 | HY 1.5   | Phase 2 Stage 2b (proposed in Causal Forcing++) | DMD initialization               | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HY15/Action2V/dmd`                                                       | HY 1.5   | Phase 2 Stage 3                     | **4-step real-time inference**         | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HY15/TI2V/{bidirectional,ar_diffusion_tf,causal_ode,causal_cd,dmd}`      | HY 1.5   | Same 4 stages, TI2V variant         | TI2V pipeline                          | [HF](https://huggingface.co/MIN-Lab/minWM)            |

</details>

## 🚀 Quick Start

> The fastest path: install → download three DMD checkpoints → run three demo commands. Full reproduction (all 4 training stages × 3 model lines) is in [§ Data & Training & Reproduction](#️-data--training--reproduction).

### 1. Download the demo checkpoints

```bash
# Wan base (T2V-1.3B)
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpts/Wan2.1-T2V-1.3B 

# Code hardcodes the load path; create a symlink.
mkdir -p Wan21/wan_models
ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" Wan21/wan_models/Wan2.1-T2V-1.3B


# HY base + text/vision encoders (required by HY pipelines)
hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts/HunyuanVideo-1.5 \
    --include "vae/*"  "scheduler/*" "transformer/480p_i2v/*"
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/llm
hf download google/byt5-small           --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 \
    --local_dir ./ckpts/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2
hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir ./ckpts/HunyuanVideo-1.5/vision_encoder/siglip --token <your_hf_token>


# 4-step DMD checkpoints
## Wan Action2V (DMD, 4-step)
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/dmd/*"

## HY Action2V (DMD, 4-step, worldplay teacher) 
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/dmd/*"

# HY Action2V (DMD, 4-step, our bidirectional teacher) 
# hf download MIN-Lab/minWM --local-dir ./ckpts \
#     --include "HY15/Action2V/dmd_ourbi/*"

## HY TI2V (DMD, 4-step)
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/TI2V/dmd/*"
```


### 2. Run the three demos

```bash
# 2.1  Wan Action2V (4-step DMD, camera control)
OUTPUT_FOLDER=./outputs/quickstart_wan_action2v \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh

# 2.2  HY Action2V (4-step DMD, camera control)
TRANSFORMER_DIR=./ckpts/HY15/Action2V/dmd \
OUTPUT_DIR=./outputs/quickstart_hy_action2v \
    bash HY15/scripts/inference/run_infer_causal_camera.sh

# 2.3  HY TI2V (4-step DMD)
TRANSFORMER_DIR=./ckpts/HY15/TI2V/dmd \
OUTPUT_DIR=./outputs/quickstart_hy_ti2v \
    bash HY15/scripts/inference/run_infer_causal.sh

```

> **Camera control.** For HY Action2V, trajectories are read per-sample from `assets/example.json` under the `"trajectory"` field. Format: `w/s/a/d` keys with `*N` repeats; comma-separated segments — e.g. `"a*4,w*8,s*7"`.

## ⚙️ Data & Training & Reproduction

Three model lines × two phases × four stages, each documented as **(1) Model download → (2) Data preparation → (3) Training script → (4) Validation**. Full reproduction guides are split by backbone:

- 📗 [`training_wan.md`](training_wan.md)
    -  **Wan Action2V**  (Wan 2.1 backbone)
- 📘 [`training_hunyuan.md`](training_hunyuan.md)
    — **HY Action2V** (HY1.5-8B backbone)
    - **HY TI2V** (HY1.5-8B backbone)

## 📚 Citation

If minWM helps your research, please cite:

```bibtex

# ICML 2026
@article{zhu2026causal,
  title={Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation},
  author={Zhu, Hongzhou and Zhao, Min and He, Guande and Su, Hang and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2602.02214},
  year={2026}
}

# Technical Report
@article{zhao2026causal,
  title={Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation},
  author={Zhao, Min and Zhu, Hongzhou and Zheng, Kaiwen and Zhou, Zihan and Yan, Bokai and Li, Xinyuan and Yang, Xiao and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2605.15141},
  year={2026}
}

# Technical Report
@article{zhao2026minwm,
  title={minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models},
  author={Zhao, Min and Zhu, Hongzhou and Yan, Bokai and Zhou, Zihan and Chen, Yimin and Sun, Wenqiang and Zheng, Kaiwen and He, Guande and Yang, Xiao and Li, Chongxuan and others},
  journal={arXiv preprint arXiv:2605.30263},
  year={2026}
}

```

## Contact

For questions, suggestions, or collaboration, please open a GitHub issue or contact: [gracezhao1997@gmail.com](mailto:gracezhao1997@gmail.com).

## 🙏 Acknowledgements

minWM stands on the shoulders of giants. We thank the authors and maintainers of [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay), [Wan 2.1](https://github.com/Wan-AI/Wan), [Causal-Forcing](https://github.com/thu-ml/Causal-Forcing), and [FastVideo](https://github.com/hao-ai-lab/FastVideo) for their open-source contributions, which made this framework possible.
