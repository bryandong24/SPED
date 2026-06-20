"""Load the LongLive-1.3B pipeline (base + LoRA), reusing the repo's loader.

Factored out of LongLive-v1/interactive_inference.py so we can drive the model
from a custom streaming loop.
"""
import os, sys
import torch
from omegaconf import OmegaConf

LONGLIVE_DIR = "/mnt/data/SPED/gino/LongLive-v1"


def load_pipeline(config_path=None, device=None):
    if LONGLIVE_DIR not in sys.path:
        sys.path.insert(0, LONGLIVE_DIR)
    os.chdir(LONGLIVE_DIR)  # configs reference relative paths (longlive_models/...)
    from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model
    import peft

    device = device or torch.device("cuda")
    config_path = config_path or os.path.join(LONGLIVE_DIR, "configs/longlive_interactive_inference.yaml")
    config = OmegaConf.load(config_path)
    torch.set_grad_enabled(False)

    pipeline = InteractiveCausalInferencePipeline(config, device=device)
    if config.generator_ckpt:
        sd = torch.load(config.generator_ckpt, map_location="cpu")
        pipeline.generator.load_state_dict(sd["generator_ema" if config.use_ema else "generator"])

    pipeline.is_lora_enabled = False
    if getattr(config, "adapter", None):
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model, model_name="generator",
            lora_config=config.adapter, is_main_process=True)
        lc_path = getattr(config, "lora_ckpt", None)
        if lc_path:
            lc = torch.load(lc_path, map_location="cpu")
            sd = lc["generator_lora"] if isinstance(lc, dict) and "generator_lora" in lc else lc
            peft.set_peft_model_state_dict(pipeline.generator.model, sd)
        pipeline.is_lora_enabled = True

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    return pipeline, config
