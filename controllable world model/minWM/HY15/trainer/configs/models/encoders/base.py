# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

import torch

from trainer.configs.models.base import ArchConfig, ModelConfig


@dataclass
class EncoderArchConfig(ArchConfig):
    architectures: list[str] = field(default_factory=lambda: [])
    output_hidden_states: bool = False
    use_return_dict: bool = True


@dataclass
class BaseEncoderOutput:
    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    attention_mask: torch.Tensor | None = None


@dataclass
class EncoderConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=EncoderArchConfig)

    prefix: str = ""
    quant_config: Any | None = None
    lora_config: Any | None = None
