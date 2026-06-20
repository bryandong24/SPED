# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE

from typing import Optional
from dataclasses import dataclass

@dataclass
class InferState:
    enable_sageattn: bool = False  # whether to use SageAttention
    sage_blocks_range: Optional[range] = None  # block range to use SageAttention

_default_infer_state = InferState()

def get_infer_state():
    return _default_infer_state
