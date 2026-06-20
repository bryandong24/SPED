# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""
Thin wrapper that delegates to shared/sp.

All parallel state is managed by the shared SP infrastructure.
HY15 code continues to call get_parallel_state() / initialize_parallel_state()
as before — this module just forwards to shared/sp.
"""

from sp.parallel_states import (
    ParallelDims,
    get_parallel_state,
)
from sp.parallel_state import maybe_init_distributed_environment_and_model_parallel

__all__ = ["ParallelDims", "get_parallel_state", "initialize_parallel_state"]


def initialize_parallel_state(sp: int = 1):
    """Initialize shared SP groups and return the ParallelDims shim."""
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=sp)
    return get_parallel_state()
