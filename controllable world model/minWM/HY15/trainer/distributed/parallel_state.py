# SPDX-License-Identifier: Apache-2.0
"""
Re-export shim: all parallel state lives in shared/sp/parallel_state.py.

This module re-exports every public symbol so that existing code using
`from trainer.distributed.parallel_state import X` continues to work,
while the actual _SP/_TP/_DP globals live in a single place (shared/sp).
"""

from sp.parallel_state import (  # noqa: F401
    # Classes
    GraphCaptureContext,
    GroupCoordinator,
    # World group
    get_world_group,
    init_world_group,
    init_model_parallel_group,
    # TP
    get_tp_group,
    set_custom_all_reduce,
    get_tp_world_size,
    get_tp_rank,
    # SP
    get_sp_group,
    get_sp_world_size,
    get_sp_parallel_rank,
    # DP
    get_dp_group,
    get_dp_world_size,
    get_dp_rank,
    # World
    get_world_size,
    get_world_rank,
    get_local_torch_device,
    # Initialization
    init_distributed_environment,
    initialize_model_parallel,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
    # Lifecycle
    destroy_model_parallel,
    destroy_distributed_environment,
    cleanup_dist_env_and_memory,
    # Utilities
    patch_tensor_parallel_group,
    is_the_same_node_as,
    # Extended init
    initialize_tensor_parallel_group,
    initialize_sequence_parallel_group,
)
