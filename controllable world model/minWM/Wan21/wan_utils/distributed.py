from datetime import timedelta
from functools import partial
import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

# ---------------------------------------------------------------------------
# CleanCode import helpers
# ---------------------------------------------------------------------------
# SP infra lives in shared/sp/. We ensure shared/ is on sys.path so that
# ``import sp.*`` works.
# ---------------------------------------------------------------------------

_SHARED_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
_shared_path_added = False


def _ensure_shared_path():
    """Ensure shared/ is on sys.path so ``import sp.*`` works."""
    global _shared_path_added
    if _shared_path_added:
        return
    _shared_path_added = True
    if _SHARED_ROOT not in sys.path:
        sys.path.insert(0, _SHARED_ROOT)


def _get_cleancode_parallel_state():
    _ensure_shared_path()
    import importlib
    return importlib.import_module("sp.parallel_state")


def _get_cleancode_sp_states():
    _ensure_shared_path()
    import importlib
    return importlib.import_module("sp.parallel_states")


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False, process_group=None):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    # HYBRID_SHARD requires process_group as Tuple[ProcessGroup, ProcessGroup].
    # When SP is active, get_fsdp_process_group() returns a single DP ProcessGroup.
    # Downgrade to FULL_SHARD within the DP group — equivalent memory savings.
    if process_group is not None and not isinstance(process_group, tuple) and \
            sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        sharding_strategy = ShardingStrategy.FULL_SHARD

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,  # Load ckpt on rank 0 and sync to other ranks
        process_group=process_group,
    )
    return module


def get_fsdp_process_group():
    """Return the FSDP process group.

    When SP is active, FSDP must shard only across the DP sub-group
    (not the world group) to avoid conflicts with SP all-to-all.
    Returns None when SP is not active (FSDP uses world group by default).
    """
    try:
        _ps = _get_cleancode_parallel_state()
        if _ps.model_parallel_is_initialized():
            _sp = _get_cleancode_sp_states()
            if _sp.get_parallel_state().sp_enabled:
                return _ps.get_dp_group().device_group
    except (ImportError, FileNotFoundError, AttributeError):
        pass
    return None


def get_sp_data_sampler(dataset, shuffle=True, drop_last=True):
    """Create a DistributedSampler that is SP-aware.

    When SP is active, ranks within the same SP group must receive the same
    data (they process different chunks of the same sequence). This is achieved
    by using DP rank/world_size instead of global rank/world_size for the sampler,
    so all ranks in an SP group map to the same DP rank and get identical indices.
    Returns a standard DistributedSampler when SP is not active.
    """
    try:
        _ps = _get_cleancode_parallel_state()
        _sp = _get_cleancode_sp_states()
        if _ps.model_parallel_is_initialized() and _sp.get_parallel_state().sp_enabled:
            return torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=_ps.get_dp_world_size(),
                rank=_ps.get_dp_rank(),
                shuffle=shuffle,
                drop_last=drop_last,
            )
    except (ImportError, FileNotFoundError, AttributeError):
        pass
    return torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=shuffle, drop_last=drop_last
    )


def barrier():
    if dist.is_initialized():
        dist.barrier()


def _sync_sp_seeds(sp_size: int):
    """Synchronize CUDA and Python random seeds within each SP group.

    When SP is active, ranks in the same SP group must produce identical
    random tensors (noise, timesteps, etc.) so that sequence-parallel
    chunks correspond to the same video.  We achieve this by giving every
    rank in an SP group the same CUDA RNG seed and Python random seed.

    NOTE: This is called from launch_distributed_job *before* training __init__,
    but trainers call set_seed(config.seed + rank_offset) later.  The rank_offset
    must use get_sp_seed_offset() (DP rank) instead of global rank so that
    SP-group peers keep the same seed.
    """
    import random
    rank = dist.get_rank()
    if rank == 0:
        print(f"[SP] SP seed sync will be handled by set_seed with DP-rank offset (sp_size={sp_size})")


def get_sp_seed_offset():
    """Return the rank offset to use when seeding RNGs.

    Under SP, ranks in the same SP group must share the same seed so they
    produce identical random tensors.  This is achieved by using the DP rank
    (identical for all ranks in an SP group) instead of the global rank.
    Returns global rank when SP is not active.
    """
    try:
        _ps = _get_cleancode_parallel_state()
        _sp = _get_cleancode_sp_states()
        if _ps.model_parallel_is_initialized() and _sp.get_parallel_state().sp_enabled:
            return _ps.get_dp_rank()
    except (ImportError, FileNotFoundError, AttributeError):
        pass
    return dist.get_rank() if dist.is_initialized() else 0


def launch_distributed_job(backend: str = "nccl", sp_size: int = 1):
    if sp_size > 1:
        # Use CleanCode's distributed init which sets up SP process groups
        _ps = _get_cleancode_parallel_state()
        _ps.maybe_init_distributed_environment_and_model_parallel(
            tp_size=1, sp_size=sp_size
        )
        _sync_sp_seeds(sp_size)
    else:
        # Original init path (no SP)
        _ensure_shared_path()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])

        if ":" in host:  # IPv6
            init_method = f"tcp://[{host}]:{port}"
        else:  # IPv4
            init_method = f"tcp://{host}:{port}"
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                                init_method=init_method, timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        for n, p in fsdp_module.module.named_parameters():
            self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        for n, p in fsdp_module.module.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        for n, p in fsdp_module.module.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(dtype=p.dtype, device=p.device))

    @torch.no_grad()
    def full_state_dict(self, fsdp_module):
        live_state = {}
        for n, p in fsdp_module.module.named_parameters():
            live_state[n] = p.detach().clone()
        for n, p in fsdp_module.module.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(dtype=p.dtype, device=p.device))

        checkpoint = fsdp_state_dict(fsdp_module)
        shadow_checkpoint = {}
        for n in self.shadow:
            k = n
            if k not in checkpoint and k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            if k in checkpoint:
                shadow_checkpoint[n] = checkpoint[k]
        for n, p in fsdp_module.module.named_parameters():
            if n in live_state:
                p.data.copy_(live_state[n].to(dtype=p.dtype, device=p.device))

        return shadow_checkpoint
