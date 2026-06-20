# SPDX-License-Identifier: Apache-2.0
"""
EMA implementation for DMD distillation.

Adapted from FastVideo's EMA_FSDP, compatible with HY-WorldPlay's distributed training.

Supports two modes:
  - mode="local_shard" (default): maintain float32 CPU EMA of local parameter shards on every rank.
    Suitable for models using DTensor/SP.
  - mode="rank0_full": maintain a consolidated float32 CPU EMA of full parameters on rank 0 only.
    Useful for checkpoint export.
"""
import torch
import torch.distributed as dist

from trainer.training.training_utils import gather_state_dict_on_cpu_rank0


class EMA:
    """
    Exponential Moving Average for model weights with distributed training support.

    Args:
        model: The model to track EMA for
        decay: EMA decay rate (typically 0.99 or 0.999)
        mode: "local_shard" or "rank0_full"
            - local_shard: Each rank maintains EMA of its local shard
            - rank0_full: Only rank 0 maintains full EMA (for checkpoint export)
    """

    def __init__(self, model, decay: float = 0.999, mode: str = "local_shard"):
        self.decay = float(decay)
        self.mode = mode
        self.shadow: dict[str, torch.Tensor] = {}
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.mode not in {"local_shard", "rank0_full"}:
            raise ValueError(f"Unsupported EMA mode: {self.mode}, must be 'local_shard' or 'rank0_full'")
        self._init_shadow(model)

    @staticmethod
    def _to_local_tensor(t: torch.Tensor) -> torch.Tensor:
        """
        DTensor-aware to_local fetch; fall back to raw tensor.

        This handles the case where parameters are sharded using DTensor.
        """
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(t, DTensor):
                return t.to_local()
        except Exception:
            pass
        return t

    @torch.no_grad()
    def _init_shadow(self, model):
        """Initialize EMA shadow parameters."""
        if self.mode == "rank0_full":
            # Gather full state dict on rank 0 only
            cpu_state = gather_state_dict_on_cpu_rank0(model, device=None)
            if self.rank == 0:
                self.shadow = {k: v.detach().clone().float().cpu() for k, v in cpu_state.items()}
            else:
                self.shadow = {}
            return

        # local_shard: maintain EMA of local shards for requires_grad params
        self.shadow = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            local = self._to_local_tensor(p.detach())
            self.shadow[name] = local.clone().float().cpu()

    @torch.no_grad()
    def update(self, model):
        """Update EMA shadow parameters."""
        d = self.decay
        if self.mode == "rank0_full":
            if self.rank != 0:
                return
            # Gather full state and update on rank 0 only
            cpu_state = gather_state_dict_on_cpu_rank0(model, device=None)
            for n, v in cpu_state.items():
                v_cpu = v.detach().float().cpu()
                if n not in self.shadow:
                    self.shadow[n] = v_cpu.clone()
                else:
                    self.shadow[n].mul_(d).add_(v_cpu, alpha=1.0 - d)
            return

        # local_shard: update local shard EMA on every rank
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            local = self._to_local_tensor(p.detach())
            v_cpu = local.float().cpu()
            if name not in self.shadow:
                self.shadow[name] = v_cpu.clone()
            else:
                self.shadow[name].mul_(d).add_(v_cpu, alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return EMA state dict."""
        if self.mode == "rank0_full":
            return {k: v.clone() for k, v in self.shadow.items()} if self.rank == 0 else {}
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]):
        """Load EMA state dict."""
        self.shadow = {k: v.clone() for k, v in sd.items()}

    @torch.no_grad()
    def copy_to_unwrapped(self, model) -> None:
        """
        Copy EMA weights into a non-sharded (unwrapped) module.
        Intended for export/eval.
        For mode="rank0_full", only rank 0 has the full EMA state.
        """
        if self.mode == "rank0_full" and self.rank != 0:
            return
        name_to_param = dict(model.named_parameters())
        for n, w in self.shadow.items():
            if n in name_to_param:
                p = name_to_param[n]
                p.data.copy_(w.to(dtype=p.dtype, device=p.device))

    class _ApplyEMACtx:
        """Context manager to temporarily apply EMA weights to model."""

        def __init__(self, ema: "EMA", model):
            self.ema = ema
            self.model = model
            self.saved: dict[str, torch.Tensor] = {}

        def __enter__(self):
            if self.ema.mode != "local_shard":
                raise RuntimeError("EMA apply_to_model is only supported for mode='local_shard'")
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    # Save local shard
                    p_local = EMA._to_local_tensor(p.detach())
                    if p_local.numel() == 0:
                        # Nothing to swap on this rank for this param
                        continue
                    self.saved[name] = p_local.clone().to(device=p_local.device, dtype=p_local.dtype)
                    if name in self.ema.shadow:
                        ema_cpu = self.ema.shadow[name]
                        if ema_cpu.numel() != p_local.numel():
                            # Shard shape mismatch (e.g., empty shard here), skip
                            continue
                        # Copy EMA shard into local param shard
                        p_local.copy_(ema_cpu.to(dtype=p_local.dtype, device=p_local.device))
            return self.model

        def __exit__(self, exc_type, exc, tb):
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in self.saved:
                        p_local = EMA._to_local_tensor(p.detach())
                        if p_local.numel() == 0:
                            continue
                        saved_local = self.saved[name]
                        if saved_local.numel() != p_local.numel():
                            continue
                        p_local.copy_(saved_local)

    def apply_to_model(self, model):
        """Return context manager to temporarily apply EMA weights."""
        return self._ApplyEMACtx(self, model)
