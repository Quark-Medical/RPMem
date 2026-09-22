"""Distributed helpers for replicated model-parallel backbones."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist


def average_parameter_gradients(
    parameters: Iterable[torch.nn.Parameter],
    *,
    bucket_cap_bytes: int = 128 * 1024 * 1024,
) -> dict[str, int]:
    """Average dense gradients across the initialized process group.

    This is used when each process owns a complete trainable hypernetwork but
    the frozen language-model backbone is sharded over all GPUs local to that
    process. Wrapping the complete model in DDP is not valid in that topology,
    so only the trainable gradients are synchronized.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed gradient averaging requires an initialized process group")
    world_size = dist.get_world_size()
    if world_size <= 1:
        return {"world_size": world_size, "gradient_tensors": 0, "buckets": 0}
    if bucket_cap_bytes <= 0:
        raise ValueError("bucket_cap_bytes must be positive")

    parameter_list = list(parameters)
    if not parameter_list:
        return {"world_size": world_size, "gradient_tensors": 0, "buckets": 0}
    presence_device = parameter_list[0].device
    gradient_presence = torch.tensor(
        [parameter.grad is not None for parameter in parameter_list],
        dtype=torch.int32,
        device=presence_device,
    )
    dist.all_reduce(gradient_presence, op=dist.ReduceOp.SUM)
    inconsistent = (gradient_presence != 0) & (gradient_presence != world_size)
    if inconsistent.any():
        indices = inconsistent.nonzero().flatten().cpu().tolist()
        raise RuntimeError(
            "trainable gradient presence differs across ranks for parameter "
            f"indices {indices[:16]}"
        )

    grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for parameter in parameter_list:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            raise TypeError("sparse gradients are not supported")
        grouped.setdefault((gradient.device, gradient.dtype), []).append(gradient)

    bucket_count = 0
    gradient_count = 0
    for gradients in grouped.values():
        bucket: list[torch.Tensor] = []
        bucket_bytes = 0

        def flush() -> None:
            nonlocal bucket, bucket_bytes, bucket_count, gradient_count
            if not bucket:
                return
            flat = torch.cat([gradient.reshape(-1) for gradient in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
            offset = 0
            for gradient in bucket:
                size = gradient.numel()
                gradient.copy_(flat[offset : offset + size].view_as(gradient))
                offset += size
            gradient_count += len(bucket)
            bucket_count += 1
            bucket = []
            bucket_bytes = 0

        for gradient in gradients:
            size_bytes = gradient.numel() * gradient.element_size()
            if bucket and bucket_bytes + size_bytes > bucket_cap_bytes:
                flush()
            bucket.append(gradient)
            bucket_bytes += size_bytes
            if bucket_bytes >= bucket_cap_bytes:
                flush()
        flush()

    return {
        "world_size": world_size,
        "gradient_tensors": gradient_count,
        "buckets": bucket_count,
    }
