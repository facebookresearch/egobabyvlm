# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed training helpers."""

import logging
import os
import random
import socket
from contextlib import closing
from dataclasses import dataclass

import submitit
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Bind to port 0 to let the OS pick a free port, then close and return it."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def is_dist_avail_and_initialized() -> bool:
    """Check if torch.distributed is available and initialized.

    Returns:
        ``True`` if a process group is initialized.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the world size.

    Returns:
        Process count, or ``1`` if distributed is not initialized.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the global process rank.

    Returns:
        Global process rank, or ``0`` if distributed is not initialized.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if the calling process is the main process.

    Returns:
        ``True`` on rank 0 (or when not distributed).
    """
    return get_rank() == 0


@dataclass(frozen=True)
class DistributedEnvironment:
    """Resolved rendezvous parameters for the current distributed launch."""

    #: Global process rank across all nodes.
    global_rank: int

    #: Local process rank within the current node.
    local_rank: int

    #: Total number of processes participating in the job.
    world_size: int

    #: Master node port for the rendezvous.
    master_port: int

    #: Master node address for the rendezvous.
    master_addr: str


def distributed_environment(*, min_master_port: int = 20_000, max_master_port: int = 60_000) -> DistributedEnvironment:
    """Detect the current launch type (torchrun, Slurm, single-GPU) and export rendezvous env vars.

    Side effects: sets ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``, ``MASTER_PORT``, ``MASTER_ADDR``
    when the current launch type didn't already populate them (i.e., Slurm and single-GPU paths).

    Args:
        min_master_port: Lower bound for the seeded random port pick.
        max_master_port: Upper bound for the seeded random port pick.

    Returns:
        Resolved :class:`DistributedEnvironment` describing the current process.
    """
    if os.getenv("LOCAL_RANK") is not None:
        return DistributedEnvironment(
            global_rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            master_port=int(os.environ["MASTER_PORT"]),
            master_addr=os.environ["MASTER_ADDR"],
        )
    slurm: submitit.JobEnvironment | None = None
    if os.getenv("SLURM_JOB_ID") is not None:
        try:
            slurm = submitit.JobEnvironment()
        except RuntimeError:
            # Running under `srun --overlap` or similar: SLURM_JOB_ID is set but
            # the standard task vars (SLURM_PROCID, ...) aren't, so submitit can't
            # detect the env. Fall through to single-process defaults.
            slurm = None
    if slurm is not None:
        # Multi-task SLURM jobs need rank 0 and the other ranks to agree on a
        # port without coordination, so use the deterministic seed; single-task
        # jobs can just grab any free port.
        if slurm.num_tasks == 1:
            master_port = _find_free_port()
        else:
            port_seed_parts = [os.environ["SLURM_JOB_ID"]]
            if os.getenv("SLURM_ARRAY_JOB_ID") is not None:
                port_seed_parts.append(os.environ["SLURM_ARRAY_JOB_ID"])
            if os.getenv("SLURM_ARRAY_TASK_ID") is not None:
                port_seed_parts.append(os.environ["SLURM_ARRAY_TASK_ID"])
            master_port = random.Random(":".join(port_seed_parts)).randint(min_master_port, max_master_port)
        env = DistributedEnvironment(
            global_rank=slurm.global_rank,
            local_rank=slurm.local_rank,
            world_size=slurm.num_tasks,
            master_port=master_port,
            master_addr=slurm.hostnames[0],
        )
    else:
        env = DistributedEnvironment(
            global_rank=0,
            local_rank=0,
            world_size=1,
            master_port=_find_free_port(),
            master_addr="127.0.0.1",
        )
    os.environ["RANK"] = str(env.global_rank)
    os.environ["LOCAL_RANK"] = str(env.local_rank)
    os.environ["WORLD_SIZE"] = str(env.world_size)
    os.environ["MASTER_PORT"] = str(env.master_port)
    os.environ["MASTER_ADDR"] = env.master_addr
    return env


def setup_distributed() -> None:
    """Initialize NCCL process group and pin the current CUDA device.

    No-ops gracefully when ``world_size == 1`` or when CUDA is unavailable.
    """
    env = distributed_environment()
    logger.debug("World size %s, local rank %s, global rank %s", env.world_size, env.local_rank, env.global_rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(env.local_rank)

    if env.world_size == 1:
        return

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=env.world_size,
        rank=env.global_rank,
        device_id=env.local_rank,
    )
    dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Recursively unwrap DDP / Lightning / similar containers."""
    inner = getattr(model, "module", None)
    if isinstance(inner, torch.nn.Module):
        return unwrap_model(inner)
    return model


def all_reduce_mean(t: torch.Tensor | float, device: torch.device | str = "cuda") -> float:
    """All-reduce a scalar across ranks and divide by world size; no-op when single-rank."""
    world_size = get_world_size()
    if world_size <= 1:
        return t if isinstance(t, float) else t.item()
    t_reduce = torch.tensor(t, device=device)
    dist.all_reduce(t_reduce)
    t_reduce /= world_size
    return t_reduce.item()


def all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor across ranks and concatenate; no-op when single-rank."""
    if not is_dist_avail_and_initialized():
        return t
    t_list = [torch.zeros_like(t) for _ in range(get_world_size())]
    dist.all_gather(tensor_list=t_list, tensor=t.contiguous())
    t_list[get_rank()] = t
    return torch.cat(t_list, 0)
