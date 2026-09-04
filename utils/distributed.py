"""Small, fail-closed distributed runtime helpers for Aurora and local tests."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from utils.device import xpu_is_available


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_size: int = 1
    backend: str = ""

    @property
    def is_main(self):
        return self.rank == 0


def launcher_topology():
    """Return launcher-owned ``(rank, world, local_rank, local_size)``.

    Importing mpi4py outside an MPI launch can abort the interpreter, so it is
    gated on PALS variables.  Under Aurora, MPI is the authority for global
    topology and PALS_LOCAL_RANKID is the authority for device placement.
    """
    if "PALS_LOCAL_RANKID" in os.environ:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = int(comm.Get_rank())
        world = int(comm.Get_size())
        local_rank = int(os.environ["PALS_LOCAL_RANKID"])
        local_size = int(os.environ.get("PALS_LOCAL_SIZE", world))
        hosts = comm.allgather(socket.gethostname().split(".")[0])
        if len(set(hosts)) != 1:
            raise RuntimeError(
                "FITS distributed training currently supports one compute node; "
                "the launcher placed ranks on {} nodes".format(len(set(hosts)))
            )
        return rank, world, local_rank, local_size, comm

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank if world > 1 else 0))
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    return rank, world, local_rank, local_size, None


def initialize_distributed(enabled, use_accelerator=True, timeout_seconds=120):
    """Initialize DDP from the active launcher and bind one rank per device."""
    rank, world, local_rank, local_size, mpi_comm = launcher_topology()

    if world > 1 and not enabled:
        raise RuntimeError(
            "A multi-rank launcher was detected, but --distributed was not set. "
            "Refusing to start duplicate independent trainings that would race on "
            "the same checkpoint."
        )
    if enabled and world < 2:
        raise RuntimeError(
            "--distributed requires a launcher with at least two ranks; use "
            "mpiexec --pmi=pmix -n N --ppn N ..."
        )
    if not enabled:
        return DistributedContext()

    if not 0 <= rank < world:
        raise RuntimeError("rank {} is outside world size {}".format(rank, world))
    if not 0 <= local_rank < local_size:
        raise RuntimeError(
            "local rank {} is outside local size {}".format(local_rank, local_size)
        )
    if local_size != world:
        raise RuntimeError(
            "FITS distributed training is single-node only, but local size {} "
            "does not match world size {}".format(local_size, world)
        )

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(local_rank)

    use_xpu = bool(use_accelerator and xpu_is_available())
    if "PALS_LOCAL_RANKID" in os.environ and not use_xpu:
        raise RuntimeError(
            "Aurora distributed mode requires an available XPU; refusing to "
            "silently run the multi-tile job on CPUs"
        )
    if use_xpu:
        if hasattr(dist, "is_xccl_available") and not dist.is_xccl_available():
            raise RuntimeError(
                "the selected PyTorch build has XPU support but no XCCL backend"
            )
        hierarchy = os.environ.get("ZE_FLAT_DEVICE_HIERARCHY", "")
        if hierarchy != "FLAT":
            raise RuntimeError(
                "one-rank-per-tile DDP requires ZE_FLAT_DEVICE_HIERARCHY=FLAT; "
                "got {!r}".format(hierarchy or "<unset>")
            )
        if os.environ.get("ZE_AFFINITY_MASK"):
            raise RuntimeError(
                "one-rank-per-tile DDP expects all tiles to be visible; unset the "
                "job-wide ZE_AFFINITY_MASK"
            )
        visible = int(torch.xpu.device_count())
        if local_size > visible or local_rank >= visible:
            raise RuntimeError(
                "launcher assigned local rank {} of {}, but only {} XPU devices "
                "are visible".format(local_rank, local_size, visible)
            )
        torch.xpu.set_device(local_rank)
        backend = "xccl"
    else:
        backend = "gloo"

    if mpi_comm is not None:
        configured_master = os.environ.get("MASTER_ADDR") if rank == 0 else None
        if rank == 0:
            master_addr = configured_master or (
                socket.gethostname().split(".")[0]
                + ".hsn.cm.aurora.alcf.anl.gov"
            )
        else:
            master_addr = None
        master_addr = mpi_comm.bcast(master_addr, root=0)
    else:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_ADDR"] = str(master_addr)

    if "MASTER_PORT" not in os.environ:
        job_id = os.environ.get("PBS_JOBID", "0").split(".")[0]
        try:
            port = 20000 + int(job_id) % 20000
        except ValueError:
            port = 29500
        os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=int(timeout_seconds)),
    )
    if dist.get_rank() != rank or dist.get_world_size() != world:
        raise RuntimeError("the initialized process group disagrees with the launcher")

    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world,
        local_rank=local_rank,
        local_size=local_size,
        backend=backend,
    )


def cleanup_distributed(context):
    if not context.enabled:
        return
    # Drain queued device work before process teardown. Hard-killing processes
    # with live Level Zero contexts has caused Aurora ze_peak prologue failures.
    try:
        if context.backend == "xccl" and xpu_is_available():
            torch.xpu.synchronize(context.local_rank)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def barrier(context):
    if context.enabled:
        dist.barrier()


def unwrap_model(model):
    return getattr(model, "module", model)


def reduce_sum_count(total, count, device, context):
    """Return globally summed floating-point numerator and integer denominator."""
    stats = torch.tensor(
        [float(total), float(count)], dtype=torch.float64, device=device
    )
    if context.enabled:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item())


def broadcast_bool(value, device, context, src=0):
    if not context.enabled:
        return bool(value)
    payload = torch.tensor(
        [1 if value else 0], dtype=torch.int32, device=device
    )
    dist.broadcast(payload, src=src)
    return bool(payload.item())
