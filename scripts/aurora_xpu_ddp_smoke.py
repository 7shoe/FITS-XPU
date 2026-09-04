"""Collective smoke test for one-rank-per-tile FITS training on Aurora."""

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from data_provider.data_factory import ExactDistributedSampler
from models import FITS
from utils.distributed import (barrier, cleanup_distributed,
                               initialize_distributed, unwrap_model)
from utils.tools import EarlyStopping


def main():
    context = initialize_distributed(True, use_accelerator=True)
    device = torch.device('xpu:{}'.format(context.local_rank))
    try:
        properties = torch.xpu.get_device_properties(device)
        print(
            'rank={} local_rank={} device={} uuid={}'.format(
                context.rank,
                context.local_rank,
                properties.name,
                properties.uuid,
            ),
            flush=True,
        )

        torch.manual_seed(2021)
        config = SimpleNamespace(
            seq_len=96,
            pred_len=24,
            individual=False,
            enc_in=7,
            cut_freq=8,
        )
        model = FITS.Model(config).to(device)
        model = DistributedDataParallel(model)
        torch.manual_seed(2021 + context.rank)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(3):
            source = torch.randn(4, 96, 7, device=device) + context.rank
            target = torch.randn(4, 120, 7, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(source)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(
                    torch.view_as_real(parameter.grad)
                ).all():
                    raise RuntimeError('invalid complex gradient for {}'.format(name))
            optimizer.step()

        flat = torch.cat([
            torch.view_as_real(parameter.detach()).reshape(-1)
            for parameter in model.parameters()
        ])
        replicas = [torch.empty_like(flat) for _ in range(context.world_size)]
        dist.all_gather(replicas, flat)
        if any(not torch.equal(replicas[0], replica) for replica in replicas[1:]):
            raise RuntimeError('DDP replicas diverged after Adam steps')

        dataset = list(range(53))
        sampler = ExactDistributedSampler(
            dataset, context.world_size, context.rank
        )
        coverage = torch.zeros(len(dataset), dtype=torch.int32, device=device)
        coverage[list(sampler)] = 1
        dist.all_reduce(coverage, op=dist.ReduceOp.SUM)
        if not torch.equal(coverage, torch.ones_like(coverage)):
            raise RuntimeError('exact sampler omitted or duplicated validation rows')

        checkpoint_root = Path(
            os.environ.get('FITS_DDP_SMOKE_OUT', 'artifacts/aurora_ddp_smoke')
        ) / os.environ.get('PBS_JOBID', 'local').split('.')[0]
        if context.is_main:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            EarlyStopping(patience=1)(
                float(loss.item()), unwrap_model(model), str(checkpoint_root)
            )
        barrier(context)
        checkpoint = checkpoint_root / 'checkpoint.pth'
        state = torch.load(
            checkpoint, map_location=device, weights_only=True
        )
        if any(key.startswith('module.') for key in state):
            raise RuntimeError('checkpoint contains DDP module.* prefixes')
        bare_model = FITS.Model(config).to(device)
        bare_model.load_state_dict(state)
        barrier(context)

        if context.is_main:
            print(
                'AURORA DDP COLLECTIVE SMOKE: PASS '
                '(world={}, complex_gradients=yes, exact_validation=yes, '
                'checkpoint_reload=yes)'.format(context.world_size),
                flush=True,
            )
    finally:
        cleanup_distributed(context)


if __name__ == '__main__':
    main()
