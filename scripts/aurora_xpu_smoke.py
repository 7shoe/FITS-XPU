"""One-tile Aurora smoke test for FITS XPU placement and core kernels."""

import importlib.util
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from exp.exp_basic import Exp_Basic
from layers.AutoCorrelation import AutoCorrelation
from models import FITS, Real_FITS
from utils.augmentations import BatchAugmentation
from utils.device import empty_cache, get_device, xpu_is_available


def check_environment():
    print("torch:", torch.__version__, flush=True)
    print("ZE_FLAT_DEVICE_HIERARCHY:", os.environ.get("ZE_FLAT_DEVICE_HIERARCHY"), flush=True)
    print("ZE_AFFINITY_MASK:", os.environ.get("ZE_AFFINITY_MASK"), flush=True)
    print("THOP installed:", importlib.util.find_spec("thop") is not None, flush=True)
    print("XPU available:", xpu_is_available(), flush=True)
    print("visible XPU devices:", torch.xpu.device_count(), flush=True)
    if not xpu_is_available():
        raise RuntimeError("Aurora smoke test requires an available XPU")
    if torch.xpu.device_count() != 1:
        raise RuntimeError("ZE_AFFINITY_MASK must expose exactly one XPU tile")

    device = get_device(0)
    torch.xpu.set_device(device)
    print("selected device:", device, flush=True)
    print("device properties:", torch.xpu.get_device_properties(device), flush=True)
    return device


def check_experiment_selection(device):
    experiment = object.__new__(Exp_Basic)
    experiment.args = SimpleNamespace(use_gpu=True, gpu=0)
    selected = experiment._acquire_device()
    if selected != device:
        raise RuntimeError(f"experiment selected {selected}, expected {device}")


def train_step(model_type, device):
    config = SimpleNamespace(
        seq_len=96,
        pred_len=24,
        individual=False,
        enc_in=7,
        cut_freq=8,
    )
    model = model_type.Model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = [parameter.detach().clone() for parameter in model.parameters()]
    for _ in range(2):
        source = torch.randn(4, config.seq_len, config.enc_in, device=device)
        target = torch.randn(4, config.seq_len + config.pred_len, config.enc_in, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(source)
        if prediction.device != device:
            raise RuntimeError(f"prediction landed on {prediction.device}, expected {device}")
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {model_type.__name__} loss")
        loss.backward()
        optimizer.step()

    torch.xpu.synchronize(device)
    changed = any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    if not changed:
        raise RuntimeError(f"{model_type.__name__} optimizer did not update parameters")
    print(f"{model_type.__name__}: loss={loss.item():.6f} PASS", flush=True)


def check_augmentations(device):
    x = torch.randn(4, 96, 7, device=device)
    y = torch.randn(4, 24, 7, device=device)
    augment = BatchAugmentation()
    outputs = {
        "freq_mask": augment.freq_mask(x, y),
        "freq_mix": augment.freq_mix(x, y),
        "noise": augment.noise(x, y),
        "noise_input": augment.noise_input(x, y),
    }
    for name, output in outputs.items():
        if output.device != device or not torch.isfinite(output).all():
            raise RuntimeError(f"{name} produced an invalid result on {output.device}")
    print("augmentations: PASS", flush=True)


def check_autocorrelation(device):
    values = torch.randn(2, 2, 3, 16, device=device)
    corr = torch.randn(2, 2, 3, 16, device=device)
    layer = AutoCorrelation(factor=1)
    inference = layer.time_delay_agg_inference(values, corr)
    full = layer.time_delay_agg_full(values, corr)
    if inference.device != device or full.device != device:
        raise RuntimeError("AutoCorrelation created tensors on the wrong device")
    print("autocorrelation: PASS", flush=True)


def main():
    device = check_environment()
    check_experiment_selection(device)

    # This import used to require THOP even though profiling was not requested.
    from exp.exp_main_F import Exp_Main  # noqa: F401

    print("forecasting import without mandatory THOP: PASS", flush=True)
    train_step(FITS, device)
    train_step(Real_FITS, device)
    check_augmentations(device)
    check_autocorrelation(device)
    empty_cache()
    print("AURORA XPU SMOKE: PASS", flush=True)


if __name__ == "__main__":
    main()
