"""XPU device helpers used by the experiment entry points."""

import torch


def xpu_is_available():
    """Return whether PyTorch has an available XPU backend."""
    return bool(getattr(torch, "xpu", None) and torch.xpu.is_available())


def get_device(index=0):
    """Prefer an XPU; use CPU when no XPU is available."""
    if xpu_is_available():
        return torch.device("xpu:{}".format(index))
    return torch.device("cpu")


def empty_cache():
    """Release cached XPU memory when the XPU backend is active."""
    if xpu_is_available():
        torch.xpu.empty_cache()


def synchronize(index=None):
    """Wait for submitted XPU work before normal interpreter teardown."""

    if not xpu_is_available():
        return
    if index is None:
        torch.xpu.synchronize()
    else:
        torch.xpu.synchronize(index)
