"""Tests pour le module training."""

import torch

from src.training.utils import get_device


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "mps", "cuda")
