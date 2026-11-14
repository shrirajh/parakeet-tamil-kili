"""Shared fixtures for the kernel tests (each test module skips itself without torch / Triton / CUDA / NeMo).

  pytest tests/kernels -v
KILI_TEST_MODEL: a .nemo path or HF id for the tests that need real Parakeet weights
(default nvidia/parakeet-tdt-0.6b-v3, downloaded once into the Hugging Face cache)."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _no_tf32():
    """fp32 comparisons with TF32 off; restored afterwards."""
    try:
        import torch
    except ImportError:
        yield
        return
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m, c


def _load():
    pytest.importorskip("nemo.collections.asr")
    from nemo.collections.asr.models import ASRModel
    src = os.environ.get("KILI_TEST_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    if src.endswith(".nemo"):
        return ASRModel.restore_from(src, map_location="cpu")
    return ASRModel.from_pretrained(src, map_location="cpu")


@pytest.fixture(scope="session")
def parakeet():
    """The real model on the CPU, shared and read-only: tests deep-copy the submodules they need."""
    return _load()


@pytest.fixture(scope="module")
def fresh_parakeet():
    """A separate copy of the model for one test module, which may modify it."""
    return _load()
