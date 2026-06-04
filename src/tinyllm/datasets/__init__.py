"""Datasets module."""

from src.tinyllm.datasets.fineweb_helper import (
    DataLoaderConfig,
    NanoGPTDataset,
    create_nanogpt_dataloader,
)


def DatasetHelper(*args, **kwargs):
    """Lazy import — only loads if HF datasets are actually used."""
    from src.tinyllm.datasets.dataset_helper import DatasetHelper as _DH

    return _DH(*args, **kwargs)


def BatchSamplerSimilarLength(*args, **kwargs):
    from src.tinyllm.datasets.dataset_helper import (
        BatchSamplerSimilarLength as _BS,
    )

    return _BS(*args, **kwargs)


__all__ = [
    "DataLoaderConfig",
    "NanoGPTDataset",
    "create_nanogpt_dataloader",
    "DatasetHelper",
    "BatchSamplerSimilarLength",
]
