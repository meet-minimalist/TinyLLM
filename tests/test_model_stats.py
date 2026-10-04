from src.tinyllm.factory.factory import model_factory
from src.tinyllm.utils.model_stats import count_params, flops_per_token

from conftest import tiny_model_config


def test_counts_tied_embedding_once():
    cfg = tiny_model_config()
    model = model_factory(cfg)
    stats = count_params(model)
    assert stats["embedding"] == cfg.vocab_size * cfg.d_model
    assert stats["total"] == sum(p.numel() for p in model.parameters())
    assert stats["active"] == stats["total"]  # dense: nothing inactive


def test_flops_include_output_layer():
    cfg = tiny_model_config()
    n = 1000
    f = flops_per_token(cfg, n, seq_len=64)
    L, d, V = cfg.blocks.count, cfg.d_model, cfg.vocab_size
    assert f == 6 * n + 6 * d * V + 6 * L * d * 64
