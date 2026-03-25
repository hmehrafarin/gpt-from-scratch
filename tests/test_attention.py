import torch
import pytest
from config import AttentionConfig
from model.attention import CausalSelfAttention


@pytest.fixture
def cfg():
    return AttentionConfig(d_model=1024, n_heads=16, max_seq_len=128)


def test_output_shape(cfg):
    model = CausalSelfAttention(cfg)
    x = torch.randn(2, 8, 1024)  # B=2, T=8, C=1024
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


def test_causal_property(cfg):
    """
    We'd like to test if a change in one middle token will
    affect the future tokens or not, if causal attention is
    implemented correctly it shouldn't
    """
    model = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 8, 1024)
    x_2 = x.clone()
    x_2[:, 4, :] += 7

    with torch.no_grad():
        out = model(x)
        out_2 = model(x_2)

    # Positions 0-3 must be identical, positions 4-7 can differ
    assert torch.allclose(out[0, :4], out_2[0, :4], atol=1e-5)


def test_mask_is_buffer(cfg):
    model = CausalSelfAttention(cfg)
    assert "mask" in dict(model.named_buffers())
