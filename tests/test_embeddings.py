import torch
import pytest
from config import GPTConfig
from model.embeddings import Embeddings


@pytest.fixture
def cfg():
    return GPTConfig(d_model=1024, max_seq_len=128, vocab_size=30000)


@pytest.fixture
def model(cfg):
    m = Embeddings(cfg)
    m.eval()
    return m


def test_embedding_output_shape(cfg, model):
    # defin input
    x = torch.randint(size=(2, 8), low=0, high=cfg.vocab_size)
    with torch.no_grad():
        out = model(x)
    # check if the output shape returned by the embeddings is correct
    assert out.shape == (x.shape[0], x.shape[1], cfg.d_model)


def test_position_difference(model):
    # a test to check if the same tokens at different positions get different values
    x = torch.Tensor([[7, 7]]).long()  # (batch=1, seq_length=2)
    with torch.no_grad():
        out = model(x)
    # check if the same token at different positions have different values
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_token_difference(model):
    # a test to check if tokens at the same positions get different values
    x = torch.Tensor([[5]]).long()
    x2 = torch.Tensor([[87]]).long()
    with torch.no_grad():
        out = model(x)
        out2 = model(x2)

    assert not torch.allclose(out[0, 0], out2[0, 0])


def test_output_has_no_nan(cfg, model):
    # test to check if the value returned by the embedding contains no NaNs
    x = torch.randint(size=(32, 64), low=0, high=cfg.vocab_size)
    with torch.no_grad():
        out = model(x)

    assert not torch.isnan(out).any()
