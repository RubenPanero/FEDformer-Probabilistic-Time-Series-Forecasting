import os
import tempfile
import torch
import numpy as np
import pytest


def test_forward_smoke(model_factory, synthetic_batch):
    x_enc, x_dec, x_regime, y = synthetic_batch(batch_size=2)
    model = model_factory(train=False)
    dist = model(x_enc, x_dec, x_regime)
    # mean should be tensor of shape (B, pred_len, c_out)
    m = dist.mean
    assert isinstance(m, torch.Tensor)
    assert m.shape == (2, model.config.pred_len, model.config.c_out)
    # log_prob returns per-batch scores
    lp = dist.log_prob(y)
    assert lp.shape == (2,)
    assert torch.isfinite(lp).all()


def test_determinism(model_factory, synthetic_batch):
    x_enc, x_dec, x_regime, y = synthetic_batch(batch_size=3)
    m1 = model_factory(train=False)
    m2 = model_factory(train=False)
    # copy state
    m2.load_state_dict(m1.state_dict())
    out1 = m1(x_enc, x_dec, x_regime).mean.detach().cpu().numpy()
    out2 = m2(x_enc, x_dec, x_regime).mean.detach().cpu().numpy()
    assert np.allclose(out1, out2, atol=1e-6)


@pytest.mark.slow
def test_backward_and_grads(model_factory, synthetic_batch):
    x_enc, x_dec, x_regime, y = synthetic_batch(batch_size=2)
    model = model_factory(train=True)
    # ensure gradients flow to a parameter
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    dist = model(x_enc, x_dec, x_regime)
    lp = dist.log_prob(y)
    loss = -lp.mean()
    optim.zero_grad()
    loss.backward()
    # check at least one param has grad
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_device_transfer(model_factory, synthetic_batch):
    x_enc, x_dec, x_regime, y = synthetic_batch(batch_size=2)
    model = model_factory(train=False)
    device = torch.device('cuda')
    model = model.to(device)
    x_enc = x_enc.to(device)
    x_dec = x_dec.to(device)
    x_regime = x_regime.to(device)
    dist = model(x_enc, x_dec, x_regime)
    assert dist.mean.device == device


@pytest.mark.slow
def test_state_dict_roundtrip(tmp_path, model_factory, synthetic_batch):
    x_enc, x_dec, x_regime, y = synthetic_batch(batch_size=2)
    model = model_factory(train=False)
    # forward before save
    out1 = model(x_enc, x_dec, x_regime).mean.detach()
    p = tmp_path / "m.pt"
    torch.save(model.state_dict(), str(p))
    m2 = model_factory(train=False)
    m2.load_state_dict(torch.load(str(p), map_location='cpu'))
    out2 = m2(x_enc, x_dec, x_regime).mean.detach()
    assert torch.allclose(out1, out2, atol=1e-6)


def test_amp_optional(model_factory, synthetic_batch):
    # Run a tiny AMP-enabled forward/backward if available
    x_enc, x_dec, x_regime, y = synthetic_batch(batch_size=2)
    model = model_factory(train=True)
    scaler = None
    if hasattr(torch.cuda.amp, 'GradScaler'):
        scaler = torch.cuda.amp.GradScaler()
    dist = None
    if scaler is not None:
        with torch.cuda.amp.autocast(enabled=False):
            dist = model(x_enc, x_dec, x_regime)
            loss = -dist.log_prob(y).mean()
        scaler.scale(loss).backward()
        # if no error, pass
    else:
        # If AMP not present, just run a normal backward
        dist = model(x_enc, x_dec, x_regime)
        loss = -dist.log_prob(y).mean()
        loss.backward()


def test_decoder_input_path_shape(model_factory, config):
    # Ensure _prepare_decoder_input computes seasonal/trend paths with correct shapes
    model = model_factory(train=False)
    B = 2
    dec_in = config.dec_in
    label_len = config.label_len
    pred_len = config.pred_len
    x_dec = torch.randn(B, label_len + pred_len, dec_in)
    seasonal_out, trend_out = model._prepare_decoder_input(x_dec)
    expected_shape_seasonal = (B, label_len + pred_len, dec_in)
    expected_shape_trend = (B, label_len + pred_len, config.d_model)
    assert seasonal_out.shape == expected_shape_seasonal
    assert trend_out.shape == expected_shape_trend
