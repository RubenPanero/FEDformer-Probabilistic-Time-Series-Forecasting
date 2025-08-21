import torch
import numpy as np
from models.flows import NormalizingFlow


def test_flow_roundtrip_even():
    torch.manual_seed(0)
    B, T = 2, 4
    flow = NormalizingFlow(n_layers=2, d_model=T, hidden_dim=8, context_dim=0)
    x = torch.randn(B, T)
    z, ldj = flow.forward(x)
    x_rec = flow.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_flow_roundtrip_odd():
    torch.manual_seed(0)
    B, T = 2, 5
    flow = NormalizingFlow(n_layers=2, d_model=T, hidden_dim=8, context_dim=0)
    x = torch.randn(B, T)
    z, ldj = flow.forward(x)
    x_rec = flow.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-5)
