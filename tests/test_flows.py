import torch
from models.flows import NormalizingFlow


def test_flow_roundtrip_even() -> None:
    torch.manual_seed(0)
    B, T = 2, 4
    flow = NormalizingFlow(n_layers=2, d_model=T, hidden_dim=16)
    x = torch.randn(B, T)
    z, _ = flow(x)
    x_recon = flow.inverse(z)
    assert torch.allclose(x, x_recon, atol=1e-5)


def test_flow_roundtrip_odd() -> None:
    torch.manual_seed(0)
    B, T = 2, 5
    flow = NormalizingFlow(n_layers=2, d_model=T, hidden_dim=16)
    x = torch.randn(B, T)
    z, _ = flow(x)
    x_recon = flow.inverse(z)
    assert torch.allclose(x, x_recon, atol=1e-5)
