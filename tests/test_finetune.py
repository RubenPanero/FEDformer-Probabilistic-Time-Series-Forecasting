from pathlib import Path

import torch
from torch.utils.data import Subset

from config import FEDformerConfig
from data import TimeSeriesDataset
from models.fedformer import Flow_FEDformer
from training.trainer import WalkForwardTrainer


def _base_config(**kwargs: object) -> FEDformerConfig:
    return FEDformerConfig(
        target_features=["Close"],
        file_path="data/nvidia_stock_2024-08-20_to_2025-08-20.csv",
        **kwargs,
    )


def test_config_accepts_finetune_runtime_fields() -> None:
    cfg = _base_config(
        finetune_from="checkpoints/mock.pt",
        freeze_backbone=True,
        finetune_lr=1e-5,
    )
    assert cfg.finetune_from == "checkpoints/mock.pt"
    assert cfg.freeze_backbone is True
    assert cfg.finetune_lr == 1e-5


def test_trainer_finetune_checkpoint_and_freeze(tmp_path: Path) -> None:
    source_cfg = _base_config()
    _ = TimeSeriesDataset(source_cfg, flag="all")
    source_model = Flow_FEDformer(source_cfg)
    state = source_model.state_dict()
    state["components.regime_embedding.weight"] = torch.full_like(
        state["components.regime_embedding.weight"], 0.1234
    )
    ckpt_path = tmp_path / "ft.pt"
    torch.save({"model_state_dict": state}, ckpt_path)

    cfg = _base_config(
        finetune_from=str(ckpt_path),
        freeze_backbone=True,
        finetune_lr=1e-5,
        batch_size=8,
    )
    ds = TimeSeriesDataset(cfg, flag="all")
    trainer = WalkForwardTrainer(cfg, ds)
    train_subset = Subset(ds, list(range(min(32, len(ds)))))
    test_subset = Subset(ds, list(range(min(32, len(ds)))))
    components = trainer._build_training_components(
        train_subset, test_subset, fold_idx=1
    )

    loaded_value = components.model.state_dict()["components.regime_embedding.weight"][
        0, 0
    ]
    assert torch.isclose(loaded_value, torch.tensor(0.1234), atol=1e-6)

    trainable = {
        name: p.requires_grad for name, p in components.model.named_parameters()
    }
    assert trainable["flows.0.layers.0.conditioner.0.weight"] is True
    assert trainable["sequence_layers.encoders.0.layers.conv.0.weight"] is False
