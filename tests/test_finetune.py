from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_dynamic_fold_checkpoint(tmp_path: Path) -> None:
    """El checkpoint del último fold debe ser dinámico según n_splits."""
    import numpy as np
    from training.forecast_output import ForecastOutput
    from training.sequential_finetuner import finetune_sequence

    for n_splits in [2, 3, 5]:
        expected_fold_idx = n_splits - 1
        expected_filename = f"best_model_fold_{expected_fold_idx}.pt"

        # Creamos el archivo de checkpoint esperado y uno base ficticio
        base_ckpt = tmp_path / "base.pt"
        base_ckpt.touch()
        ckpt_dir = tmp_path / f"out_{n_splits}" / "MOCK"
        ckpt_dir.mkdir(parents=True)
        expected_ckpt = ckpt_dir / expected_filename
        expected_ckpt.touch()

        mock_cfg = MagicMock()
        with (
            patch(
                "training.sequential_finetuner.FEDformerConfig", return_value=mock_cfg
            ),
            patch("training.sequential_finetuner.TimeSeriesDataset"),
            patch("training.sequential_finetuner.WalkForwardTrainer") as MockTrainer,
            patch("os.path.exists", return_value=True),
        ):
            mock_trainer = MagicMock()
            mock_trainer.checkpoint_dir = ckpt_dir
            mock_trainer.run_backtest.return_value = ForecastOutput(
                preds_scaled=np.zeros((1, 1, 1)),
                gt_scaled=np.zeros((1, 1, 1)),
                samples_scaled=np.zeros((1, 1, 1, 1)),
                preds_real=np.zeros((1, 1, 1)),
                gt_real=np.zeros((1, 1, 1)),
                samples_real=np.zeros((1, 1, 1, 1)),
                metric_space="returns",
                return_transform="none",
                target_names=["Close"],
            )
            MockTrainer.return_value = mock_trainer

            result_ckpt = finetune_sequence(
                symbols=["MOCK"],
                base_checkpoint=str(base_ckpt),
                output_dir=str(tmp_path / f"out_{n_splits}"),
                n_splits=n_splits,
            )

        assert result_ckpt is not None
        assert expected_filename in result_ckpt, (
            f"n_splits={n_splits}: esperado '{expected_filename}', obtenido '{result_ckpt}'"
        )
