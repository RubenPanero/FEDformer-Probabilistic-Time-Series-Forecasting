# -*- coding: utf-8 -*-
"""Tests para los scripts de publicación y descarga de especialistas."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from scripts import fetch_specialists as fetch
from scripts import prepare_specialist_release as release


def test_create_deterministic_zip_includes_relative_files(tmp_path: Path) -> None:
    """El ZIP de preprocessing conserva rutas relativas estables."""
    source_dir = tmp_path / "nvda_preprocessing"
    nested = source_dir / "subdir"
    nested.mkdir(parents=True)
    (source_dir / "metadata.json").write_text('{"ticker":"NVDA"}', encoding="utf-8")
    (nested / "schema.json").write_text("{}", encoding="utf-8")

    zip_path = tmp_path / "nvda_preprocessing.zip"
    required_files = release.create_deterministic_zip(source_dir, zip_path)

    assert required_files == ["metadata.json", "subdir/schema.json"]
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["metadata.json", "subdir/schema.json"]


def test_fetch_ticker_downloads_and_installs_assets(tmp_path: Path) -> None:
    """fetch_ticker descarga, valida e instala checkpoint y preprocessing."""
    project_dir = fetch.PROJECT_DIR
    registry = {
        "specialists": {
            "NVDA": {
                "checkpoint": "checkpoints/nvda_canonical.pt",
                "data": {
                    "preprocessing_artifacts": "checkpoints/nvda_preprocessing",
                },
            }
        }
    }

    checkpoint_source = tmp_path / "nvda_canonical.pt"
    checkpoint_source.write_bytes(b"checkpoint-bytes")

    preprocessing_source = tmp_path / "nvda_preprocessing"
    preprocessing_source.mkdir()
    (preprocessing_source / "metadata.json").write_text(
        '{"ticker":"NVDA"}', encoding="utf-8"
    )
    (preprocessing_source / "schema.json").write_text("{}", encoding="utf-8")

    zip_path = tmp_path / "nvda_preprocessing.zip"
    required_files = release.create_deterministic_zip(preprocessing_source, zip_path)

    checkpoint_target = project_dir / "checkpoints" / "nvda_canonical.pt"
    preprocessing_target = project_dir / "checkpoints" / "nvda_preprocessing"

    checkpoint_backup = (
        checkpoint_target.read_bytes() if checkpoint_target.exists() else None
    )
    preprocessing_backup_dir = None
    if preprocessing_target.exists():
        preprocessing_backup_dir = tmp_path / "backup_preprocessing"
        preprocessing_backup_dir.mkdir()
        for item in preprocessing_target.rglob("*"):
            target = preprocessing_backup_dir / item.relative_to(preprocessing_target)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(item.read_bytes())

    artifact_info = {
        "checkpoint": {
            "asset_name": checkpoint_source.name,
            "url": checkpoint_source.resolve().as_uri(),
            "sha256": release.sha256_file(checkpoint_source),
            "size_bytes": checkpoint_source.stat().st_size,
            "install_path": "checkpoints/nvda_canonical.pt",
        },
        "preprocessing": {
            "asset_name": zip_path.name,
            "url": zip_path.resolve().as_uri(),
            "sha256": release.sha256_file(zip_path),
            "size_bytes": zip_path.stat().st_size,
            "install_dir": "checkpoints/nvda_preprocessing",
            "required_files": required_files,
            "format": "zip",
        },
    }

    try:
        fetch.fetch_ticker("NVDA", artifact_info, registry, force=True)
        assert checkpoint_target.read_bytes() == b"checkpoint-bytes"
        assert (preprocessing_target / "metadata.json").exists()
        assert (preprocessing_target / "schema.json").exists()
    finally:
        if checkpoint_backup is None:
            checkpoint_target.unlink(missing_ok=True)
        else:
            checkpoint_target.write_bytes(checkpoint_backup)

        if preprocessing_target.exists():
            for item in sorted(preprocessing_target.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink()
                else:
                    item.rmdir()
            preprocessing_target.rmdir()

        if preprocessing_backup_dir is not None:
            preprocessing_target.mkdir(parents=True, exist_ok=True)
            for item in preprocessing_backup_dir.rglob("*"):
                target = preprocessing_target / item.relative_to(
                    preprocessing_backup_dir
                )
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(item.read_bytes())


def test_build_manifest_contains_release_urls(tmp_path: Path) -> None:
    """El manifest publicado usa URLs de GitHub Releases y nombres de asset estables."""
    registry = json.loads(
        (release.PROJECT_DIR / "checkpoints" / "model_registry.json").read_text(
            encoding="utf-8"
        )
    )

    manifest = release.build_manifest(
        registry=registry,
        repo="RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting",
        release_tag="latest",
        output_dir=tmp_path,
        tickers=["NVDA"],
    )

    checkpoint_url = manifest["artifacts"]["NVDA"]["checkpoint"]["url"]
    preprocessing_url = manifest["artifacts"]["NVDA"]["preprocessing"]["url"]
    assert checkpoint_url.endswith("/releases/latest/download/nvda_canonical.pt")
    assert preprocessing_url.endswith(
        "/releases/latest/download/nvda_preprocessing.zip"
    )
