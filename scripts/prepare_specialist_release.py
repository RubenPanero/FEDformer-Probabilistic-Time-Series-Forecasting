"""Prepara assets de release y genera el manifest descargable de especialistas."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = PROJECT_DIR / "checkpoints" / "model_registry.json"
DEFAULT_MANIFEST_PATH = PROJECT_DIR / "checkpoints" / "artifacts.manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "artifacts_archive" / "specialist_release_assets"
DEFAULT_REPO = "RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting"


def parse_args() -> argparse.Namespace:
    """Define la CLI del preparador de release."""
    parser = argparse.ArgumentParser(
        description=(
            "Empaqueta checkpoints y preprocessing artifacts para GitHub Releases "
            "y genera checkpoints/artifacts.manifest.json."
        )
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Ruta al model_registry.json local.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Ruta donde escribir el manifest versionado.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio local donde dejar los assets listos para subir a Releases.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Repositorio GitHub owner/name usado para construir las URLs de release.",
    )
    parser.add_argument(
        "--release-tag",
        default="latest",
        help=(
            "Tag de release a usar en las URLs. Usa 'latest' para releases/latest/download, "
            "o un tag explícito como v0.1.0."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Subset de tickers a empaquetar. Por defecto empaqueta todos los del registry.",
    )
    return parser.parse_args()


def load_registry(registry_path: Path) -> dict:
    """Carga el model_registry desde disco."""
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    """Calcula SHA256 de un archivo."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_release_url(repo: str, release_tag: str, asset_name: str) -> str:
    """Construye la URL pública de descarga del asset."""
    if release_tag == "latest":
        return f"https://github.com/{repo}/releases/latest/download/{asset_name}"
    return f"https://github.com/{repo}/releases/download/{release_tag}/{asset_name}"


def create_deterministic_zip(source_dir: Path, destination_zip: Path) -> list[str]:
    """Empaqueta un directorio de forma determinista y retorna la lista de archivos."""
    file_list: list[str] = []
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination_zip, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_dir).as_posix()
            file_list.append(relative)
            info = ZipInfo(relative)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return file_list


def copy_release_checkpoint(source_path: Path, output_dir: Path) -> Path:
    """Copia el checkpoint local al directorio de assets de release."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source_path.name
    shutil.copy2(source_path, destination)
    return destination


def build_manifest_entry(
    ticker: str,
    entry: dict,
    repo: str,
    release_tag: str,
    output_dir: Path,
) -> dict:
    """Construye la entrada de manifest y deja listos los assets locales."""
    checkpoint_path = PROJECT_DIR / entry["checkpoint"]
    preprocessing_dir = PROJECT_DIR / entry["data"]["preprocessing_artifacts"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    if not preprocessing_dir.exists():
        raise FileNotFoundError(
            f"Directorio de preprocessing no encontrado: {preprocessing_dir}"
        )

    checkpoint_asset = copy_release_checkpoint(checkpoint_path, output_dir)
    preprocessing_asset = output_dir / f"{preprocessing_dir.name}.zip"
    required_files = create_deterministic_zip(preprocessing_dir, preprocessing_asset)

    checkpoint_name = checkpoint_asset.name
    preprocessing_name = preprocessing_asset.name

    return {
        "registry_key": ticker,
        "checkpoint": {
            "asset_name": checkpoint_name,
            "url": make_release_url(repo, release_tag, checkpoint_name),
            "sha256": sha256_file(checkpoint_asset),
            "size_bytes": checkpoint_asset.stat().st_size,
            "install_path": entry["checkpoint"],
        },
        "preprocessing": {
            "asset_name": preprocessing_name,
            "url": make_release_url(repo, release_tag, preprocessing_name),
            "sha256": sha256_file(preprocessing_asset),
            "size_bytes": preprocessing_asset.stat().st_size,
            "install_dir": entry["data"]["preprocessing_artifacts"],
            "required_files": required_files,
            "format": "zip",
        },
    }


def build_manifest(
    registry: dict,
    repo: str,
    release_tag: str,
    output_dir: Path,
    tickers: list[str] | None = None,
) -> dict:
    """Construye el manifest completo a partir del registry."""
    specialists = registry.get("specialists", {})
    selected = tickers if tickers else list(specialists.keys())
    generated = {
        "version": 1,
        "repo": repo,
        "release_tag": release_tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {},
    }

    for ticker in selected:
        if ticker not in specialists:
            raise KeyError(f"Ticker '{ticker}' no encontrado en model_registry.json")
        generated["artifacts"][ticker] = build_manifest_entry(
            ticker=ticker,
            entry=specialists[ticker],
            repo=repo,
            release_tag=release_tag,
            output_dir=output_dir,
        )
    return generated


def main() -> int:
    """Punto de entrada del preparador de release."""
    args = parse_args()
    registry = load_registry(args.registry_path)
    manifest = build_manifest(
        registry=registry,
        repo=args.repo,
        release_tag=args.release_tag,
        output_dir=args.output_dir,
        tickers=args.tickers,
    )

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest escrito en: {args.manifest_path}")
    print(f"Assets listos para subir a release en: {args.output_dir}")
    for ticker, artifact_info in manifest["artifacts"].items():
        print(
            f"- {ticker}: {artifact_info['checkpoint']['asset_name']}, "
            f"{artifact_info['preprocessing']['asset_name']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
