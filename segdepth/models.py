"""
models.py
---------
Local storage of Hugging Face models for offline reuse.

On first use, a model and its image processor are downloaded from the Hub and
saved under <models_dir>/<org>--<name>/ in safetensors format. Later runs load
the local copy. A manifest.json records each model's source ID, the
transformers version used to save it and the save date.

Functions:
  * local_model_dir           Map a Hub ID to its local folder.
  * update_manifest           Record provenance in manifest.json.
  * load_model_and_processor  Load locally, or download and save.
"""

import json
from datetime import datetime
from pathlib import Path

import transformers
from transformers import AutoImageProcessor


def local_model_dir(models_dir: Path, model_id: str) -> Path:
    """facebook/mask2former-... -> models/facebook--mask2former-..."""
    return Path(models_dir) / model_id.replace("/", "--")


def update_manifest(models_dir: Path, model_id: str, local_path: Path) -> None:
    manifest_path = Path(models_dir) / "manifest.json"
    data = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    data[model_id] = {
        "local_path": str(local_path),
        "transformers_version": transformers.__version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(data, indent=2))


def load_model_and_processor(model_id: str, model_cls, models_dir, offline: bool = False):
    """
    Return (processor, model). `model_id` may be a Hub ID or a path to an
    existing local model directory. With offline=True, never contact the Hub.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    as_path = Path(model_id)
    local_path = as_path if as_path.is_dir() else local_model_dir(models_dir, model_id)

    if (local_path / "config.json").exists():
        print(f"  Loading local copy: {local_path}")
        processor = AutoImageProcessor.from_pretrained(local_path, local_files_only=True)
        model = model_cls.from_pretrained(local_path, local_files_only=True)
        return processor, model

    if offline:
        raise FileNotFoundError(
            f"No local copy of '{model_id}' at {local_path} and offline mode is set. "
            f"Run once with network access (e.g. run_image.py --download-only)."
        )

    print(f"  Downloading {model_id} from the Hugging Face Hub...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = model_cls.from_pretrained(model_id)

    local_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(local_path)
    model.save_pretrained(local_path)
    update_manifest(models_dir, model_id, local_path)
    print(f"  Saved local copy to: {local_path}")
    return processor, model
