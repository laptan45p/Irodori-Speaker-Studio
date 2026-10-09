"""Add human-readable output names without changing the training implementation."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "irodori"))


def install(trainer, prefix):
    original = trainer.save_checkpoint

    def save(path, *args, **kwargs):
        path = Path(path)
        if path.name.startswith("checkpoint_") and path.name.endswith(".speaker.safetensors"):
            path = path.with_name(prefix + path.name[len("checkpoint"):])
        result = original(path, *args, **kwargs)
        print(f"[Studio] 話者埋め込みを保存: {path}", flush=True)
        return result

    trainer.save_checkpoint = save


if __name__ == "__main__":
    import train
    from engine import validate_name
    install(train, validate_name(os.environ.get("STUDIO_OUTPUT_PREFIX", "checkpoint")) or "checkpoint")
    train.main()
