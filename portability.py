"""Relocate project paths without modifying hashed training data or state."""
from pathlib import Path

from model_profiles import DEFAULT_MODEL, model_profile

LEGACY_RESUME_EXTENSION = "7b290557a8f7eb7076f65c0d2b0af43410afb63493e4c669215a0b24e71de8a0"


def basename(value):
    return str(value).replace("\\", "/").rstrip("/").split("/")[-1]


def output_path(job, record):
    name = basename(record["output_dir"])
    if not name or name in {".", ".."}:
        raise ValueError("学習出力先が不正です。")
    return Path(job) / "output" / name


def training_config_path(job, record):
    return Path(job) / model_profile(record.get("model_variant", DEFAULT_MODEL))["config_file"]


def resume_command(record, job, checkpoint, root, upstream, python):
    command = list(record["command"])
    if len(command) < 3 or command[1] != "-u":
        raise ValueError("保存済み学習コマンドの形式が不正です。")
    command[0], command[2] = str(python), str(Path(upstream) / "train.py")
    for flag, value in {
        "--config": training_config_path(job, record),
        "--manifest": Path(job) / "manifest.jsonl",
        "--init-checkpoint": checkpoint,
        "--output-dir": output_path(job, record),
    }.items():
        if command.count(flag) != 1:
            raise ValueError("保存済み学習コマンドが不正です: " + flag)
        command[command.index(flag) + 1] = str(value)
    return command


def verify_context(record, current, allow_environment_change=False):
    expected = record.get("runtime_context", record["context"])
    def essential(value):
        return {k: v for k, v in value.items() if k not in {"gpu", "cuda"}}
    original = essential(record["context"])
    actual = essential(current)
    if original.get("extension_sha256") == LEGACY_RESUME_EXTENSION:
        import hashlib
        installed = Path(__file__).resolve().parent / "irodori/studio_state.py"
        if actual.get("extension_sha256") == hashlib.sha256(installed.read_bytes()).hexdigest():
            # Only this recognized legacy extension can use the corrected reader.
            # Keep the state's immutable original context and all other hashes.
            original["extension_sha256"] = actual["extension_sha256"]
    if original != actual:
        raise ValueError("モデル・素材・設定・学習コードが変更されています。再開できません。")
    changed = any(expected.get(k) != current.get(k) for k in ("gpu", "cuda"))
    if changed and not allow_environment_change:
        raise ValueError("GPU・CUDAが変わっています。移行先環境の違いを許可して再開してください。")
    return changed
