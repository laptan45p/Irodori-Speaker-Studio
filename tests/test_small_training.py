"""Model selection through config generation, UI persistence and relocation."""
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine
import worker
from model_profiles import DEFAULT_MODEL, MODEL_PROFILES, model_profile
from training_config import make_config

ROOT = Path(__file__).resolve().parents[1]


def base_config(variant):
    filename = "train_v4_large_duration.yaml" if variant == DEFAULT_MODEL else "train_v4_small.yaml"
    return yaml.safe_load((ROOT / "irodori/configs" / filename).read_text(encoding="utf-8"))["model"]


@pytest.mark.parametrize("variant", MODEL_PROFILES)
def test_architecture_and_duration_are_preserved(variant):
    model = base_config(variant)
    defaults = yaml.safe_load((ROOT / "irodori/configs/train_v4_small_speaker_inversion.yaml").read_text(encoding="utf-8"))["train"]
    generated = make_config({"config_json": json.dumps(model)}, set(model), defaults,
                            [{"num_frames": 2100, "text": "長い文章"}],
                            dict(model_variant=variant, batch_size=1, accumulation=1, steps=10, lr=0.01), 600)
    assert generated["model"] == model
    assert generated["train"]["max_latent_steps"] == 2100
    assert generated["train"]["max_text_len"] == 608
    assert generated["train"]["duration_loss_weight"] == 0
    assert generated["train"]["speaker_inversion_enabled"]


@pytest.mark.parametrize("variant", MODEL_PROFILES)
def test_mismatched_model_is_rejected(variant):
    other = next(v for v in MODEL_PROFILES if v != variant)
    model = base_config(other)
    with pytest.raises(ValueError, match="構成"):
        make_config({"config_json": json.dumps(model)}, set(model), {}, [],
                    {"model_variant": variant}, 20)


@pytest.mark.parametrize("variant", [None, *MODEL_PROFILES])
def test_ui_settings_restore_selection_including_legacy(variant, tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(engine, "JOBS", tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    args = [5, 10, "large-v3", "cpu", "ja", "", 10, 0.01, 1, 1, "", "", "声"]
    cfg = app.settings(*args, *([variant] if variant else []))
    if variant is None:
        cfg.pop("model_variant")
    identity = engine.create_job([source], cfg)
    engine.write(engine.resolve_job(identity) / "rows.json", [])
    restored = app.load(identity)
    assert restored[-1] == (variant or DEFAULT_MODEL)
    assert restored[-2] == "声"


@pytest.mark.parametrize("variant", MODEL_PROFILES)
def test_worker_download_config_and_provenance(variant, tmp_path, monkeypatch):
    import torch
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from safetensors.torch import save_file

    profile = model_profile(variant)
    checkpoint = tmp_path / "model.safetensors"
    model = base_config(variant)
    save_file({"dummy": torch.zeros(1)}, checkpoint, metadata={"config_json": json.dumps(model)})
    cfg = dict(model_variant=variant, checkpoint="", batch_size=1, accumulation=1, steps=10, lr=0.01)
    engine.write(tmp_path / "settings.json", cfg)
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"text": "テスト", "num_frames": 1200}), encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    downloads, tokenizers, commands = [], [], []

    def download(repo, **kwargs):
        downloads.append(repo)
        return str(tmp_path)

    def tokenizer(repo, **kwargs):
        tokenizers.append(repo)
        return SimpleNamespace(encode=lambda text: [1, 2, 3])

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr(PretrainedTextTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(worker, "setup_resume_environment", lambda *args: None)
    monkeypatch.setattr(worker, "launch_trainer", commands.append)
    worker.train(tmp_path)
    assert downloads == [profile["repo"]]
    assert tokenizers == [profile["text_tokenizer_repo"]]
    generated = yaml.safe_load((tmp_path / profile["config_file"]).read_text(encoding="utf-8"))
    assert generated["model"] == model
    assert engine.read(tmp_path / "training.json")["model_variant"] == variant
    assert engine.read(tmp_path / "provenance.json")["model_repo"] == profile["repo"]
    assert commands[0][commands[0].index("--config") + 1] == str(tmp_path / profile["config_file"])


@pytest.mark.parametrize("variant", [None, *MODEL_PROFILES])
def test_resume_download_uses_saved_training_selection(variant, tmp_path, monkeypatch):
    profile = model_profile(variant or DEFAULT_MODEL)
    output = tmp_path / "output/run"
    output.mkdir(parents=True)
    (output / "resume_state.pt").write_bytes(b"state")
    saved = dict(output_dir=str(output), checkpoint=str(tmp_path / "missing.safetensors"),
                 context={}, command=["python", "-u", "train.py", "--config", "old.yaml",
                                      "--manifest", "old.jsonl", "--init-checkpoint", "old.safetensors",
                                      "--output-dir", "old", "--device", "cuda"])
    if variant:
        saved["model_variant"] = variant
    engine.write(tmp_path / "training.json", saved)
    # Current UI settings may have changed since this run was saved.
    engine.write(tmp_path / "settings.json", {"model_variant": "unrelated"})
    downloads, launched = [], []
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda repo, **kwargs: downloads.append(repo) or str(tmp_path))
    monkeypatch.setattr(worker, "training_identity", lambda *args: {})
    monkeypatch.setattr(worker, "launch_trainer", launched.append)
    for key in ["STUDIO_OUTPUT_PREFIX", "STUDIO_STATE_DIR", "STUDIO_PAUSE_FILE",
                "STUDIO_CONTEXT", "STUDIO_RESUME_STATE"]:
        monkeypatch.setenv(key, "")
    worker.resume(tmp_path)
    assert downloads == [profile["repo"]]
    command = launched[0]
    assert command[command.index("--config") + 1] == str(tmp_path / profile["config_file"])
    assert (output / "resume_state.pt").read_bytes() == b"state"


def test_small_transfer_preserves_config_and_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "JOBS", tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    identity = engine.create_job([source], {"model_variant": "v4.1-Small"})
    job = engine.resolve_job(identity)
    (job / "sources/000.wav").write_bytes(b"audio")
    engine.write(job / "rows.json", [])
    (job / "train_small_speaker.yaml").write_bytes(b"model: unchanged")
    with zipfile.ZipFile(engine.export_project(identity)) as archive:
        assert archive.read("train_small_speaker.yaml") == b"model: unchanged"
    transferred = engine.import_project(job / "project-transfer.zip")
    relocated = engine.resolve_job(transferred)
    assert engine.read(relocated / "settings.json")["model_variant"] == "v4.1-Small"
    assert (relocated / "train_small_speaker.yaml").read_bytes() == b"model: unchanged"


def test_small_access_check_does_not_require_large_gated_encoder(monkeypatch):
    import hf_access
    requested = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_url=lambda repo, filename: requested.append(repo) or "url",
        get_hf_file_metadata=lambda *args, **kwargs: None,
    ))
    assert "ModernBERT" in hf_access.check_access("v4.1-Small")
    assert requested == [model_profile("v4.1-Small")["text_tokenizer_repo"]]


def test_ui_wires_selection_to_start_train_and_restore(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(engine, "JOBS", tmp_path)
    demo = app.build()
    selector = next(c for c in demo.blocks.values() if getattr(c, "label", None) == "学習モデル")
    assert selector.value == DEFAULT_MODEL
    for callback in [app.prepare_start, app.train_existing]:
        fn = next(fn for fn in demo.fns.values() if fn.fn is callback)
        assert fn.inputs[-1] is selector
    for callback in [app.load_selected, app.import_transfer]:
        fn = next(fn for fn in demo.fns.values() if fn.fn is callback)
        assert selector in fn.outputs
    access = [fn for fn in demo.fns.values() if fn.fn is app.check_access]
    assert len(access) == 3 and all(fn.inputs == [selector] for fn in access)


@pytest.mark.parametrize("variant", [None, "v4.1-Small"])
def test_resume_identity_hashes_saved_model_config(variant, tmp_path, monkeypatch):
    import studio_state
    import torch
    saved = {"model_variant": variant} if variant else {}
    engine.write(tmp_path / "training.json", saved)
    (tmp_path / "train_large_speaker.yaml").write_bytes(b"large config")
    (tmp_path / "train_small_speaker.yaml").write_bytes(b"small config")
    (tmp_path / "manifest.jsonl").write_bytes(b"")
    (tmp_path / "rows.json").write_bytes(b"[]")
    checkpoint = tmp_path / "base.safetensors"
    checkpoint.write_bytes(b"base")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test GPU")
    identity = worker.training_identity(tmp_path, checkpoint)
    expected = model_profile(variant or DEFAULT_MODEL)["config_file"]
    assert identity["config_sha256"] == studio_state.digest(tmp_path / expected)
