import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_manual_split import material as _material_fixture

import engine
import worker
from portability import output_path, resume_command, verify_context

material = _material_fixture


def record(job):
    output = job / "output/abcdef123456"
    output.mkdir()
    (output / "resume_state.pt").write_bytes(b"unchanged optimizer, scheduler, RNG, loader")
    engine.write(output / "resume_info.json", {"step": 750, "target_steps": 3000})
    context = {"gpu": "old GPU", "cuda": "12.8", "base_sha256": "base",
               "manifest_sha256": "manifest", "config_sha256": "config",
               "trainer_sha256": "trainer", "extension_sha256": "extension"}
    saved = dict(output_dir=r"F:\old\projects\old-id\output\abcdef123456",
                 checkpoint=r"F:\cache\model.safetensors", context=context,
                 output_prefix="テスト", command=[r"F:\old\python.exe", "-u", r"F:\old\irodori\train.py",
                 "--config", r"F:\old\train_large_speaker.yaml", "--manifest", r"F:\old\manifest.jsonl",
                 "--init-checkpoint", r"F:\cache\model.safetensors", "--output-dir", str(output),
                 "--device", "cuda"])
    engine.write(job / "training.json", saved)
    return saved


def test_transfer_contains_sources_state_and_relocates_without_overwriting(material):
    identity, job, _, _ = material
    saved = record(job)
    archive = engine.export_project(identity)
    with zipfile.ZipFile(archive) as z:
        assert "sources/000.wav" in z.namelist()
        assert "input/000_input.wav" in z.namelist()
        assert "output/abcdef123456/resume_state.pt" in z.namelist()
    transferred = engine.import_project(archive)
    new_job = engine.resolve_job(transferred)
    assert transferred != identity
    assert engine.table(transferred) == engine.table(identity)
    assert (new_job / "rows.json").read_bytes() == (job / "rows.json").read_bytes()
    assert (output_path(new_job, saved) / "resume_state.pt").read_bytes() == (
        job / "output/abcdef123456/resume_state.pt").read_bytes()
    assert all(Path(p).is_file() and str(new_job) in p
               for p in engine.read(new_job / "settings.json")["sources"])
    assert any(value == transferred and "750/3000" in label for label, value in engine.project_choices())


def test_rebased_command_and_hardware_permission(material):
    _, job, _, _ = material
    saved = record(job)
    command = resume_command(saved, job, "/new/model.safetensors", "/new", "/new/irodori", "/new/python")
    assert command[0] == "/new/python" and command[2] == "/new/irodori/train.py"
    assert command[command.index("--manifest")+1] == str(job / "manifest.jsonl")
    assert command[command.index("--output-dir")+1] == str(job / "output/abcdef123456")
    current = dict(saved["context"], gpu="new GPU", cuda="13.0")
    with pytest.raises(ValueError, match="GPU"):
        verify_context(saved, current)
    assert verify_context(saved, current, True)
    with pytest.raises(ValueError, match="モデル"):
        verify_context(saved, dict(current, base_sha256="changed"), True)
    saved["runtime_context"] = current
    assert not verify_context(saved, current)


def test_worker_resume_preserves_state_anchor_and_uses_new_paths(material, monkeypatch):
    _, job, _, _ = material
    saved = record(job)
    for key in ["STUDIO_OUTPUT_PREFIX", "STUDIO_STATE_DIR", "STUDIO_PAUSE_FILE",
                "STUDIO_CONTEXT", "STUDIO_RESUME_STATE"]:
        monkeypatch.setenv(key, "")
    checkpoint = job / "base.safetensors"
    checkpoint.write_bytes(b"base")
    engine.write(job / "resume_request.json", {"checkpoint": str(checkpoint), "allow_environment_change": True})
    current = dict(saved["context"], gpu="new GPU")
    monkeypatch.setattr(worker, "training_identity", lambda *args: current)
    launched = []
    monkeypatch.setattr(worker, "launch_trainer", launched.append)
    before = (job / "output/abcdef123456/resume_state.pt").read_bytes()
    worker.resume(job)
    assert len(launched) == 1
    assert os.environ["STUDIO_STATE_DIR"] == str(job / "output/abcdef123456")
    assert os.environ["STUDIO_RESUME_STATE"] == str(job / "output/abcdef123456/resume_state.pt")
    import json
    assert json.loads(os.environ["STUDIO_CONTEXT"]) == saved["context"]
    assert (job / "output/abcdef123456/resume_state.pt").read_bytes() == before
    engine.write(job / "resume_request.json", {"allow_environment_change": False})
    worker.resume(job)
    assert len(launched) == 2


def test_transfer_refuses_active_export_and_zip_path_escape(material, monkeypatch, tmp_path):
    identity, _, _, _ = material
    monkeypatch.setitem(engine.ACTIVE, identity, {"stage": "train"})
    with pytest.raises(ValueError, match="中断"):
        engine.export_project(identity)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("settings.json", "{}")
        z.writestr("rows.json", "[]")
        z.writestr("sources/000.wav", b"source")
        z.writestr("../outside", "bad")
    with pytest.raises(ValueError, match="不正なパス"):
        engine.import_project(bad)
    assert not (tmp_path / "outside").exists()


def test_initial_embedding_transfer_and_resume_metadata(material, monkeypatch, tmp_path):
    import copy
    import json
    from types import SimpleNamespace

    identity, job, _, _ = material
    saved = record(job)
    initial = tmp_path / "external.speaker.safetensors"
    initial.write_bytes(b"initial embedding")
    settings = engine.read(job / "settings.json")
    settings["initial_embedding"] = str(initial)
    engine.write(job / "settings.json", settings)
    saved["command"] += ["--speaker-inversion-init-embedding", str(initial)]
    engine.write(job / "training.json", saved)
    transferred = engine.import_project(engine.export_project(identity))
    new_job = engine.resolve_job(transferred)
    asset = new_job / "assets/initial.speaker.safetensors"
    assert asset.read_bytes() == initial.read_bytes()
    assert engine.read(new_job / "settings.json")["initial_embedding"] == str(asset)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"base")
    engine.write(new_job / "resume_request.json", {"checkpoint": str(checkpoint)})
    payload = {"train_config": {"speaker_inversion_init_embedding": str(initial)},
               "optimizer": {"step": 750}, "rng": [1, 2, 3], "loader": {"epoch": 2}}
    original = copy.deepcopy(payload)
    # Exercise the metadata relocation with a serialization stub; no tensor training.
    fake_torch = SimpleNamespace(load=lambda *a, **kw: copy.deepcopy(payload),
                                 save=lambda data, path: Path(path).write_text(json.dumps(data)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(worker, "training_identity", lambda *args: saved["context"])
    launched = []
    monkeypatch.setattr(worker, "launch_trainer", launched.append)
    for key in ["STUDIO_OUTPUT_PREFIX", "STUDIO_STATE_DIR", "STUDIO_PAUSE_FILE",
                "STUDIO_CONTEXT", "STUDIO_RESUME_STATE"]:
        monkeypatch.setenv(key, "")
    worker.resume(new_job)
    relocated = json.loads(Path(os.environ["STUDIO_RESUME_STATE"]).read_text())
    assert relocated.pop("train_config") == {"speaker_inversion_init_embedding": str(asset)}
    original.pop("train_config")
    assert relocated == original
    assert (output_path(new_job, saved) / "resume_state.pt").read_bytes() == b"unchanged optimizer, scheduler, RNG, loader"
    assert launched[0][-1] == str(asset)
