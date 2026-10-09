"""Real CPU AdamW/RNG/StatefulDataLoader continuation checks (no Large GPU model)."""

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "irodori"))
import studio_state


@dataclass
class Config:
    max_steps: int = 40


class RandomDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 40

    def __getitem__(self, index):
        return torch.tensor([index / 40 + random.random(), random.random()], dtype=torch.float32)


def objects(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    generator = torch.Generator().manual_seed(seed)
    loader = StatefulDataLoader(
        RandomDataset(), batch_size=2, shuffle=True, num_workers=0, generator=generator
    )
    return model, optimizer, scheduler, loader


def update(model, opt, scheduler, batch):
    noise = torch.randn_like(batch)
    loss = model(batch + noise).square().mean()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    scheduler.step()
    return float(loss.detach())


def test_exact_cpu_continuation(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CONTEXT", '{"base":"same"}')
    model, opt, scheduler, loader = objects()
    iterator = iter(loader)
    for _ in range(7):
        update(model, opt, scheduler, next(iterator))
    studio_state.save(
        tmp_path,
        model,
        opt,
        scheduler,
        7,
        Config(),
        Config(),
        {"rank_states": [loader.state_dict()], "studio_generator": loader.generator.get_state()},
        {"sampler_epoch": 0, "epoch_step": 7},
    )
    expected = []
    for _ in range(8):
        batch = next(iterator)
        expected.append((batch.clone(), update(model, opt, scheduler, batch)))
    final = {n: p.detach().clone() for n, p in model.named_parameters()}
    # Reconstruct a new process's objects, intentionally disturb all RNG.
    resumed, opt2, scheduler2, loader2 = objects(999)
    state = studio_state.restore(
        tmp_path / "resume_state.pt", resumed, opt2, scheduler2, Config(), Config()
    )
    loader2.load_state_dict(state["dataloader"]["rank_states"][0])
    studio_state.restore_rng(state["rng"])
    iterator2 = studio_state.restored_iterator(loader2, state)
    for expected_batch, expected_loss in expected:
        batch = next(iterator2)
        assert torch.equal(batch, expected_batch)
        assert update(resumed, opt2, scheduler2, batch) == expected_loss
    assert all(torch.equal(p, final[n]) for n, p in resumed.named_parameters())
    assert scheduler2.state_dict() == scheduler.state_dict()


def test_changed_context_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CONTEXT", '{"base":"a"}')
    model, opt, scheduler, loader = objects()
    studio_state.save(tmp_path, model, opt, scheduler, 0, Config(), Config(), {}, {})
    monkeypatch.setenv("STUDIO_CONTEXT", '{"base":"b"}')
    with pytest.raises(ValueError, match="変更"):
        studio_state.restore(
            tmp_path / "resume_state.pt", model, opt, scheduler, Config(), Config()
        )


def test_atomic_failure_preserves_previous_state(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CONTEXT", "{}")
    model, opt, scheduler, loader = objects()
    studio_state.save(tmp_path, model, opt, scheduler, 2, Config(), Config(), {}, {})

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(studio_state.torch, "save", fail)
    with pytest.raises(OSError):
        studio_state.save(tmp_path, model, opt, scheduler, 3, Config(), Config(), {}, {})
    assert torch.load(tmp_path / "resume_state.pt", weights_only=True)["step"] == 2


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@pytest.mark.parametrize("saved_steps", [7, 19, 20])
def test_resume_in_separate_process(tmp_path, monkeypatch, saved_steps):
    import subprocess

    monkeypatch.setenv("STUDIO_CONTEXT", '{"base":"same"}')
    model, opt, scheduler, loader = objects()
    iterator = iter(loader)
    for _ in range(saved_steps):
        batch, iterator = next_batch(loader, iterator)
        update(model, opt, scheduler, batch)
    studio_state.save(
        tmp_path,
        model,
        opt,
        scheduler,
        saved_steps,
        Config(),
        Config(),
        {"rank_states": [loader.state_dict()], "studio_generator": loader.generator.get_state()},
        {"sampler_epoch": 0, "epoch_step": saved_steps},
    )
    for _ in range(8):
        batch, iterator = next_batch(loader, iterator)
        update(model, opt, scheduler, batch)
    expected = {n: p.detach().clone() for n, p in model.named_parameters()}
    script = """import sys,torch
from test_resume import objects,Config,update,studio_state,next_batch
from pathlib import Path
folder=Path(sys.argv[1])
model,opt,scheduler,loader=objects(999)
state=studio_state.restore(folder/'resume_state.pt',model,opt,scheduler,Config(),Config())
loader.load_state_dict(state['dataloader']['rank_states'][0])
studio_state.restore_rng(state['rng'])
iterator=studio_state.restored_iterator(loader,state)
for i in range(8):
    batch,iterator=next_batch(loader,iterator)
    update(model,opt,scheduler,batch)
torch.save({n:p.detach() for n,p in model.named_parameters()},folder/'child_result.pt')
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], cwd=Path(__file__).parent, check=True
    )
    result = torch.load(tmp_path / "child_result.pt", weights_only=True)
    assert all(torch.equal(result[n], p) for n, p in expected.items())


def test_pause_request_is_graceful(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import engine

    monkeypatch.setattr(engine, "JOBS", tmp_path)
    job_id = "a" * 32
    (tmp_path / job_id).mkdir()
    engine.ACTIVE[job_id] = {"stage": "train", "process": None, "cancelled": False}
    try:
        message = engine.pause(job_id)
        assert "完了" in message
        assert (tmp_path / job_id / "pause.request").exists()
        assert not engine.ACTIVE[job_id]["cancelled"]
    finally:
        engine.ACTIVE.pop(job_id, None)


def test_stage_marks_paused_instead_of_completed(tmp_path, monkeypatch):
    import subprocess

    import engine

    monkeypatch.setattr(engine, "JOBS", tmp_path)
    job_id = "b" * 32
    folder = tmp_path / job_id
    folder.mkdir()
    original = subprocess.Popen

    def fake(*args, **kwargs):
        script = (
            "from pathlib import Path; Path(r'"
            + str(folder / "pause.request.complete")
            + "').write_text('100'); print('saved pause')"
        )
        return original([sys.executable, "-u", "-c", script], **kwargs)

    monkeypatch.setattr(engine.subprocess, "Popen", fake)
    list(engine.run_stage(job_id, "train"))
    assert engine.read(folder / "status.json")["state"] == "paused"
    assert engine.is_paused(job_id)


def test_resume_ui_skips_encoding(tmp_path, monkeypatch):
    import app
    import engine

    monkeypatch.setattr(engine, "JOBS", tmp_path)
    job_id = "c" * 32
    folder = tmp_path / job_id
    folder.mkdir()
    calls = []

    def stage(job_id, name):
        calls.append(name)
        (folder / "pause.request.complete").write_text("120")
        yield "paused again"

    monkeypatch.setattr(engine, "run_stage", stage)
    monkeypatch.setattr(engine, "bundle", lambda job_id: "result.zip")
    result = list(app.resume_training(job_id))
    assert calls == ["resume"]
    assert "中断" in result[-1][0]
