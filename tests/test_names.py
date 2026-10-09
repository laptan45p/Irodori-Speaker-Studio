from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine
import named_trainer


def test_name_and_existing_project_rename(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'JOBS', tmp_path)
    source = tmp_path / 'test.wav'
    source.write_bytes(b'audio')
    job_id = engine.create_job([source], {'project_name': 'みあ'})
    job = engine.resolve_job(job_id)
    engine.write(job / 'rows.json', [])
    assert engine.project_name(job) == 'みあ'
    assert engine.project_choices()[0][0].startswith('みあ / ')
    output = job / 'output/run'
    output.mkdir()
    for suffix in ['0000250', 'pause', 'final']:
        (output / f'checkpoint_{suffix}.speaker.safetensors').write_bytes(b'weights')
    (output / 'resume_state.pt').write_bytes(b'optimizer')
    engine.write(job / 'training.json', {'output_dir': str(output), 'context': {'hash': 'same'}})
    engine.rename_project(job_id, '新しい声')
    assert engine.final_embedding(job_id).read_bytes() == b'weights'
    assert (output / '新しい声_0000250.speaker.safetensors').exists()
    assert (output / 'resume_state.pt').read_bytes() == b'optimizer'
    assert engine.read(job / 'training.json')['context'] == {'hash': 'same'}
    assert engine.resolve_job(job_id) == job
    engine.ACTIVE[job_id] = {}
    try:
        with pytest.raises(ValueError, match='中断'):
            engine.rename_project(job_id, '変更')
    finally:
        engine.ACTIVE.pop(job_id)


@pytest.mark.parametrize('value', ['../escape', 'CON', 'NUL.txt', 'a:b', 'test.', 'a'*81])
def test_invalid_windows_name(value):
    with pytest.raises(ValueError):
        engine.validate_name(value)


def test_wrapper_names_outputs_without_touching_state(tmp_path):
    calls = []
    trainer = SimpleNamespace(save_checkpoint=lambda path, *args, **kwargs: calls.append((path, args, kwargs)))
    named_trainer.install(trainer, 'みあ')
    for suffix in ['0000250', 'pause', 'final']:
        trainer.save_checkpoint(tmp_path / f'checkpoint_{suffix}.speaker.safetensors', 'model', step=250)
    assert [p.name for p, _, _ in calls] == ['みあ_0000250.speaker.safetensors', 'みあ_pause.speaker.safetensors', 'みあ_final.speaker.safetensors']
    assert all(args == ('model',) and kwargs == {'step': 250} for _, args, kwargs in calls)


def test_named_resume_keeps_saved_command_and_context(tmp_path, monkeypatch):
    import worker
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'resume_state.pt').write_bytes(b'state')
    command = ['python', '-u', str(worker.UPSTREAM / 'train.py'), '--config', 'same.yaml']
    record = dict(output_dir=str(output), checkpoint='base', command=command,
                  context={'same': True}, output_prefix='みあ')
    worker.write_json(tmp_path / 'training.json', record)
    monkeypatch.setattr(worker, 'training_identity', lambda *args: {'same': True})
    calls = []
    monkeypatch.setattr(worker, 'launch_trainer', lambda cmd: calls.append(cmd))
    for key in ['STUDIO_OUTPUT_PREFIX', 'STUDIO_STATE_DIR', 'STUDIO_PAUSE_FILE',
                'STUDIO_CONTEXT', 'STUDIO_RESUME_STATE']:
        monkeypatch.setenv(key, '')
    worker.resume(tmp_path)
    import os
    assert os.environ['STUDIO_OUTPUT_PREFIX'] == 'みあ'
    assert calls == [command]
    assert worker.read_json(tmp_path / 'training.json') == record
