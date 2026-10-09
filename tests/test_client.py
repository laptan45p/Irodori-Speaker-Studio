import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine
from segmentation import Word, padded_bounds, segment_words
from training_config import make_config


def test_sentence_priority_over_ten_seconds():
    words = [Word(0, 5, "今日は"), Word(5, 14, "長い文です。"), Word(14.2, 20.2, "次の文です。")]
    spans = segment_words(words)
    assert [s.text for s in spans] == ["今日は長い文です。", "次の文です。"]
    assert spans[0].end == 14


def test_short_sentences_merge_without_losing_words():
    words = [Word(0, 2, "一文目。"), Word(2.2, 4.2, "二文目。"), Word(4.4, 6.4, "三文目。")]
    result = segment_words(words)
    assert len(result) == 1
    assert result[0].text == "".join(w.text for w in words)


def test_long_pause_is_not_merged():
    result = segment_words([Word(0, 2, "はい。"), Word(8, 11, "わかりました。")])
    assert len(result) == 2


def test_no_punctuation_preserves_long_continuous_phrase():
    words = [Word(i, i + 1, "あ") for i in range(20)]
    result = segment_words(words)
    assert len(result) == 1 and result[0].end == 20


def test_fallback_uses_gap_and_padding_has_no_overlap():
    words = [Word(0, 6, "一つ目"), Word(6.9, 12.9, "二つ目")]
    spans = segment_words(words)
    assert len(spans) == 2
    bounds = padded_bounds(spans, 13)
    assert bounds[0][0] == 0 and bounds[0][1] <= bounds[1][0]
    assert bounds[-1][1] <= 13


def test_randomized_word_coverage():
    import random

    rng = random.Random(4)
    for _ in range(100):
        words, t = [], 0.0
        for i in range(50):
            length = rng.uniform(0.1, 1.2)
            words.append(Word(t, t + length, str(i) + ("。" if rng.random() < 0.2 else ",")))
            t += length + rng.uniform(0, 0.2)
        result = segment_words(words)
        assert "".join(s.text for s in result) == "".join(w.text for w in words)
        assert all(a.end <= b.start for a, b in zip(result, result[1:]))


def test_large_config_preserves_architecture_and_does_not_crop():
    import yaml

    large = yaml.safe_load((ROOT / "irodori/configs/train_v4_large_duration.yaml").read_text())[
        "model"
    ]
    defaults = yaml.safe_load(
        (ROOT / "irodori/configs/train_v4_small_speaker_inversion.yaml").read_text()
    )["train"]
    cfg = dict(batch_size=1, accumulation=2, steps=10, lr=0.01)
    model = dict(large, max_text_len=256)
    generated = make_config(
        {"config_json": json.dumps(model)},
        set(large),
        defaults,
        [{"num_frames": 2000, "text": "長い文章"}],
        cfg,
        600,
    )
    assert generated["model"] == large
    assert generated["train"]["max_latent_steps"] == 2000
    assert generated["train"]["max_text_len"] == 608
    assert generated["train"]["speaker_inversion_enabled"]
    assert generated["train"]["optimizer"] == "adamw"
    assert generated["train"]["duration_loss_weight"] == 0
    with pytest.raises(ValueError):
        make_config({"config_json": "{}"}, set(large), defaults, [], cfg, 10)


@pytest.fixture
def job(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf

    monkeypatch.setattr(engine, "JOBS", tmp_path)
    source = tmp_path / "original.wav"
    sf.write(source, np.zeros(480000, dtype="float32"), 48000)
    job_id = engine.create_job([str(source)], {"test": True})
    folder = engine.resolve_job(job_id)
    sf.write(folder / "sources/000.wav", np.zeros(480000, dtype="float32"), 48000)
    engine.write(
        folder / "rows.json",
        [
            dict(
                use=True,
                id="000_00000",
                source=0,
                start=0,
                end=6,
                duration=6,
                text="テスト。",
                note="",
                confidence=1,
            )
        ],
    )
    return job_id, folder


def test_edit_export_and_restore(job):
    import zipfile

    import soundfile as sf

    job_id, folder = job
    values = engine.table(job_id)
    values[0][2:4] = [1, 9]
    values[0][5] = "修正した文字。"
    engine.save_table(job_id, values)
    data, sr = sf.read(folder / "clips/000_00000.wav")
    assert len(data) / sr == 8
    assert engine.table(job_id)[0][5] == "修正した文字。"
    with zipfile.ZipFile(engine.bundle(job_id)) as archive:
        assert "rows.json" in archive.namelist()
        assert "clips/000_00000.wav" in archive.namelist()
    values[0][3] = 20
    with pytest.raises(ValueError):
        engine.save_table(job_id, values)
    assert sf.info(folder / "clips/000_00000.wav").duration == 8


def test_id_cannot_escape_project():
    with pytest.raises(ValueError):
        engine.resolve_job("../../etc")


def test_cancellation_between_phases(job):
    job_id, _ = job
    engine.cancel(job_id)
    with pytest.raises(RuntimeError, match="停止"):
        list(engine.run_stage(job_id, "train"))
    engine.CANCELLED.discard(job_id)


def test_process_stage_log_and_status(job, monkeypatch):
    job_id, folder = job
    original = engine.subprocess.Popen

    def fake_worker(*args, **kwargs):
        return original([sys.executable, "-c", 'print("mock phase completed")'], **kwargs)

    monkeypatch.setattr(engine.subprocess, "Popen", fake_worker)
    logs = list(engine.run_stage(job_id, "split"))
    assert "mock phase completed" in logs[-1]
    assert engine.read(folder / "status.json")["state"] == "completed"
    assert not engine.ACTIVE


def test_failure_cleans_active_state(job, monkeypatch):
    job_id, folder = job
    original = engine.subprocess.Popen

    def fake_worker(*args, **kwargs):
        return original([sys.executable, "-c", 'raise RuntimeError("intentional")'], **kwargs)

    monkeypatch.setattr(engine.subprocess, "Popen", fake_worker)
    with pytest.raises(RuntimeError, match="intentional"):
        list(engine.run_stage(job_id, "encode"))
    assert engine.read(folder / "status.json")["state"] == "failed"
    assert not engine.ACTIVE


def test_ui_builds():
    import app

    demo = app.build()
    assert demo.get_config_file()["mode"] == "blocks"


def test_multifile_split_audio_pipeline(tmp_path, monkeypatch):
    import types

    import numpy as np
    import soundfile as sf

    import worker

    job = tmp_path / "job"
    for name in ["sources", "clips"]:
        (job / name).mkdir(parents=True)
    sources = []
    for i in range(2):
        path = tmp_path / f"input{i}.wav"
        sf.write(path, np.sin(np.arange(120 * 48000) * 0.02).astype("float32") * 0.05, 48000)
        sources.append(str(path))
    worker.write_json(
        job / "settings.json",
        dict(
            sources=sources,
            asr_model="mock",
            asr_device="cpu",
            language="ja",
            minimum=5,
            maximum=10,
        ),
    )

    class FakeWhisper:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            words = [
                types.SimpleNamespace(
                    start=i * 4 + 0.1, end=i * 4 + 3.8, word=f"文章{i}。", probability=0.99
                )
                for i in range(30)
            ]
            return iter([types.SimpleNamespace(words=words)]), None

    whisper = types.ModuleType("faster_whisper")
    whisper.WhisperModel = FakeWhisper
    decoder = types.ModuleType("faster_whisper.audio")
    decoder.decode_audio = lambda path, sampling_rate: sf.read(path, dtype="float32")[0]
    monkeypatch.setitem(sys.modules, "faster_whisper", whisper)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", decoder)
    worker.split(job)
    rows = worker.read_json(job / "rows.json")
    assert len(rows) == 30
    assert len({r["id"] for r in rows}) == 30
    assert {r["source"] for r in rows} == {0, 1}
    assert all(5 <= r["duration"] <= 10 for r in rows)
    for row in rows:
        assert sf.info(job / "clips" / (row["id"] + ".wav")).frames == round(
            row["duration"] * 48000
        )


def test_cancel_running_process(job, monkeypatch):
    job_id, folder = job
    original = engine.subprocess.Popen

    def fake_worker(*args, **kwargs):
        return original([sys.executable, "-c", "import time; time.sleep(20)"], **kwargs)

    monkeypatch.setattr(engine.subprocess, "Popen", fake_worker)
    stream = engine.run_stage(job_id, "split")
    next(stream)
    assert "停止" in engine.cancel(job_id)
    with pytest.raises(RuntimeError, match="停止"):
        list(stream)
    assert engine.read(folder / "status.json")["state"] == "cancelled"
    assert not engine.ACTIVE
    engine.CANCELLED.discard(job_id)


def test_one_click_ui_pipeline(job, monkeypatch):
    import app

    job_id, folder = job
    calls = []

    def stage(job_id, name):
        calls.append(name)
        if name == "train":
            output = folder / "output/test"
            output.mkdir(parents=True, exist_ok=True)
            (output / "checkpoint_final.speaker.safetensors").write_bytes(b"mock embedding")
            engine.write(folder / "training.json", {"output_dir": str(output)})
        yield "mock " + name

    monkeypatch.setattr(engine, "create_job", lambda files, cfg: job_id)
    monkeypatch.setattr(engine, "run_stage", stage)
    args = [5, 10, "large-v3", "cpu", "ja", "", 10, 0.01, 1, 1, "", ""]
    results = list(app.start(["mock.wav"], "分割から学習まで一括実行", *args))
    assert calls == ["split", "encode", "train"]
    assert "完了" in results[-1][1]
    assert Path(results[-1][-1]).is_file()


def test_terminal_streams_worker_output_and_status(job, monkeypatch, capsys):
    job_id, folder = job
    original = engine.subprocess.Popen

    def fake_worker(*args, **kwargs):
        return original(
            [sys.executable, "-u", "-c", 'print("文字起こし 1/2"); print("学習ログ loss=0.12")'],
            **kwargs,
        )

    monkeypatch.setattr(engine.subprocess, "Popen", fake_worker)
    list(engine.run_stage(job_id, "split"))
    captured = capsys.readouterr().out
    assert "文字起こし・音声分割 開始" in captured
    assert "文字起こし・音声分割 完了" in captured
    assert captured.count("文字起こし 1/2") == 1
    assert captured.count("学習ログ loss=0.12") == 1
    assert "学習ログ loss=0.12" in (folder / "run.log").read_text(encoding="utf-8")


def test_terminal_preserves_partial_utf8_without_old_logs(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_bytes("以前のログ\n".encode("utf-8"))
    offset = log.stat().st_size
    console = engine.ConsoleLog(log, offset)
    payload = "日本語の進捗\n".encode("utf-8")
    with log.open("ab") as writer:
        for byte in payload:
            writer.write(bytes([byte]))
            writer.flush()
            console.forward()
    console.forward(final=True)
    console.close()
    assert capsys.readouterr().out == "日本語の進捗\n"


def test_terminal_reports_failure(job, monkeypatch, capsys):
    job_id, _ = job
    original = engine.subprocess.Popen

    def fake_worker(*args, **kwargs):
        return original(
            [sys.executable, "-u", "-c", 'print("GPU ERROR"); raise SystemExit(3)'], **kwargs
        )

    monkeypatch.setattr(engine.subprocess, "Popen", fake_worker)
    with pytest.raises(RuntimeError):
        list(engine.run_stage(job_id, "train"))
    captured = capsys.readouterr().out
    assert "GPU ERROR" in captured
    assert "失敗（終了コード 3）" in captured


def test_terminal_heartbeat_when_worker_is_silent(job, monkeypatch, capsys):
    from types import SimpleNamespace

    job_id, _ = job
    ticks = iter([0, 0, 16, 16])
    monkeypatch.setattr(
        engine,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(ticks),
            strftime=lambda fmt: "20:00:00",
            sleep=lambda seconds: None,
        ),
    )
    polls = iter([None, 0])
    process = SimpleNamespace(poll=lambda: next(polls, 0), returncode=0, pid=12345)
    monkeypatch.setattr(engine.subprocess, "Popen", lambda *args, **kwargs: process)
    list(engine.run_stage(job_id, "split"))
    assert "実行中（経過 0分16秒）" in capsys.readouterr().out


def test_split_only_returns_rows_and_committed_refresh(job, monkeypatch):
    import app
    job_id, folder = job
    engine.save_table(job_id, engine.table(job_id))
    (folder / 'run.log').write_text('文字起こし・音声分割 完了', encoding='utf-8')
    calls = []
    monkeypatch.setattr(engine, 'create_job', lambda files, cfg: job_id)
    def stage(value, name):
        calls.append(name)
        yield '分割のログ'
    monkeypatch.setattr(engine, 'run_stage', stage)
    args = [5, 10, 'large-v3', 'cpu', 'ja', '', 10, 0.01, 1, 1, '', '']
    results = list(app.start(['mock.wav'], '分割して確認', *args))
    assert calls == ['split']
    assert results[-1][2] == engine.table(job_id)
    assert results[-1][3]['value'] == '000_00000'
    rows, clip, audio, log, projects = app.refresh_results(job_id)
    assert rows == engine.table(job_id)
    assert clip['choices'] == ['000_00000']
    assert Path(audio).is_file()
    assert '1クリップ' in log and '完了' in log
    assert projects['value'] == job_id


def test_refresh_after_failed_split_shows_log(job):
    import app
    job_id, folder = job
    (folder / 'rows.json').unlink()
    (folder / 'run.log').write_text('Whisper error: no speech', encoding='utf-8')
    rows, clip, audio, log, _ = app.refresh_results(job_id)
    assert rows == [] and clip['value'] is None and audio is None
    assert 'Whisper error' in log


def test_completion_has_separate_result_refresh_event():
    import app
    demo = app.build()
    cfg = demo.get_config_file()
    component_by_id = {item['id']: item for item in cfg['components']}
    start_event = next(d for d in cfg['dependencies'] if d['api_name'] == 'start')
    chained = [d for d in cfg['dependencies'] if d['trigger_after'] == start_event['id']]
    assert len(chained) == 1
    assert chained[0]['queue'] is False
    outputs = [component_by_id[i]['type'] for i in chained[0]['outputs']]
    assert outputs == ['dataframe', 'dropdown', 'audio', 'textbox', 'dropdown']


def test_result_refresh_gradio_serializes_table_and_audio(job):
    import asyncio
    import app
    job_id, folder = job
    engine.save_table(job_id, engine.table(job_id))
    (folder / 'run.log').write_text('分割完了', encoding='utf-8')
    demo = app.build()
    index = next(i for i, fn in demo.fns.items() if fn.fn is app.refresh_results)
    result = asyncio.run(demo.process_api(index, [job_id]))
    data = result['data']
    assert data[0]['data'] == engine.table(job_id)
    assert data[1]['value'] == '000_00000'
    assert data[2]['path'].endswith('.wav')
    assert '1クリップ' in data[3]


def test_initial_saved_selection_and_id_match(job):
    import app
    job_id, _ = job
    cfg = app.build().get_config_file()
    props = [c['props'] for c in cfg['components']]
    saved = next(p for p in props if p.get('label', '').startswith('保存済みプロジェクト'))
    identity = next(p for p in props if p.get('label', '').startswith('プロジェクトID'))
    assert saved['value'] == identity['value'] == job_id


def test_open_uses_selection_even_if_id_empty_or_stale(monkeypatch):
    import app
    monkeypatch.setattr(app, 'load', lambda target: ('loaded', target))
    assert app.load_selected('selected', '') == ('selected', 'loaded', 'selected')
    assert app.load_selected('selected', 'stale') == ('selected', 'loaded', 'selected')
    assert app.load_selected(None, 'manual') == ('manual', 'loaded', 'manual')
    with pytest.raises(Exception, match='選択'):
        app.load_selected(None, '')


def test_no_saved_projects_has_empty_selection_and_id(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(engine, 'JOBS', tmp_path)
    cfg = app.build().get_config_file()
    props = [c['props'] for c in cfg['components']]
    saved = next(p for p in props if p.get('label', '').startswith('保存済みプロジェクト'))
    identity = next(p for p in props if p.get('label', '').startswith('プロジェクトID'))
    assert saved.get('value') is None and identity['value'] == ''
