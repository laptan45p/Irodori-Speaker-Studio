import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine


@pytest.fixture
def material(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "JOBS", tmp_path)
    rate = 48000
    original = tmp_path / "input.wav"
    sf.write(original, np.linspace(-.5, .5, rate * 10), rate, subtype="PCM_24")
    identity = engine.create_job([original], {})
    job = engine.resolve_job(identity)
    audio, _ = sf.read(original)
    sf.write(job / "sources/000.wav", audio, rate, subtype="PCM_24")
    rows = [dict(use=True, id="000_00000", source=0, start=1, end=5,
                 duration=4, text="前半です。後半です。", note="", confidence=.9),
            dict(use=True, id="000_00001", source=0, start=6, end=9,
                 duration=3, text="別の素材。", note="", confidence=.9)]
    engine.write(job / "rows.json", rows)
    engine.save_table(identity, engine.table(identity))
    return identity, job, audio, rate


def test_split_preserves_samples_other_edits_and_reload_export(material):
    identity, job, source, rate = material
    values = engine.table(identity)
    values[1][5] = "別の素材も修正。"
    result, new_id = engine.split_clip(identity, values, "000_00000", 3.0001,
                                       "前半です。", "後半です。")
    assert len(result) == 3 and result[0][1] == "000_00000"
    assert result[1][1] == new_id and new_id != result[2][1]
    assert result[2][5] == "別の素材も修正。"
    left, _ = sf.read(job / "clips/000_00000.wav")
    right, _ = sf.read(job / "clips" / (new_id + ".wav"))
    assert np.array_equal(np.concatenate([left, right]), source[rate:5 * rate])
    assert result[0][3] == result[1][2]
    assert len(left) == int(3.0001 * rate) - rate
    assert engine.table(identity) == result
    assert engine.save_table(identity, result) == result
    with zipfile.ZipFile(engine.bundle(identity)) as z:
        assert "clips/" + new_id + ".wav" in z.namelist()
        assert len(json.loads(z.read("rows.json"))) == 3


@pytest.mark.parametrize("cut,left,right", [
    (1, "前", "後"), (5, "前", "後"), (float("nan"), "前", "後"),
    (float("inf"), "前", "後"), (3, "", "後"), (3, "前", " "),
])
def test_invalid_split_leaves_saved_material_unchanged(material, cut, left, right):
    identity, job, _, _ = material
    before = {p.relative_to(job): p.read_bytes() for p in job.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        engine.split_clip(identity, engine.table(identity), "000_00000", cut, left, right)
    after = {p.relative_to(job): p.read_bytes() for p in job.rglob("*") if p.is_file()}
    assert after == before


def test_split_rejects_active_work_and_unknown_selection(material, monkeypatch):
    identity, _, _, _ = material
    with pytest.raises(ValueError, match="選択"):
        engine.split_clip(identity, engine.table(identity), "missing", 3, "前", "後")
    monkeypatch.setitem(engine.ACTIVE, identity, {"stage": "encode"})
    with pytest.raises(ValueError, match="中断"):
        engine.split_clip(identity, engine.table(identity), "000_00000", 3, "前", "後")


def test_ui_split_serialization_and_poll_preserve_second_half(material):
    import asyncio

    from gradio.state_holder import SessionState

    import app
    identity, _, _, _ = material
    demo = app.build()
    index = next(i for i, fn in demo.fns.items() if fn.fn is app.split_material)
    state = SessionState(demo)
    refresh = next(i for i, fn in demo.fns.items() if fn.fn is app.refresh_results)
    asyncio.run(demo.process_api(refresh, [identity], state=state))
    inputs = [identity, {"headers": engine.HEADERS, "data": engine.table(identity)},
              "000_00000", 3, "前半", "後半"]
    response = asyncio.run(demo.process_api(index, inputs, state=state))
    assert len(response["data"][0]["data"]) == 3
    selected = response["data"][1]["value"]
    assert selected != "000_00000"
    assert response["data"][2]["path"].endswith(selected + ".wav")
    signature = app.poll_results(identity, {})[-1]
    assert app.poll_results(identity, signature)[1] == app.gr.skip()


def test_save_preserves_selected_clip_and_refreshes_audio(material):
    import app
    identity, job, _, rate = material
    old_signature = app.poll_results(identity, {})[-1]
    rows = engine.table(identity)
    rows[1][2:4] = [7, 8.5]
    result = app.save_selected(identity, rows, "000_00001")
    assert result[2]["value"] == "000_00001"
    assert result[3] == str(job / "clips/000_00001.wav")
    audio, actual_rate = sf.read(result[3])
    assert actual_rate == rate and len(audio) == int(1.5 * rate)
    # A timer request using the pre-save signature still keeps the same choice.
    poll = app.poll_results(identity, old_signature, "000_00001")
    assert poll[1]["value"] == "000_00001"
    assert app.poll_results(identity, result[-1], "000_00001")[1] == app.gr.skip()


def test_save_selection_gradio_response(material):
    import asyncio

    from gradio.state_holder import SessionState

    import app
    identity, _, _, _ = material
    demo = app.build()
    state = SessionState(demo)
    refresh = next(i for i, fn in demo.fns.items() if fn.fn is app.refresh_results)
    asyncio.run(demo.process_api(refresh, [identity], state=state))
    index = next(i for i, fn in demo.fns.items() if fn.fn is app.save_selected)
    rows = engine.table(identity)
    rows[1][2:4] = [7, 8]
    result = asyncio.run(demo.process_api(index, [identity,
        {"headers": engine.HEADERS, "data": rows}, "000_00001"], state=state))
    assert result["data"][2]["value"] == "000_00001"
    samples, rate = sf.read(result["data"][3]["path"])
    assert len(samples) == rate
