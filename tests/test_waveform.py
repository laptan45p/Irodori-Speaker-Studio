import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_manual_split import material as _material_fixture

from waveform import draw_waveform, time_at_pixel

material = _material_fixture


def test_source_and_zoom_coordinates_have_correct_seconds():
    audio = np.zeros(48000 * 60)
    full, info = draw_waveform(audio, 48000, 20, 40, "元音声全体")
    assert full.size == (1200, 240)
    assert time_at_pixel(info, 24) == 0
    assert time_at_pixel(info, 600) == 30
    assert time_at_pixel(info, 1176) == 60
    assert time_at_pixel(info, 0) is None
    _, zoom = draw_waveform(audio, 48000, 20, 40, "素材周辺")
    assert time_at_pixel(zoom, 24) == 18
    assert time_at_pixel(zoom, 1176) == 42
    assert time_at_pixel(zoom, 600) == 30


def test_wave_click_edits_table_then_save_rebuilds_audio(material):
    import soundfile as sf

    import app
    import engine
    import gradio as gr
    identity, job, source, rate = material
    rows = engine.table(identity)
    _, info, source_file = app.render_waveform(identity, "000_00000", rows, "元音声全体")
    assert Path(source_file) == job / "sources/000.wav"
    before = (job / "clips/000_00000.wav").read_bytes()
    event = gr.SelectData(None, {"index": [254.4, 100], "value": None})
    result = app.choose_wave_position(identity, "000_00000", rows, "開始位置", info, event)
    assert result[0][0][2] == pytest.approx(2, abs=1/rate)
    assert engine.table(identity)[0][2] == 1
    assert (job / "clips/000_00000.wav").read_bytes() == before
    engine.save_table(identity, result[0])
    samples, _ = sf.read(job / "clips/000_00000.wav")
    first = round(result[0][0][2] * rate)
    assert np.array_equal(samples, source[first:5*rate])


def test_wave_click_split_position_and_stale_selection(material):
    import app
    import engine
    import gradio as gr
    identity, _, _, _ = material
    rows = engine.table(identity)
    _, info, _ = app.render_waveform(identity, "000_00000", rows, "元音声全体")
    event = gr.SelectData(None, {"index": [369.6, 100], "value": None})
    result = app.choose_wave_position(identity, "000_00000", rows, "分割位置", info, event)
    assert result[0] == gr.skip() and result[1] == pytest.approx(3, abs=1/48000)
    assert len(engine.table(identity)) == 2
    with pytest.raises(Exception, match="波形を更新"):
        app.choose_wave_position(identity, "000_00001", rows, "分割位置", info, event)


def test_waveform_gradio_response_serialization(material):
    import asyncio

    from gradio.state_holder import SessionState

    import app
    import engine
    identity, _, source, rate = material
    demo = app.build()
    index = next(i for i, fn in demo.fns.items() if fn.fn is app.render_waveform)
    state = SessionState(demo)
    refresh = next(i for i, fn in demo.fns.items() if fn.fn is app.refresh_results)
    asyncio.run(demo.process_api(refresh, [identity], state=state))
    payload = {"headers": engine.HEADERS, "data": engine.table(identity)}
    response = asyncio.run(demo.process_api(index, [identity, "000_00000", payload, "素材周辺"], state=state))
    assert response["data"][0]["path"].endswith(".webp")
    import soundfile as sf
    cached, cached_rate = sf.read(response["data"][2]["path"])
    assert cached_rate == rate and np.array_equal(cached, source)
