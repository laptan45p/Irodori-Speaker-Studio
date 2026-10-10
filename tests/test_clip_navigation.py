import asyncio
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_manual_split import material as _material_fixture

import app
import engine

material = _material_fixture


def test_navigation_preserves_unsaved_edits_and_unchecked_clips(material):
    identity, job, _, _ = material
    rows = engine.table(identity)
    rows[1][0] = False
    rows[1][2] = 6.2
    rows[1][5] = '未保存の修正'
    before = copy.deepcopy(rows)
    saved = (job / 'rows.json').read_bytes()
    result = app.next_clip(identity, rows[0][1], rows, '素材周辺')
    assert result[0]['value'] == rows[1][1]
    assert result[1].endswith(rows[1][1] + '.wav')
    assert result[3]['clip_id'] == rows[1][1]
    assert rows == before and (job / 'rows.json').read_bytes() == saved
    assert app.previous_clip(identity, rows[1][1], rows, '素材周辺')[0]['value'] == rows[0][1]


def test_navigation_clamps_edges_and_handles_stale_selection(material):
    identity, job, _, _ = material
    rows = engine.table(identity)
    assert app.previous_clip(identity, rows[0][1], rows, '素材周辺')[0]['value'] == rows[0][1]
    assert app.next_clip(identity, rows[-1][1], rows, '素材周辺')[0]['value'] == rows[-1][1]
    assert app.next_clip(identity, 'stale-id', rows, '素材周辺')[0]['value'] == rows[0][1]
    (job / 'rows.json').unlink()
    assert app.next_clip(identity, 'stale-id', [], '素材周辺')[0]['value'] is None
    assert app.previous_clip('', None, [], '素材周辺')[0]['value'] is None


def test_navigation_gradio_updates_audio_waveform_and_selection(material):
    identity, _, _, _ = material
    rows = engine.table(identity)
    demo = app.build()
    index = next(i for i, fn in demo.fns.items() if fn.fn is app.next_clip)
    from gradio.state_holder import SessionState
    state = SessionState(demo)
    result = asyncio.run(demo.process_api(index, [identity, rows[0][1], {'headers': engine.HEADERS, 'data': rows, 'metadata': None}, '素材周辺'], state=state))
    assert result['data'][0]['value'] == rows[1][1]
    assert result['data'][1] and result['data'][2] and result['data'][4]
    assert state[demo.fns[index].outputs[3]._id]['clip_id'] == rows[1][1]
