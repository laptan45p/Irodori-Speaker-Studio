import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import file_picker


@pytest.mark.parametrize('kind,suffix', [('model','.safetensors'),('embedding','.speaker.safetensors')])
def test_windows_picker_returns_original_path_without_copy(tmp_path, monkeypatch,kind,suffix):
    path=tmp_path / ('日本語 空白 $テスト'+suffix)
    path.write_bytes(b'original')
    calls=[]
    monkeypatch.setattr(file_picker.sys,'platform','win32')
    def run(command,**kwargs):
        calls.append((command,kwargs))
        return SimpleNamespace(returncode=0,stdout=json.dumps({'path':str(path)},ensure_ascii=False))
    monkeypatch.setattr(file_picker.subprocess,'run',run)
    assert file_picker.choose_file(kind,str(path)) == str(path.resolve())
    command,kwargs=calls[0]
    assert command[:3] == ['powershell.exe','-NoProfile','-STA']
    assert str(path) not in ' '.join(command)
    assert json.loads(kwargs['input'])['current'] == str(path)
    assert kwargs['encoding'] == 'utf-8'
    assert list(tmp_path.iterdir()) == [path] and path.read_bytes() == b'original'


def test_picker_cancel_keeps_text_and_dialog_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(file_picker.sys,'platform','win32')
    monkeypatch.setattr(file_picker.subprocess,'run',lambda *a,**kw: SimpleNamespace(returncode=0,stdout='{"path":null}'))
    assert app.pick_model('existing-path') == app.gr.skip()
    monkeypatch.setattr(file_picker.subprocess,'run',lambda *a,**kw: SimpleNamespace(returncode=1,stdout=''))
    with pytest.raises(ValueError,match='直接入力'):
        file_picker.choose_file('model')


def test_picker_rejects_wrong_type_or_missing_file(tmp_path,monkeypatch):
    monkeypatch.setattr(file_picker.sys,'platform','win32')
    path=tmp_path/'wrong.txt'
    path.write_text('wrong')
    for chosen in [path,tmp_path/'missing.safetensors']:
        monkeypatch.setattr(file_picker.subprocess,'run',lambda *a,**kw: SimpleNamespace(returncode=0,stdout=json.dumps({'path':str(chosen)})))
        with pytest.raises(ValueError,match='保存先'):
            file_picker.choose_file('model')


@pytest.mark.parametrize('handler,kind', [(app.pick_model,'model'),(app.pick_embedding,'embedding')])
def test_gradio_picker_updates_the_matching_textbox(monkeypatch,handler,kind):
    calls=[]
    monkeypatch.setattr(file_picker,'choose_file',lambda k,current: calls.append((k,current)) or 'F:/選択/model.safetensors')
    demo=app.build()
    index=next(i for i,fn in demo.fns.items() if fn.fn is handler)
    assert demo.fns[index].inputs[0]._id == demo.fns[index].outputs[0]._id
    result=asyncio.run(demo.process_api(index,['old-path']))
    assert result['data'][0] == 'F:/選択/model.safetensors'
    assert calls == [(kind,'old-path')]
