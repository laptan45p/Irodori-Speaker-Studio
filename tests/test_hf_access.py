import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hf_access


@pytest.mark.parametrize("mode", ["missing", "ok", "denied", "offline"])
def test_startup_auth_status_and_secret_redaction(mode, monkeypatch, capsys):
    calls = []

    def metadata(url, **kwargs):
        calls.append(kwargs)
        if mode == "denied":
            error = RuntimeError("secret-token must never be printed")
            error.response = SimpleNamespace(status_code=401)
            raise error
        if mode == "offline":
            raise RuntimeError("secret-token network failure")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        get_token=lambda: None if mode == "missing" else "secret-token",
        hf_hub_url=lambda *args, **kwargs: "model-url",
        get_hf_file_metadata=metadata,
    ))
    result = hf_access.check_access()
    expected = {"missing": "未ログイン", "ok": "アクセスを確認", "denied": "アクセスできません", "offline": "通信エラー"}
    assert expected[mode] in result
    assert "secret-token" not in result + capsys.readouterr().out
    assert ("auth login" in result) == (mode != "ok")
    assert len(calls) == (mode != "missing")
    if calls:
        assert calls[0]["timeout"] == 5


def test_training_gated_error_help_preserves_other_errors():
    GatedRepoError = type("GatedRepoError", (Exception,), {})
    outer = OSError("wrapped error")
    outer.__cause__ = GatedRepoError("secret-token")
    result = hf_access.access_failure_help(outer)
    assert "auth login" in result and "secret-token" not in result
    assert hf_access.access_failure_help(ValueError("bad model config")) is None
