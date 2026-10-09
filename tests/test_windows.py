import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine
import setup_client
import worker


def test_windows_venv_paths_and_commands():
    root = Path("/folder with spaces/client")
    assert str(setup_client.environment_python(root / "irodori", True)).endswith(
        "Scripts/python.exe"
    )
    commands = setup_client.setup_commands(["uv.exe"], root, True)
    assert commands[0][0] == ["uv.exe", "sync", "--frozen", "--extra", "cu128", "--python", "3.11"]
    assert "Scripts/python.exe" in commands[1][0][4]
    assert commands[2][0][-1] == "--setup-check"
    assert commands[0][1] == root / "irodori"


def test_windows_spawn_and_stop_tree(monkeypatch):
    monkeypatch.setattr(engine, "WINDOWS", True)
    monkeypatch.setattr(engine.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(engine.subprocess, "CREATE_NO_WINDOW", 0x8000000, raising=False)
    assert engine.process_options() == {"creationflags": 0x8000200}
    calls = []
    monkeypatch.setattr(
        engine.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or types.SimpleNamespace(returncode=0),
    )
    process = types.SimpleNamespace(
        pid=1234, poll=lambda: None, kill=lambda: pytest.fail("fallback not expected")
    )
    engine.terminate(process)
    assert calls[0][0] == ["taskkill", "/PID", "1234", "/T", "/F"]
    assert calls[0][1]["timeout"] == 10


def test_windows_stop_falls_back_to_kill(monkeypatch):
    monkeypatch.setattr(engine, "WINDOWS", True)
    monkeypatch.setattr(engine.subprocess, "CREATE_NO_WINDOW", 0x8000000, raising=False)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("taskkill", 10)

    monkeypatch.setattr(engine.subprocess, "run", timeout)
    killed = []
    engine.terminate(
        types.SimpleNamespace(pid=1234, poll=lambda: None, kill=lambda: killed.append(True))
    )
    assert killed == [True]


def test_windows_trainer_keeps_worker_parent_and_exit_code(monkeypatch):
    monkeypatch.setattr(
        worker, "sys", types.SimpleNamespace(platform="win32", executable="python.exe")
    )
    calls = []
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or types.SimpleNamespace(returncode=7),
    )
    with pytest.raises(SystemExit) as result:
        worker.launch_trainer(
            ["python.exe", "train.py", "--manifest", "G:\\voice data\\data.jsonl"]
        )
    assert result.value.code == 7
    assert calls[0][1]["cwd"] == worker.UPSTREAM
    assert calls[0][0][-1] == "G:\\voice data\\data.jsonl"


def test_setup_logs_utf8_and_propagates_errors(tmp_path):
    path = tmp_path / "setup.log"
    with path.open("w", encoding="utf-8") as log:
        setup_client.run_logged(
            [sys.executable, "-c", 'print("日本語ログ"); raise SystemExit(0)'], tmp_path, log
        )
    assert "日本語ログ" in path.read_text(encoding="utf-8")
    with path.open("a", encoding="utf-8") as log, pytest.raises(subprocess.CalledProcessError):
        setup_client.run_logged([sys.executable, "-c", "raise SystemExit(5)"], tmp_path, log)


def test_batch_launchers_are_crlf_ascii_and_quote_paths():
    for name in ["setup.bat", "start.bat", "check_environment.bat"]:
        data = (ROOT / name).read_bytes()
        assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")
        assert b'cd /d "%~dp0"' in data
        assert b"PYTHONUTF8=1" in data
        assert not any(byte > 127 for byte in data)
    assert b"--open-browser" in (ROOT / "start.bat").read_bytes()
    assert b" >setup.log" not in (ROOT / "setup.bat").read_bytes()
