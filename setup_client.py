"""Set up an isolated CUDA environment on Windows and Linux."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def environment_python(upstream, windows=None):
    if windows is None:
        windows = sys.platform == "win32"
    return upstream / ".venv" / ("Scripts/python.exe" if windows else "bin/python")


def find_uv():
    executable = shutil.which("uv")
    if executable:
        return [executable]
    if importlib.util.find_spec("uv"):
        return [sys.executable, "-m", "uv"]
    raise RuntimeError("uvがありません。python -m pip install uv を実行してください。")


def setup_commands(uv, root, windows=None):
    upstream = root / "irodori"
    python = environment_python(upstream, windows)
    return [
        (uv + ["sync", "--frozen", "--extra", "cu128", "--python", "3.11"], upstream),
        (
            uv
            + [
                "pip",
                "install",
                "--python",
                str(python),
                "-r",
                str(root / "requirements-client.txt"),
            ],
            root,
        ),
        ([str(python), "-u", str(root / "doctor.py"), "--setup-check"], root),
    ]


def run_logged(command, cwd, log):
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    with subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    if not shutil.which("git"):
        print(
            "Gitが見つかりません。Git for Windowsをインストールして、コマンド画面を開き直してください。"
        )
        print("https://git-scm.com/downloads/win")
        return 1
    try:
        uv = find_uv()
        with (ROOT / "setup.log").open("w", encoding="utf-8") as log:
            for command, cwd in setup_commands(uv, ROOT):
                message = "\n実行: " + subprocess.list2cmdline(command)
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                run_logged(command, cwd, log)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print("\nセットアップに失敗しました。上のエラーとsetup.logをご確認ください。\n" + str(exc))
        return 1
    print(
        "\nインストール完了。"
        + ("start.bat" if sys.platform == "win32" else "bash start.sh")
        + "で起動してください。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
