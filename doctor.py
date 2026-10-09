"""Check imports and the official training CLI without downloading model weights."""

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "irodori"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-check", action="store_true")
    args = parser.parse_args()
    print("Python: " + sys.version.split()[0])
    print("Platform: " + sys.platform)
    errors = []
    for name in [
        "torch",
        "torchaudio",
        "gradio",
        "soundfile",
        "faster_whisper",
        "dacvae",
        "audiotools",
        "irodori_tts",
    ]:
        try:
            importlib.import_module(name)
            print("[OK] " + name, flush=True)
        except Exception as exc:
            errors.append(name + ": " + str(exc))
            print("[ERROR] " + errors[-1], flush=True)
    if errors:
        print("依存パッケージの読み込みに失敗しました。setup.logを確認してください。")
        return 1
    import torch

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(
                f"GPU {index}: {properties.name} / {properties.total_memory / 1024**3:.1f} GiB VRAM"
            )
        print("BF16: " + str(torch.cuda.is_bf16_supported()))
        try:
            x = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
            _ = x @ x
            torch.cuda.synchronize()
            print("[OK] CUDA BF16計算")
        except Exception as exc:
            print("[ERROR] GPU計算失敗: " + str(exc))
            return 1
    else:
        print("[注意] CUDA GPUが使えません。CPUで分割できますが、この環境では学習できません。")
    if args.setup_check:
        print("公式train.pyの起動確認...", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "irodori/train.py"), "--help"],
                cwd=ROOT / "irodori",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print("[ERROR] " + str(exc))
            return 1
        if result.returncode:
            print(result.stdout[-4000:] + result.stderr[-4000:])
            return 1
        print("[OK] 公式train.py --help")
    print("環境確認完了（モデル取得・実学習の完走確認は含みません）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
