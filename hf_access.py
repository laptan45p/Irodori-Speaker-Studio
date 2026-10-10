"""Read-only startup access check; never print or store credentials."""
import sys
from pathlib import Path

REPO = "google/t5gemma-2-1b-1b"
REVISION = "dd0a2683227859151b1730ca3a63087df5b5f39b"


def access_failure_help(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if type(error).__name__ == "GatedRepoError":
            return "Hugging Faceのモデルへのアクセスが拒否されました。認証と利用条件への同意を確認してください。\n\n" + login_help()
        error = error.__cause__ or error.__context__
    return None


def login_help():
    root = Path(__file__).resolve().parent
    if sys.platform == "win32":
        command = f'cd /d "{root}"\nirodori\\.venv\\Scripts\\hf.exe auth login'
    else:
        command = f'"{root / "irodori/.venv/bin/hf"}" auth login'
    return (
        f"[モデルの利用条件・アクセス権を確認](https://huggingface.co/{REPO}) → "
        "同じアカウントで、このPCのターミナルからログインしてください。\n\n"
        f"```text\n{command}\n```\n\n"
        "トークンを求められた場合は、[Hugging Faceの設定](https://huggingface.co/settings/tokens)"
        "で読み取り権限のトークンを作成して入力してください。ログイン後に「認証状態を再確認」を押します。"
    )


def check_access():
    try:
        from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
        token = get_token()
        if not token:
            status = "⚠️ Hugging Face：未ログインです。学習に必要なモデルの取得には認証が必要です。"
        else:
            try:
                get_hf_file_metadata(hf_hub_url(REPO, "config.json", revision=REVISION),
                                     token=token, timeout=5)
                status = "✅ Hugging Face：認証済み。v4 Largeが使用するT5Gemmaへのアクセスを確認しました。"
            except Exception as error:
                code = getattr(getattr(error, "response", None), "status_code", None)
                if code in (401, 403):
                    status = "⚠️ Hugging Face：保存された認証情報ではモデルにアクセスできません。利用条件・アクセス権とログイン先のアカウントを確認してください。"
                else:
                    status = "⚠️ Hugging Face：認証情報はありますが、通信エラーなどによりモデルへのアクセスを確認できませんでした。接続を確認して再確認してください。"
    except Exception:
        status = "⚠️ Hugging Face：認証状態を確認できませんでした。セットアップと接続を確認してください。"
    print(status, flush=True)
    if status.startswith("✅"):
        return status
    return status + "\n\n" + login_help() + "\n\n音声の分割・素材の修正は引き続き利用できます。モデルがキャッシュ済みの場合は学習できることもあります。"
