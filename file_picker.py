"""Select an existing local file without uploading or copying its contents."""
import json
import subprocess
import sys
from pathlib import Path

# Fixed script; paths are supplied as JSON on stdin, never interpolated as code.
WINDOWS_DIALOG = r'''
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$selected = $null
try {
    $dialog.CheckFileExists = $true
    $dialog.Multiselect = $false
    $dialog.RestoreDirectory = $true
    if ($request.kind -eq 'embedding') {
        $dialog.Title = '追加学習する話者埋め込みを選択'
        $dialog.Filter = '話者埋め込み (*.speaker.safetensors)|*.speaker.safetensors'
    } else {
        $dialog.Title = '学習モデルのmodel.safetensorsを選択'
        $dialog.Filter = 'モデル (*.safetensors)|*.safetensors'
    }
    $dialog.InitialDirectory = $request.initial_directory
    if ($request.current -and (Test-Path -LiteralPath $request.current -PathType Leaf)) {
        $existing = Get-Item -LiteralPath $request.current
        $dialog.InitialDirectory = $existing.DirectoryName
        $dialog.FileName = $existing.Name
    }
    $owner.Show()
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        $selected = $dialog.FileName
    }
} finally {
    $dialog.Dispose()
    $owner.Dispose()
}
[Console]::WriteLine((@{path=$selected} | ConvertTo-Json -Compress))
'''


def choose_file(kind, current=""):
    if kind not in {"model", "embedding"}:
        raise ValueError("ファイル選択の種類が不正です。")
    if sys.platform != "win32":
        raise ValueError("ファイル選択ダイアログはWindowsのローカル起動で利用できます。パスを直接入力してください。")
    import base64
    root = Path(__file__).resolve().parent
    initial = root / "projects" if kind == "embedding" else root
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-EncodedCommand",
         base64.b64encode(WINDOWS_DIALOG.encode("utf-16le")).decode("ascii")],
        input=json.dumps({"kind": kind, "current": current or "",
                          "initial_directory": str(initial)}, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise ValueError("Windowsのファイル選択ダイアログを開けませんでした。パスを直接入力してください。")
    selected = json.loads(result.stdout.lstrip("\ufeff"))["path"]
    if not selected:
        return None
    suffix = ".speaker.safetensors" if kind == "embedding" else ".safetensors"
    if not isinstance(selected, str) or not selected.lower().endswith(suffix) or not Path(selected).is_file():
        raise ValueError("選択したファイルの種類または保存先を確認してください。")
    return str(Path(selected).resolve())
