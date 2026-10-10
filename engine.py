import codecs
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = Path(os.environ.get("IRODORI_JOBS", ROOT / "projects")).resolve()
ACTIVE = {}
CANCELLED = set()
LOCK = threading.Lock()
WINDOWS = sys.platform == "win32"


def process_options():
    if WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_job(job_id):
    if not job_id or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("プロジェクトIDが不正です。")
    path = JOBS / job_id
    if not path.is_dir():
        raise ValueError("プロジェクトがありません。")
    return path


def project_name(job):
    try:
        return read(Path(job) / "project.json")["name"]
    except (OSError, ValueError, KeyError):
        return ""


def validate_name(name):
    name = name.strip()
    if len(name) > 80 or any(c in '<>:"/\\|?*' or ord(c) < 32 for c in name):
        raise ValueError('名前は80文字以内で、\\ / : * ? " < > | や制御文字は使えません。')
    if name and (name.endswith(".") or name.split(".")[0].upper() in
                 {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                  *[f"LPT{i}" for i in range(1, 10)]}):
        raise ValueError("Windowsで使えないファイル名です。")
    return name


def rename_project(job_id, name):
    job = resolve_job(job_id)
    name = validate_name(name)
    with LOCK:
        if job_id in ACTIVE:
            raise ValueError("処理を中断または終了してから名前を変更してください。")
        record_path = job / "training.json"
        if record_path.exists():
            record = read(record_path)
            output = Path(record["output_dir"])
            old = record.get("output_prefix", "checkpoint")
            new = name or "checkpoint"
            moves = [(p, p.with_name(new + p.name[len(old):]))
                     for p in output.glob("*.speaker.safetensors")
                     if p.name.startswith(old + "_") and old != new]
            if any(target.exists() for _, target in moves):
                raise ValueError("同名の出力があるため名前を変更できません。")
            for source, target in moves:
                source.rename(target)
            record["output_prefix"] = new
            write(record_path, record)
        write(job / "project.json", {"name": name})
    return name


def final_embedding(job_id):
    record = read(resolve_job(job_id) / "training.json")
    return Path(record["output_dir"]) / (record.get("output_prefix", "checkpoint") +
                                         "_final.speaker.safetensors")


def create_job(files, settings):
    import shutil

    if not files:
        raise ValueError("音声ファイルを追加してください。")
    requested_name = validate_name(settings.pop("project_name", ""))
    job_id = uuid.uuid4().hex
    job = JOBS / job_id
    for name in ["input", "sources", "clips", "latents", "output"]:
        (job / name).mkdir(parents=True, exist_ok=True)
    write(job / "project.json", {"name": requested_name})
    sources = []
    for i, source in enumerate(files):
        source = Path(source)
        target = job / "input" / (f"{i:03d}_" + source.name)
        shutil.copy2(source, target)
        sources.append(str(target))
    settings["sources"] = sources
    write(job / "settings.json", settings)
    return job_id


def tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 16000))
        return f.read().decode("utf-8", errors="replace")


STAGE_LABELS = {
    "split": "文字起こし・音声分割",
    "encode": "DACVAE音声変換",
    "train": "Speaker Inversion学習",
    "resume": "Speaker Inversion再開",
}


class ConsoleLog:
    """Forward only new bytes, preserving UTF-8 across partial file writes."""

    def __init__(self, path, offset):
        self.reader = Path(path).open("rb")
        self.reader.seek(offset)
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def forward(self, final=False):
        changed = False
        while data := self.reader.read(65536):
            changed = True
            value = self.decoder.decode(data)
            if value:
                print(value, end="", flush=True)
        if final:
            value = self.decoder.decode(b"", final=True)
            if value:
                print(value, end="", flush=True)
        return changed

    def close(self):
        self.reader.close()


def stage_message(output, job_id, message):
    output.write(f"\n[{time.strftime('%H:%M:%S')}] [{job_id[:8]}] {message}\n")
    output.flush()


def run_stage(job_id, stage):
    job = resolve_job(job_id)
    log = job / "run.log"
    with LOCK:
        if job_id in CANCELLED:
            raise RuntimeError("処理を停止しました。")
        if ACTIVE:
            raise RuntimeError("別の処理が実行中です。完了または停止してから実行してください。")
        if stage in ("train", "resume"):
            (job / "pause.request").unlink(missing_ok=True)
            (job / "pause.request.complete").unlink(missing_ok=True)
        state = {"process": None, "cancelled": False, "stage": stage}
        ACTIVE[job_id] = state
    label = STAGE_LABELS.get(stage, stage)
    started = time.monotonic()
    console = None
    try:
        write(job / "status.json", {"stage": stage, "state": "running"})
        with log.open("a", encoding="utf-8") as output:
            offset = output.tell()
            stage_message(output, job_id, label + " 開始 / ログ: " + str(log))
            console = ConsoleLog(log, offset)
            console.forward()
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "worker.py"), stage, str(job)],
                stdout=output,
                stderr=subprocess.STDOUT,
                **process_options(),
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            )
            with LOCK:
                state["process"] = process
                cancelled = state["cancelled"]
            if cancelled:
                terminate(process)
            last_activity = time.monotonic()
            while process.poll() is None:
                now = time.monotonic()
                if console.forward():
                    last_activity = now
                elif now - last_activity >= 15:
                    elapsed = int(now - started)
                    stage_message(
                        output, job_id, f"{label} 実行中（経過 {elapsed // 60}分{elapsed % 60}秒）"
                    )
                    console.forward()
                    last_activity = now
                yield tail(log)
                time.sleep(0.6)
            console.forward()
            if state["cancelled"]:
                stage_message(output, job_id, label + " 停止")
                console.forward(final=True)
                write(job / "status.json", {"stage": stage, "state": "cancelled"})
                raise RuntimeError(
                    "処理を停止しました。保存済みの素材とチェックポイントは残っています。"
                )
            if process.returncode:
                stage_message(output, job_id, f"{label} 失敗（終了コード {process.returncode}）")
                console.forward(final=True)
                write(job / "status.json", {"stage": stage, "state": "failed"})
                raise RuntimeError(
                    "処理に失敗しました。ログ末尾をご確認ください。\n" + tail(log)[-4000:]
                )
            if (job / "pause.request.complete").exists() and stage in ("train", "resume"):
                stage_message(
                    output, job_id, "学習状態を保存して中断しました。PCを再起動しても再開できます。"
                )
                console.forward(final=True)
                write(job / "status.json", {"stage": stage, "state": "paused"})
                yield tail(log)
                return
            stage_message(
                output, job_id, label + f" 完了（経過 {int(time.monotonic() - started)}秒）"
            )
            console.forward(final=True)
        write(job / "status.json", {"stage": stage, "state": "completed"})
        yield tail(log)
    finally:
        if console is not None:
            console.close()
        with LOCK:
            process = state["process"]
            if process is not None and process.poll() is None:
                terminate(process)
            ACTIVE.pop(job_id, None)


def terminate(process):
    if WINDOWS:
        # Keep the worker alive while training so taskkill can stop its entire tree.
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
                check=False,
            )
            if result.returncode and process.poll() is None:
                process.kill()
        except (OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    def force():
        try:
            # Also kill remaining subprocesses after parent exits.
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(5, force)
    timer.daemon = True
    timer.start()


def cancel(job_id):
    with LOCK:
        CANCELLED.add(job_id)
        state = ACTIVE.get(job_id)
        if not state:
            return "実行中の処理はありません。"
        state["cancelled"] = True
        if state["process"]:
            terminate(state["process"])
    return "停止を要求しました。"


def pause(job_id):
    job = resolve_job(job_id)
    with LOCK:
        active = ACTIVE.get(job_id)
        if not active or active["stage"] not in ("train", "resume"):
            return "保存して中断は、学習実行中に使用できます。"
        (job / "pause.request").write_text("pause", encoding="utf-8")
    return "保存して中断を要求しました。現在の学習ステップ完了と保存完了をお待ちください。"


def is_paused(job_id):
    return (resolve_job(job_id) / "pause.request.complete").exists()


def project_choices():
    if not JOBS.exists():
        return []
    result = []
    for job in sorted(JOBS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not job.is_dir() or not (job / "rows.json").exists():
            continue
        description = (project_name(job) or "名称未設定") + " / " + job.name[:8]
        try:
            training = read(job / "training.json")
            info = read(Path(training["output_dir"]) / "resume_info.json")
            description += f" / 再開状態 {info['step']}/{info['target_steps']} step"
        except (OSError, ValueError, KeyError):
            pass
        result.append((description, job.name))
    return result


HEADERS = ["使用", "ID", "開始秒", "終了秒", "長さ秒", "文字起こし", "確認メモ"]


def table(job_id):
    return [
        [r["use"], r["id"], r["start"], r["end"], round(r["duration"], 3), r["text"], r["note"]]
        for r in read(resolve_job(job_id) / "rows.json")
    ]


def _prepare_table(job_id, values):
    import soundfile as sf

    job = resolve_job(job_id)
    originals = {r["id"]: r for r in read(job / "rows.json")}
    rows, seen, cache = [], set(), {}
    for value in values:
        if len(value) != 7:
            raise ValueError("表の列数を変更しないでください。")
        use, clip_id, start, end, _, text, note = value
        if clip_id not in originals or clip_id in seen:
            raise ValueError(
                "IDの変更・重複や空行の追加はできません。行を追加して分割する場合は"
                "「長い素材を2つに分割」を使用してください。除外する場合は「使用」をOFFにしてください。"
            )
        seen.add(clip_id)
        row = dict(originals[clip_id])
        start, end = float(start), float(end)
        text = str(text).strip()
        if row["source"] not in cache:
            cache[row["source"]] = sf.read(
                job / "sources" / f"{row['source']:03d}.wav", dtype="float32"
            )
        audio, sr = cache[row["source"]]
        if not (
            math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= len(audio) / sr
        ):
            raise ValueError("開始・終了秒が音声の範囲外です: " + clip_id)
        use = use is True or str(use).lower() in ("true", "1")
        if use and not text:
            raise ValueError("使用する素材の文字起こしを空にはできません: " + clip_id)
        first, last = int(start * sr), int(end * sr)
        if last <= first:
            raise ValueError("音声が短すぎます: " + clip_id)
        row.update(
            use=use,
            start=first / sr,
            end=last / sr,
            duration=(last - first) / sr,
            text=text,
            note=str(note),
        )
        rows.append(row)
    if seen != set(originals):
        raise ValueError("行を削除せず、使用をOFFにして除外してください。")
    return job, rows, cache


def _commit_rows(job, rows, cache):
    import tempfile

    import soundfile as sf

    # Validate all rows before replacing any clip.
    with tempfile.TemporaryDirectory(dir=job / "clips") as staging:
        for row in rows:
            audio, sr = cache[row["source"]]
            sf.write(
                Path(staging) / (row["id"] + ".wav"),
                audio[round(row["start"] * sr) : round(row["end"] * sr)],
                sr,
                subtype="PCM_24",
            )
        for row in rows:
            (Path(staging) / (row["id"] + ".wav")).replace(
                job / "clips" / (row["id"] + ".wav")
            )
    write(job / "rows.json", rows)


def save_table(job_id, values):
    job, rows, cache = _prepare_table(job_id, values)
    _commit_rows(job, rows, cache)
    return table(job_id)


def split_clip(job_id, values, clip_id, split_seconds, left_text, right_text):
    """Split edited material at one sample boundary, preserving its source."""
    with LOCK:
        if job_id in ACTIVE:
            raise ValueError("処理を中断または終了してから素材を分割してください。")
        job, rows, cache = _prepare_table(job_id, values)
        index = next((i for i, row in enumerate(rows) if row["id"] == clip_id), None)
        if index is None:
            raise ValueError("分割するクリップを選択してください。")
        left_text, right_text = str(left_text or "").strip(), str(right_text or "").strip()
        if not left_text or not right_text:
            raise ValueError("分割後の前半・後半の文字起こしを両方入力してください。")
        row = rows[index]
        _, sr = cache[row["source"]]
        split_seconds = float(split_seconds)
        if not math.isfinite(split_seconds):
            raise ValueError("分割位置には有効な秒数を入力してください。")
        cut = int(split_seconds * sr)
        first, last = round(row["start"] * sr), round(row["end"] * sr)
        if not first < cut < last:
            raise ValueError("分割位置は対象素材の開始秒と終了秒の間にしてください。")
        boundary = cut / sr
        new_id = f'{row["source"]:03d}_{uuid.uuid4().hex}'
        while any(item["id"] == new_id for item in rows):
            new_id = f'{row["source"]:03d}_{uuid.uuid4().hex}'
        note = (row["note"] + " / 手動分割").strip(" /")
        left = dict(row, end=boundary, duration=boundary-row["start"], text=left_text, note=note)
        right = dict(row, id=new_id, start=boundary, duration=row["end"]-boundary,
                     text=right_text, note=note)
        rows[index:index + 1] = [left, right]
        _commit_rows(job, rows, cache)
        return table(job_id), new_id


def bundle(job_id):
    import zipfile

    job = resolve_job(job_id)
    target = job / "result.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name in [
            "project.json",
            "rows.json",
            "warnings.json",
            "manifest.jsonl",
            "train_large_speaker.yaml",
            "provenance.json",
            "training.json",
            "run.log",
            "settings.json",
        ]:
            path = job / name
            if path.exists():
                z.write(path, name)
        for folder in ["clips", "latents", "output", "asr"]:
            for path in (job / folder).rglob("*"):
                if path.is_file():
                    z.write(path, str(path.relative_to(job)))
    return str(target)
