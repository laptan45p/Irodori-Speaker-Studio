import argparse
import os

import engine
import gradio as gr
from hf_access import check_access


def settings(
    minimum,
    maximum,
    asr_model,
    asr_device,
    language,
    caption,
    steps,
    lr,
    batch,
    accumulation,
    checkpoint,
    embedding,
    project_name="",
):
    if not 0 < minimum <= maximum:
        raise gr.Error("秒数の目安は 0 < 最小 <= 最大 にしてください。")
    if steps < 1 or lr <= 0 or batch < 1 or accumulation < 1:
        raise gr.Error("学習設定は正の値にしてください。")
    return dict(
        project_name=engine.validate_name(project_name),
        minimum=float(minimum),
        maximum=float(maximum),
        asr_model=asr_model,
        asr_device=asr_device,
        language=language.strip(),
        caption=caption.strip(),
        steps=int(steps),
        lr=float(lr),
        batch_size=int(batch),
        accumulation=int(accumulation),
        checkpoint=checkpoint.strip(),
        initial_embedding=embedding.strip(),
    )


def prepare_start(files, *args):
    """Commit the project identity in a short response before heavy work."""
    job_id = engine.create_job(files, settings(*args))
    engine.CANCELLED.discard(job_id)
    return job_id, "処理を開始します。", [], gr.update(choices=[], value=None), None


def start(files, mode, *args):
    initial = prepare_start(files, *args)
    yield initial
    yield from run_prepared(initial[0], mode)


def run_prepared(job_id, mode):
    try:
        for log in engine.run_stage(job_id, "split"):
            yield job_id, log, gr.skip(), gr.skip(), gr.skip()
        rows = engine.table(job_id)
        choices = [r[1] for r in rows]
        log += f"\n分割完了: {len(rows)}クリップ。素材一覧で確認してください。"
        yield job_id, log, rows, gr.update(choices=choices, value=choices[0] if choices else None), None
        if mode == "分割から学習まで一括実行":
            for stage in ["encode", "train"]:
                for log in engine.run_stage(job_id, stage):
                    yield job_id, log, gr.skip(), gr.skip(), gr.skip()
            if engine.is_paused(job_id):
                yield (
                    job_id,
                    log + "\n保存して中断しました。「保存状態から再開」で続行できます。",
                    gr.skip(),
                    gr.skip(),
                    engine.bundle(job_id),
                )
                return
            final = engine.final_embedding(job_id)
            if not final.is_file():
                raise RuntimeError("最終埋め込みが見つかりません。ログを確認してください。")
            yield job_id, log + "\n完了: " + str(final), gr.skip(), gr.skip(), engine.bundle(job_id)
    except Exception as exc:
        yield job_id, str(exc), gr.skip(), gr.skip(), gr.skip()


def run_prepared_ui(job_id, mode):
    # Table/clip updates use independent short requests, never stream diffs.
    for _, log, _, _, download in run_prepared(job_id, mode):
        yield log, download


def poll_results(job_id, rendered, selected_clip=None):
    """Recover committed results even when streaming/completion delivery stalls."""
    if not job_id:
        return (gr.skip(),) * 5
    try:
        job = engine.resolve_job(job_id)
        rendered = rendered or {}
        rows_path, log_path = job / "rows.json", job / "run.log"
        def stamp(path):
            try:
                stat = path.stat()
                return [job_id, stat.st_mtime_ns, stat.st_size]
            except FileNotFoundError:
                return [job_id, None, 0]
        rows_stamp, log_stamp = stamp(rows_path), stamp(log_path)
        rows, clip, projects, log = (gr.skip(),) * 4
        if rows_stamp != rendered.get("rows"):
            rows = engine.table(job_id) if rows_path.is_file() else []
            choices = [row[1] for row in rows]
            clip = gr.update(choices=choices, value=selected_clip if selected_clip in choices
                             else (choices[0] if choices else None))
            projects = gr.update(choices=engine.project_choices(), value=job_id)
        if log_stamp != rendered.get("log") and log_path.is_file():
            log = engine.tail(log_path)
        return rows, clip, log, projects, {"rows": rows_stamp, "log": log_stamp}
    except Exception as exc:
        # Keep the old signature so a temporarily unreadable file is retried.
        return gr.skip(), gr.skip(), str(exc), gr.skip(), gr.skip()


def train_existing(job_id, values, *args):
    try:
        engine.CANCELLED.discard(job_id)
        engine.save_table(job_id, values)
        cfg = engine.read(engine.resolve_job(job_id) / "settings.json")
        updated = settings(*args)
        engine.rename_project(job_id, updated.pop("project_name"))
        cfg.update(updated)
        engine.write(engine.resolve_job(job_id) / "settings.json", cfg)
        for stage in ["encode", "train"]:
            for log in engine.run_stage(job_id, stage):
                yield log, gr.skip()
        if engine.is_paused(job_id):
            yield (
                log + "\n保存して中断しました。「保存状態から再開」で続行できます。",
                engine.bundle(job_id),
            )
            return
        final = engine.final_embedding(job_id)
        if not final.is_file():
            raise RuntimeError("最終埋め込みが見つかりません。ログを確認してください。")
        yield log + "\n完了: " + str(final), engine.bundle(job_id)
    except Exception as exc:
        yield str(exc), gr.skip()


def resume_training(job_id, migration_checkpoint="", allow_environment_change=False):
    try:
        with engine.LOCK:
            if job_id in engine.ACTIVE:
                raise ValueError("処理を中断または終了してから再開してください。")
            engine.write(engine.resolve_job(job_id) / "resume_request.json", {
                "checkpoint": migration_checkpoint.strip(),
                "allow_environment_change": bool(allow_environment_change),
            })
        engine.CANCELLED.discard(job_id)
        for log in engine.run_stage(job_id, "resume"):
            yield log, gr.skip()
        if engine.is_paused(job_id):
            yield log + "\n保存して中断しました。", engine.bundle(job_id)
            return
        final = engine.final_embedding(job_id)
        if not final.is_file():
            raise RuntimeError("最終埋め込みが見つかりません。ログを確認してください。")
        yield log + "\n学習が完了しました。", engine.bundle(job_id)
    except Exception as exc:
        yield str(exc), gr.skip()


def save(job_id, values):
    try:
        return engine.save_table(job_id, values), "修正を保存しました。試聴にも反映されます。"
    except Exception as exc:
        raise gr.Error(str(exc))


def save_selected(job_id, values, selected_clip):
    rows, log = save(job_id, values)
    choices = [row[1] for row in rows]
    selected_clip = selected_clip if selected_clip in choices else (choices[0] if choices else None)
    return (
        rows, log, gr.update(choices=choices, value=selected_clip),
        preview(job_id, selected_clip), poll_results(job_id, {}, selected_clip)[-1],
    )


def split_material(job_id, values, clip_id, split_seconds, left_text, right_text):
    try:
        rows, new_id = engine.split_clip(
            job_id, values, clip_id, split_seconds, left_text, right_text
        )
        # Mark the committed table as rendered to keep the timer from changing
        # the selected second half after this response.
        rendered = poll_results(job_id, {})[-1]
        return (
            rows, gr.update(choices=[row[1] for row in rows], value=new_id),
            preview(job_id, new_id),
            "2つの素材に分割し、表の修正も保存しました。前半・後半を試聴してください。"
            "学習に反映する場合は「修正を保存して新しい学習」を使用してください。",
            rendered,
        )
    except Exception as exc:
        raise gr.Error(str(exc))


def render_waveform(job_id, clip_id, values, scope, marker=None):
    import soundfile as sf

    from waveform import draw_waveform

    if not job_id or not clip_id:
        return None, {}, None
    try:
        job = engine.resolve_job(job_id)
        original = next(row for row in engine.read(job / "rows.json") if row["id"] == clip_id)
        edited = next(row for row in values if row[1] == clip_id)
        source = job / "sources" / f'{original["source"]:03d}.wav'
        audio, rate = sf.read(source)
        image, info = draw_waveform(audio, rate, float(edited[2]), float(edited[3]), scope, marker)
        info.update(job_id=job_id, clip_id=clip_id, rate=rate, scope=scope)
        return image, info, str(source)
    except Exception as exc:
        raise gr.Error("波形を表示できません: " + str(exc))


def choose_wave_position(job_id, clip_id, values, kind, info, evt: gr.SelectData):
    from waveform import time_at_pixel

    if not info or info.get("job_id") != job_id or info.get("clip_id") != clip_id:
        raise gr.Error("対象の波形を更新してからクリックしてください。")
    seconds = time_at_pixel(info, evt.index[0])
    if seconds is None:
        return (gr.skip(),) * 5
    # Use the source sample grid, shared by preview bounds and saved clips.
    seconds = int(seconds * info["rate"]) / info["rate"]
    rows = [list(row) for row in values]
    selected = next(row for row in rows if row[1] == clip_id)
    if kind == "分割位置":
        if not float(selected[2]) < seconds < float(selected[3]):
            raise gr.Error("分割位置は、緑の開始線と黄色の終了線の間をクリックしてください。")
        image, updated, _ = render_waveform(job_id, clip_id, rows, info["scope"], seconds)
        return gr.skip(), seconds, image, updated, f"分割位置: {seconds:.3f}秒。前半・後半の文字起こしを入力して分割してください。"
    column = 2 if kind == "開始位置" else 3
    selected[column] = seconds
    if not 0 <= float(selected[2]) < float(selected[3]) <= info["duration"]:
        raise gr.Error("開始位置が終了位置より前になるようクリックしてください。")
    selected[4] = round(float(selected[3])-float(selected[2]), 3)
    image, updated, _ = render_waveform(job_id, clip_id, rows, info["scope"])
    return rows, gr.skip(), image, updated, "切り位置を変更しました。「文字起こし・切り位置の修正を保存」でWAVに反映してください。"


def preview(job_id, clip_id):
    if not clip_id:
        return None
    job = engine.resolve_job(job_id)
    ids = {r["id"] for r in engine.read(job / "rows.json")}
    if clip_id not in ids:
        raise gr.Error("クリップIDが不正です。")
    return str(job / "clips" / (clip_id + ".wav"))


def load(job_id):
    try:
        rows = engine.table(job_id)
        cfg = engine.read(engine.resolve_job(job_id) / "settings.json")
        return (
            rows,
            gr.update(choices=[r[1] for r in rows], value=rows[0][1] if rows else None),
            "プロジェクトを読み込みました。完全再開は設定を変更せず「保存状態から再開」を押してください。",
            *[
                cfg[k]
                for k in [
                    "minimum",
                    "maximum",
                    "asr_model",
                    "asr_device",
                    "language",
                    "caption",
                    "steps",
                    "lr",
                    "batch_size",
                    "accumulation",
                    "checkpoint",
                    "initial_embedding",
                ]
            ],
            engine.project_name(engine.resolve_job(job_id)),
        )
    except Exception as exc:
        raise gr.Error(str(exc))


def rename(job_id, name):
    try:
        name = engine.rename_project(job_id, name)
        return name, gr.update(choices=engine.project_choices(), value=job_id), "プロジェクト名と現在の学習出力名を保存しました。"
    except Exception as exc:
        raise gr.Error(str(exc))


def refresh_results(job_id):
    """Read committed results in a separate, non-streaming UI event."""
    try:
        job = engine.resolve_job(job_id)
        rows = engine.table(job_id) if (job / "rows.json").is_file() else []
        choices = [row[1] for row in rows]
        clip_id = choices[0] if choices else None
        log = engine.tail(job / "run.log")
        if rows:
            log += f"\n素材一覧を読み込みました: {len(rows)}クリップ。文字起こしと切り位置を確認できます。"
        else:
            log += "\n素材一覧はまだありません。上のログで処理結果を確認してください。"
        return (
            rows,
            gr.update(choices=choices, value=clip_id),
            preview(job_id, clip_id),
            log,
            gr.update(choices=engine.project_choices(), value=job_id),
        )
    except Exception as exc:
        return gr.skip(), gr.skip(), gr.skip(), str(exc), gr.skip()


def load_selected(selected, job_id):
    target = selected or job_id
    if not target:
        raise gr.Error("保存済みプロジェクトを選択してください。")
    return (target, *load(target))


def import_transfer(archive):
    try:
        target = engine.import_project(archive)
        return (target, *load(target), gr.update(choices=engine.project_choices(), value=target))
    except Exception as exc:
        raise gr.Error(str(exc))


def build():
    with gr.Blocks(title="Irodori Speaker Studio") as demo:
        gr.Markdown(
            "# Irodori Speaker Studio\n複数音声 → 文脈を優先して分割 → 話者埋め込みを学習（v4 Large）"
        )
        gr.Markdown(
            "5〜10秒は目安です。長い文や独立した短い発話も保持します。音声は同じ話者の素材をまとめてください。"
        )
        auth_status = gr.Markdown("Hugging Faceの認証・モデルアクセスを確認中…")
        auth_refresh = gr.Button("認証状態を再確認", size="sm")
        demo.load(check_access, outputs=auth_status, queue=False, show_progress="hidden")
        auth_refresh.click(check_access, outputs=auth_status, queue=False, show_progress="minimal")
        project_name = gr.Textbox(label="プロジェクト名（任意・日本語可）", placeholder="例：みあ")
        with gr.Row():
            files = gr.File(
                label="音声ファイル（複数選択）",
                file_count="multiple",
                type="filepath",
                file_types=[".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".aac"],
            )
            with gr.Column():
                mode = gr.Radio(
                    ["分割して確認", "分割から学習まで一括実行"],
                    value="分割して確認",
                    label="実行モード",
                )
                begin = gr.Button("開始", variant="primary")
                pause_button = gr.Button("保存して中断（学習中）")
                stop = gr.Button("強制停止（未保存の学習分は失われます）")
        with gr.Accordion("分割・文字起こし設定", open=True):
            with gr.Row():
                minimum = gr.Number(5, label="目安の最小秒数")
                maximum = gr.Number(10, label="目安の最大秒数")
                asr_model = gr.Dropdown(
                    ["large-v3", "medium", "small"], value="large-v3", label="Whisperモデル"
                )
                asr_device = gr.Radio(["cpu", "cuda"], value="cpu", label="文字起こし実行先")
                language = gr.Textbox("ja", label="言語（日本語: ja / 空欄: 自動）")
        with gr.Accordion("v4 Large Speaker Inversion設定", open=True):
            caption = gr.Textbox("", label="声・スタイルの説明（任意）")
            with gr.Row():
                steps = gr.Number(3000, precision=0, label="学習ステップ数")
                lr = gr.Number(0.01, label="学習率")
                batch = gr.Number(1, precision=0, label="バッチサイズ")
                accumulation = gr.Number(1, precision=0, label="勾配蓄積回数")
            checkpoint = gr.Textbox(
                "", label="v4 Large model.safetensorsのパス（空欄なら自動取得）"
            )
            embedding = gr.Textbox("", label="追加学習する.speaker.safetensorsのパス（任意）")
            gr.Markdown(
                "BF16・Gradient Checkpointingを使用。バッチ1から開始できます。GPUメモリ不足時はログに表示されます。"
            )
        config_inputs = [
            minimum,
            maximum,
            asr_model,
            asr_device,
            language,
            caption,
            steps,
            lr,
            batch,
            accumulation,
            checkpoint,
            embedding,
            project_name,
        ]
        project_choices = engine.project_choices()
        initial_project = project_choices[0][1] if project_choices else None
        with gr.Row():
            saved_project = gr.Dropdown(
                choices=project_choices,
                value=initial_project,
                label="保存済みプロジェクト（再起動後はこちらから選択）",
            )
            refresh_projects = gr.Button("プロジェクト一覧を更新")
        with gr.Row():
            job_id = gr.Textbox(value=initial_project or "", label="プロジェクトID（再起動後もこのIDで開けます）")
            reload_button = gr.Button("保存プロジェクトを開く")
            rename_button = gr.Button("プロジェクト名を保存")
            refresh_results_button = gr.Button("素材一覧を再読み込み")
        table = gr.Dataframe(
            headers=engine.HEADERS,
            datatype=["bool", "str", "number", "number", "number", "str", "str"],
            type="array",
            interactive=True,
            label="素材一覧・修正（IDと長さ秒は変更不要）",
            wrap=True,
        )
        with gr.Row():
            clip = gr.Dropdown(label="試聴するクリップ")
            audio = gr.Audio(label="分割音声", type="filepath")
            listen = gr.Button("試聴を更新")
        with gr.Accordion("元音声の波形で切り位置を指定", open=True):
            gr.Markdown(
                "操作を選んで波形をクリックしてください。緑線が開始、黄色線が終了です。"
                "開始・終了の変更は下の「文字起こし・切り位置の修正を保存」で反映します。"
                "分割位置を選んだ場合は、下の2分割機能を使ってください。"
            )
            with gr.Row():
                wave_scope = gr.Radio(["素材周辺", "元音声全体"], value="素材周辺", label="波形の表示範囲")
                wave_kind = gr.Radio(["分割位置", "開始位置", "終了位置"], value="分割位置", label="クリックで指定する位置")
                update_wave = gr.Button("波形を更新")
            wave_info = gr.State({})
            wave_image = gr.Image(label="元音声の波形（クリックして指定）", type="pil",
                                  interactive=False, show_download_button=False)
            source_audio = gr.Audio(label="元音声の試聴（全体）", type="filepath")
        with gr.Accordion("長い素材を2つに分割（1行追加）", open=False):
            gr.Markdown(
                "上の「試聴するクリップ」で分割対象を選び、波形のクリックで"
                "分割位置を指定してください。秒数欄でも微調整できます。"
                "前半・後半の文字起こしを入力して分割すると、表の修正も一緒に保存します。"
                "表の＋で空行を追加する代わりに、この機能を使ってください。"
            )
            split_seconds = gr.Number(label="分割位置（元音声の先頭からの秒数）")
            with gr.Row():
                left_text = gr.Textbox(label="前半の文字起こし", lines=3)
                right_text = gr.Textbox(label="後半の文字起こし", lines=3)
            split_button = gr.Button("選択した素材を2つに分割して保存")
        with gr.Row():
            save_button = gr.Button("文字起こし・切り位置の修正を保存")
            train_button = gr.Button("修正を保存して新しい学習", variant="primary")
            resume_button = gr.Button("保存状態から再開", variant="primary")
            export_button = gr.Button("素材・途中結果をZIPで取得")
        logs = gr.Textbox(label="進捗・学習ログ", lines=14, max_lines=20, interactive=False)
        download = gr.File(label="結果ZIP（音声・文字起こし・潜在表現・学習済み埋め込み）")
        with gr.Accordion("別PCへの移行・移行先での再開", open=False):
            gr.Markdown("元PCで保存して中断し、移行用ZIPを取得してください。移行先では同じ版のアプリをセットアップしてZIPを読み込みます。モデルはZIPに含まれず、空欄なら必要時に取得します。PyTorchのバージョンとCUDAデバイス数は元PCと揃えてください。")
            transfer_export = gr.Button("PC移行用ZIPを取得（元音声・素材・学習状態）")
            transfer_file = gr.File(label="移行用ZIP", file_types=[".zip"], type="filepath")
            transfer_import = gr.Button("移行用ZIPを読み込む")
            migration_checkpoint = gr.Textbox(label="移行先のv4 Largeモデルのパス（任意）")
            allow_environment_change = gr.Checkbox(label="移行先のGPU・CUDAの違いを許可して再開", value=False)
        saved_project.input(lambda value: value, saved_project, job_id, queue=False)
        refresh_projects.click(
            lambda: gr.update(choices=engine.project_choices()), None, saved_project
        )
        # Opening a saved project loads its settings as well as its table.
        # Merely selecting the initial dropdown must not load rows with default
        # training settings. New projects get a different ID and are polled.
        initial_rendered = poll_results(initial_project, {})[-1] if initial_project else {}
        rendered = gr.State(initial_rendered if isinstance(initial_rendered, dict) else {})
        result_timer = gr.Timer(2.0)
        split_button.click(
            split_material, [job_id, table, clip, split_seconds, left_text, right_text],
            [table, clip, audio, logs, rendered], concurrency_id="work",
            concurrency_limit=1,
        )
        prepared = begin.click(
            prepare_start,
            [files] + config_inputs,
            [job_id, logs, table, clip, download],
            concurrency_id="work",
            concurrency_limit=1,
            show_progress="minimal",
        )
        started = prepared.success(
            run_prepared_ui, [job_id, mode], [logs, download],
            concurrency_id="work", concurrency_limit=1, show_progress="hidden",
        )
        result_timer.tick(
            poll_results, [job_id, rendered, clip], [table, clip, logs, saved_project, rendered],
            queue=False, show_progress="hidden",
        )
        started.then(
            refresh_results, job_id, [table, clip, audio, logs, saved_project],
            queue=False, show_progress="minimal",
        )
        train_button.click(
            train_existing,
            [job_id, table] + config_inputs,
            [logs, download],
            concurrency_id="work",
            concurrency_limit=1,
        )
        save_button.click(save_selected, [job_id, table, clip],
                          [table, logs, clip, audio, rendered], concurrency_id="work").success(
            render_waveform, [job_id, clip, table, wave_scope],
            [wave_image, wave_info, source_audio], queue=False, show_progress="hidden",
        )
        reload_button.click(
            load_selected, [saved_project, job_id], [job_id, table, clip, logs] + config_inputs,
            concurrency_id="work"
        ).then(
            refresh_results, job_id, [table, clip, audio, logs, saved_project],
            queue=False, show_progress="minimal",
        ).success(
            render_waveform, [job_id, clip, table, wave_scope],
            [wave_image, wave_info, source_audio], queue=False, show_progress="hidden",
        )
        refresh_results_button.click(
            refresh_results, job_id, [table, clip, audio, logs, saved_project],
            queue=False, show_progress="minimal",
        )
        rename_button.click(
            rename, [job_id, project_name], [project_name, saved_project, logs],
            concurrency_id="work",
        )
        clip.change(preview, [job_id, clip], audio, queue=False, show_progress="hidden")
        for trigger in [clip.change, wave_scope.change, update_wave.click]:
            trigger(render_waveform, [job_id, clip, table, wave_scope],
                    [wave_image, wave_info, source_audio], queue=False, show_progress="hidden")
        wave_image.select(
            choose_wave_position, [job_id, clip, table, wave_kind, wave_info],
            [table, split_seconds, wave_image, wave_info, logs],
            queue=False, show_progress="hidden",
        )
        listen.click(preview, [job_id, clip], audio)
        export_button.click(engine.bundle, job_id, download, concurrency_id="work")
        transfer_export.click(engine.export_project, job_id, download, concurrency_id="work")
        transfer_import.click(import_transfer, transfer_file,
                              [job_id, table, clip, logs] + config_inputs + [saved_project],
                              concurrency_id="work").success(
            refresh_results, job_id, [table, clip, audio, logs, saved_project], queue=False,
        ).success(render_waveform, [job_id, clip, table, wave_scope],
                  [wave_image, wave_info, source_audio], queue=False)
        pause_button.click(engine.pause, job_id, logs, queue=False)
        resume_button.click(
            resume_training, [job_id, migration_checkpoint, allow_environment_change],
            [logs, download], concurrency_id="work", concurrency_limit=1
        )
        stop.click(engine.cancel, job_id, logs, queue=False)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7865, type=int)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    username, password = os.getenv("IRODORI_USER"), os.getenv("IRODORI_PASSWORD")
    if args.host not in ("127.0.0.1", "localhost") and not (username and password):
        parser.error("外部公開時はIRODORI_USERとIRODORI_PASSWORDを設定してください。")
    print(
        "Irodori Speaker Studioを起動します。処理の進捗と学習ログはこの画面にも表示されます。",
        flush=True,
    )
    print("素材・ログの保存先: " + str(engine.JOBS), flush=True)
    build().queue().launch(
        inbrowser=args.open_browser,
        server_name=args.host,
        server_port=args.port,
        auth=(username, password) if username and password else None,
        allowed_paths=[str(engine.JOBS)],
    )
