"""Heavy work runs in separate processes to release GPU memory between phases."""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "irodori"
sys.path.insert(0, str(UPSTREAM))
from segmentation import Word, padded_bounds, segment_words


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def split(job):
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    cfg = read_json(job / "settings.json")
    model = WhisperModel(
        cfg["asr_model"],
        device=cfg["asr_device"],
        compute_type="int8" if cfg["asr_device"] == "cpu" else "float16",
    )
    rows, warnings = [], []
    for index, source in enumerate(cfg["sources"]):
        print(f"文字起こし {index + 1}/{len(cfg['sources'])}: {Path(source).name}", flush=True)
        # Decode original at 48kHz for saved clips; ASR separately resamples to 16kHz.
        audio = decode_audio(source, sampling_rate=48000)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"空または不正な音声: {source}")
        sf.write(job / "sources" / f"{index:03d}.wav", audio, 48000, subtype="PCM_24")
        segments, _ = model.transcribe(
            source,
            language=cfg["language"] or None,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
        words = []
        for segment in segments:
            for w in segment.words or []:
                words.append(Word(w.start, w.end, w.word, w.probability))
        if not words:
            warnings.append(f"{Path(source).name}: 発話を検出できませんでした")
            continue
        spans = segment_words(words, cfg["minimum"], cfg["maximum"])
        bounds = padded_bounds(spans, len(audio) / 48000)
        for span, (start, end) in zip(spans, bounds):
            clip_id = f"{index:03d}_{len(rows):05d}"
            # Durations are based on saved samples, not estimates from ASR.
            first, last = int(start * 48000), min(len(audio), int(end * 48000))
            sf.write(job / "clips" / f"{clip_id}.wav", audio[first:last], 48000, subtype="PCM_24")
            duration = (last - first) / 48000
            notes = []
            if duration < cfg["minimum"] or duration > cfg["maximum"]:
                notes.append("文脈優先・目安の秒数外")
            if span.confidence < 0.75:
                notes.append("文字起こし要確認")
            if duration > 30:
                notes.append("長い音声・VRAM使用量増加")
            rows.append(
                {
                    "use": True,
                    "id": clip_id,
                    "source": index,
                    "start": first / 48000,
                    "end": last / 48000,
                    "duration": duration,
                    "text": span.text,
                    "confidence": span.confidence,
                    "note": " / ".join(notes),
                }
            )
        print(f"分割完了: {len(spans)}クリップ", flush=True)
    if not rows:
        raise ValueError("学習に使用できる発話がありません。音声と言語設定を確認してください。")
    write_json(job / "rows.json", rows)
    write_json(job / "warnings.json", warnings)
    for warning in warnings:
        print("注意: " + warning, flush=True)


def encode(job):
    import soundfile as sf
    import torch
    from irodori_tts.codec import DACVAECodec

    cfg = read_json(job / "settings.json")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPUが見つかりません。分割はCPUでも可能ですが学習にはCUDAが必要です。"
        )
    codec = DACVAECodec.load(device="cuda", normalize_db=-16.0)
    rows = [r for r in read_json(job / "rows.json") if r["use"]]
    if not rows:
        raise ValueError("学習対象が0件です。")
    temporary = job / "manifest.tmp"
    with temporary.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            audio, sr = sf.read(job / "clips" / (row["id"] + ".wav"), dtype="float32")
            with torch.inference_mode():
                latent = codec.encode_waveform(torch.from_numpy(audio).unsqueeze(0), sr)[0].cpu()
            if latent.ndim != 2 or latent.shape[1] != 32 or not torch.isfinite(latent).all():
                raise ValueError("不正なDACVAE潜在表現: " + row["id"])
            name = "latents/" + row["id"] + ".pt"
            torch.save(latent, job / name)
            record = {
                "text": row["text"],
                "caption": cfg["caption"],
                "latent_path": name,
                "num_frames": int(latent.shape[0]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"DACVAE変換 {i + 1}/{len(rows)} ({latent.shape[0]} frames)", flush=True)
    temporary.replace(job / "manifest.jsonl")


def train(job):
    import torch
    import yaml
    from huggingface_hub import snapshot_download
    from irodori_tts.config import ModelConfig
    from safetensors import safe_open

    cfg = read_json(job / "settings.json")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDAとBF16対応GPUが必要です。")
    checkpoint = cfg.get("checkpoint", "").strip()
    if checkpoint:
        checkpoint = str(Path(checkpoint).expanduser().resolve())
        if not Path(checkpoint).is_file() or Path(checkpoint).suffix != ".safetensors":
            raise ValueError("v4 Largeの非量子化model.safetensorsを指定してください。")
    else:
        print("v4 Largeをダウンロードします（初回のみ・大容量）", flush=True)
        directory = snapshot_download(
            "Aratako/Irodori-TTS-v4-Large",
            allow_patterns=["*.json", "*.safetensors", "tokenizer/*", "*.model"],
        )
        checkpoint = str(Path(directory) / "model.safetensors")
    # Follow checkpoint metadata, including its duration architecture; do not reuse Small dimensions.
    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    from irodori_tts.quantization import parse_quantization_metadata

    if parse_quantization_metadata(metadata) is not None:
        raise ValueError("量子化モデルは学習には使用できません。")
    model = json.loads(metadata.get("config_json", "{}"))
    valid_keys = {field.name for field in dataclasses.fields(ModelConfig)}
    model = {k: v for k, v in model.items() if k in valid_keys}
    if model.get("model_dim") != 2048 or model.get("num_layers") != 24:
        raise ValueError("指定モデルはv4 Largeの構成ではありません。")
    train_cfg = yaml.safe_load(
        (UPSTREAM / "configs/train_v4_small_speaker_inversion.yaml").read_text()
    )["train"]
    records = [json.loads(line) for line in (job / "manifest.jsonl").read_text().splitlines()]
    # Retain context-priority long clips: raise frame and text limits instead of cropping.
    from irodori_tts.tokenizer import PretrainedTextTokenizer

    tokenizer = PretrainedTextTokenizer.from_pretrained(
        model["text_tokenizer_repo"],
        add_bos=model.get("text_add_bos", True),
        revision=model.get("text_encoder_revision"),
    )
    max_text = max(len(tokenizer.encode(r["text"])) for r in records)
    from training_config import make_config

    generated = make_config(metadata, valid_keys, train_cfg, records, cfg, max_text)
    train_cfg = generated["train"]
    output_dir = job / "output" / uuid.uuid4().hex[:12]
    output_dir.mkdir(parents=True)
    write_json(job / "training.json", {"output_dir": str(output_dir), "checkpoint": checkpoint})
    run_config = job / "train_large_speaker.yaml"
    run_config.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
    write_json(
        job / "provenance.json",
        {
            "model_repo": "Aratako/Irodori-TTS-v4-Large",
            "checkpoint": checkpoint,
            "upstream_commit": "89f9d8fbd4d51ea019867ee1197725ede1df13c5",
            "samples": len(records),
            "settings": cfg,
        },
    )
    command = [
        sys.executable,
        "-u",
        str(UPSTREAM / "train.py"),
        "--config",
        str(run_config),
        "--manifest",
        str(job / "manifest.jsonl"),
        "--init-checkpoint",
        checkpoint,
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
    ]
    if cfg.get("initial_embedding"):
        embedding = Path(cfg["initial_embedding"]).expanduser().resolve()
        if not embedding.is_file():
            raise ValueError("追加学習用埋め込みファイルがありません。")
        command += ["--speaker-inversion-init-embedding", str(embedding)]
    print(
        "話者埋め込みのみ学習します。最長音声のframe数: " + str(train_cfg["max_latent_steps"]),
        flush=True,
    )
    setup_resume_environment(job, output_dir, command, checkpoint)
    launch_trainer(command)


def training_identity(job, checkpoint):
    import studio_state
    import torch

    records = [
        json.loads(line)
        for line in (job / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    print("再開用に基盤モデルと素材の一致確認を行っています...", flush=True)
    return {
        "base_sha256": studio_state.digest(checkpoint),
        "manifest_sha256": studio_state.digest(job / "manifest.jsonl"),
        "latents": {r["latent_path"]: studio_state.digest(job / r["latent_path"]) for r in records},
        "rows_sha256": studio_state.digest(job / "rows.json"),
        "clips": {
            r["latent_path"]: studio_state.digest(
                job / "clips" / (Path(r["latent_path"]).stem + ".wav")
            )
            for r in records
        },
        "config_sha256": studio_state.digest(job / "train_large_speaker.yaml"),
        "trainer_sha256": studio_state.digest(UPSTREAM / "train.py"),
        "extension_sha256": studio_state.digest(UPSTREAM / "studio_state.py"),
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
    }


def setup_resume_environment(job, output_dir, command, checkpoint):
    context = training_identity(job, checkpoint)
    record = read_json(job / "training.json")
    from engine import project_name
    record.update(command=command, context=context, output_prefix=project_name(job) or "checkpoint")
    write_json(job / "training.json", record)
    os.environ["STUDIO_OUTPUT_PREFIX"] = record["output_prefix"]
    os.environ["STUDIO_STATE_DIR"] = str(output_dir)
    os.environ["STUDIO_PAUSE_FILE"] = str(job / "pause.request")
    os.environ["STUDIO_CONTEXT"] = json.dumps(context, sort_keys=True)
    os.environ.pop("STUDIO_RESUME_STATE", None)
    (job / "pause.request.complete").unlink(missing_ok=True)


def resume(job):
    record = read_json(job / "training.json")
    state = Path(record["output_dir"]) / "resume_state.pt"
    if not state.is_file() or "command" not in record:
        raise ValueError("再開用状態がありません。この機能の導入前の学習は完全再開できません。")
    context = training_identity(job, record["checkpoint"])
    if context != record["context"]:
        raise ValueError("モデル・素材・設定・GPU・コードが変更されています。完全再開できません。")
    (job / "pause.request.complete").unlink(missing_ok=True)
    os.environ["STUDIO_OUTPUT_PREFIX"] = record.get("output_prefix", "checkpoint")
    os.environ["STUDIO_STATE_DIR"] = record["output_dir"]
    os.environ["STUDIO_PAUSE_FILE"] = str(job / "pause.request")
    os.environ["STUDIO_CONTEXT"] = json.dumps(context, sort_keys=True)
    os.environ["STUDIO_RESUME_STATE"] = str(state)
    launch_trainer(record["command"])


def launch_trainer(command):
    command = list(command)
    command = [str(ROOT / "named_trainer.py") if value == str(UPSTREAM / "train.py") else value
               for value in command]
    if sys.platform == "win32":
        # os.execv on Windows can lose the parent relationship needed for taskkill /T.
        completed = subprocess.run(command, cwd=UPSTREAM, check=False)
        raise SystemExit(completed.returncode)
    os.chdir(UPSTREAM)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["split", "encode", "train", "resume"])
    parser.add_argument("job")
    args = parser.parse_args()
    globals()[args.stage](Path(args.job).resolve())
