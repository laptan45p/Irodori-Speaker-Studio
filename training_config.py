"""Build Speaker Inversion configurations without changing base architecture."""

import json

from model_profiles import DEFAULT_MODEL, model_profile


def checkpoint_model(metadata, allowed_model_keys, variant=DEFAULT_MODEL):
    profile = model_profile(variant)
    model = json.loads(metadata.get("config_json", "{}"))
    model = {k: v for k, v in model.items() if k in allowed_model_keys}
    expected = {k: profile[k] for k in ("model_dim", "num_layers", "text_tokenizer_repo")}
    expected["latent_dim"] = 32
    if any(model.get(key) != value for key, value in expected.items()):
        raise ValueError(f"指定モデルは{variant}の構成ではありません。学習モデルの選択とパスを確認してください。")
    if model.get("use_speaker_condition") is False:
        raise ValueError("話者条件を使用しないモデルではSpeaker Inversionを学習できません。")
    return model


def make_config(metadata, allowed_model_keys, defaults, records, cfg, max_text):
    model = checkpoint_model(metadata, allowed_model_keys, cfg.get("model_variant", DEFAULT_MODEL))
    if not records or any(r["num_frames"] <= 0 or not r["text"].strip() for r in records):
        raise ValueError("学習マニフェストが空または不正です。")
    train = dict(defaults)
    train.update(
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["accumulation"],
        max_steps=cfg["steps"],
        learning_rate=cfg["lr"],
        num_workers=0,
        dataloader_persistent_workers=False,
        max_latent_steps=max(r["num_frames"] for r in records),
        max_text_len=max(256, max_text + 8),
        precision="bf16",
        gradient_checkpointing=True,
        speaker_inversion_enabled=True,
        duration_loss_weight=0.0,
        save_every=min(250, cfg["steps"]),
        wandb_enabled=False,
    )
    return {"model": model, "train": train}
