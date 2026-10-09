"""Build a Large Speaker Inversion configuration without changing base architecture."""

import json


def make_config(metadata, allowed_model_keys, defaults, records, cfg, max_text):
    model = json.loads(metadata.get("config_json", "{}"))
    model = {k: v for k, v in model.items() if k in allowed_model_keys}
    if (
        model.get("model_dim") != 2048
        or model.get("num_layers") != 24
        or model.get("latent_dim") != 32
        or model.get("text_tokenizer_repo") != "google/t5gemma-2-1b-1b"
    ):
        raise ValueError("指定モデルはv4 Largeの構成ではありません。")
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
