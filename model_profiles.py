"""Supported base models; missing selections in old projects mean Large."""

DEFAULT_MODEL = "v4-Large"
MODEL_PROFILES = {
    "v4-Large": {
        "repo": "Aratako/Irodori-TTS-v4-Large",
        "model_dim": 2048,
        "num_layers": 24,
        "text_tokenizer_repo": "google/t5gemma-2-1b-1b",
        "config_file": "train_large_speaker.yaml",
    },
    "v4.1-Small": {
        "repo": "Aratako/Irodori-TTS-v4.1-Small",
        "model_dim": 1280,
        "num_layers": 12,
        "text_tokenizer_repo": "sbintuitions/modernbert-ja-310m",
        "config_file": "train_small_speaker.yaml",
    },
}


def model_profile(variant=DEFAULT_MODEL):
    if variant not in MODEL_PROFILES:
        raise ValueError("未対応の学習モデルです。v4-Large / v4.1-Smallを選択してください。")
    return MODEL_PROFILES[variant]
