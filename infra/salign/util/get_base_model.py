from infra.salign.util.get_config import get_config


def get_base_model():
    config = get_config()

    return config.get("base_model", "qwen")
