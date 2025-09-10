from infra.salign.util.get_config import get_config


def get_train_api_url():
    config = get_config()

    return config.get("train_api_url", "http://localhost:8000")
