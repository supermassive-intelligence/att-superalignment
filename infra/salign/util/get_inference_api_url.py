from infra.salign.util.get_config import get_config


def get_inference_api_url():
    config = get_config()

    return config.get("inference_api_url", "http://localhost:8000")
