from infra.salign.util.get_config import get_config


def query_results_to_string(results):
    config = get_config()

    string = str(results)

    max_len = config["max_query_result_length"]

    if len(string) <= max_len:
        return string

    half_max = max_len // 2

    return (
        string[:half_max]
        + "..."
        + string[-half_max:]
        + f" (total rows: {len(results)})"
    )
