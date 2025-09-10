from infra.salign.superalignment.text2sql import text2sql


def evaluate_llm(llm, dataset):

    results = text2sql(
        dataset,
        model_name=llm["model_name"],
        api_url=llm["api_url"],
    )

    total_count = 0
    match_count = 0

    for result in results:
        if result["score"]:
            match_count += result["score"]
        total_count += 1

    return {"accuracy": match_count / total_count, "results": results}
