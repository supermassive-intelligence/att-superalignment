from infra.salign.sql.execute_query import execute_query
from infra.salign.sql.extract_sql import extract_sql
from infra.salign.sql.extract_reasoning import extract_reasoning

from infra.salign.superalignment.text2sql import add_metrics

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import copy
import random

import scalarlm

import logging

logger = logging.getLogger(__name__)


def explore_trajectories(results, llm_info, reasoners, seed=42):

    errors = [result for result in results["results"] if result["score"] < 1.0]

    if not errors:
        logger.info("No errors found in results.")
        return results

    logger.info(f"Found {len(errors)} errors to explore.")

    if len(reasoners) == 0:
        logger.warning("No reasoners provided for exploration.")
        return results

    config = get_config()

    trajectories_per_error = min(len(reasoners), config["trajectories_per_error"])

    prompts = make_trajectory_prompts(errors, trajectories_per_error, reasoners, seed)

    responses = explore_trajectories_inference(prompts, llm_info)
    print(responses)

    new_results = make_results(errors, responses, trajectories_per_error)

    new_results = add_metrics(new_results)

    # Add new results as well as all keys from the original results other than results
    final_results = {
        "results": new_results,
        **{k: v for k, v in results.items() if k != "results"},
    }

    # recompute accuracy
    final_results["accuracy"] = (
        sum(1 for r in final_results["results"] if r and r["score"] >= 1.0)
        / len(final_results["results"])
        if len(final_results["results"]) > 0
        else 0
    )

    return final_results

def explore_trajectories_inference(prompts, llm_info):
    try:
        llm = scalarlm.SupermassiveIntelligence(api_url=llm_info["api_url"])

        responses = llm.generate(
            prompts, max_tokens=1024, model_name=llm_info["model_name"]
        )
        return responses
    except AssertionError as e:
        logger.error(f"Explore trajectories failed: {e}")
        return []


def make_trajectory_prompts(errors, trajectories_per_error, reasoners, seed=42):
    prompts = []

    random.seed(seed)
    for index, error in enumerate(errors):
        selected_reasoners = random.sample(reasoners, trajectories_per_error)

        for reasoner in selected_reasoners:
            prompt = make_prompt(error, reasoner, seed * len(errors) + index)
            prompts.append(prompt)

    return prompts


def make_prompt(error, reasoner, seed=42):
    return reasoner.forward(error, seed=seed)


def make_results(errors, responses, trajectories_per_error):
    results = []

    if len(responses) == 0:
        return results

    index = 0

    for error in errors:
        for i in range(trajectories_per_error):
            response = responses[index]
            result = copy.deepcopy(error)
            add_alternate_query(result)

            result["generated_sql"] = extract_sql(response)
            result["reasoning"] = extract_reasoning(response)
            generated_result, generated_failed = execute_query(result, result["generated_sql"])

            result["generated_result"] = generated_result
            result["generated_failed"] = generated_failed
            results.append(result)
            index += 1

    return results


def add_alternate_query(example):

    if "generated_sql" not in example:
        return

    if not "alternate_queries" in example:
        example["alternate_queries"] = []

    alternate_query = {
        "sql": example["generated_sql"],
        "result": example["generated_result"],
        "failed": example["generated_failed"],
    }

    example["alternate_queries"].append(alternate_query)

    config = get_config()

    maximum_alternate_queries = config["maximum_alternate_query_history"]

    if len(example["alternate_queries"]) > maximum_alternate_queries:
        example["alternate_queries"] = remove_worst(
            example["alternate_queries"], maximum_alternate_queries
        )


def remove_worst(alternate_queries, maximum_alternate_queries):
    # Sort by failed status first, then by result length (higher is better)
    alternate_queries.sort(key=lambda x: (x["failed"], -len(x["result"])))

    # Keep the best `maximum_alternate_queries` entries
    return alternate_queries[:maximum_alternate_queries]
