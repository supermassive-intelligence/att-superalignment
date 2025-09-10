from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import random

import logging

logger = logging.getLogger(__name__)


def get_alternate_queries(example, seed):

    if not "generated_sql" in example:
        return ""

    config = get_config()
    maximum_queries = config["maximum_alternate_queries"]

    prompt = "The analyst already made several failed attempts to write a query.\n"

    alternate_queries = []

    if "alternate_queries" in example:
        alternate_queries = example["alternate_queries"]

    random.seed(seed)
    past_queries = random.sample(
        alternate_queries, min(len(alternate_queries), maximum_queries)
    )

    if len(past_queries) > 0:
        prompt += "Previous incorrect queries:\n"
        for i, q in enumerate(past_queries, start=1):
            prompt += f"Incorrect Query {i}:\n"
            prompt += f"```sql\n{q['sql']}\n```\n"
            prompt += f"Result: {query_results_to_string(q['result'])}\n\n"

    prompt += "Most recent incorrect query:\n"
    prompt += f"```sql\n{example['generated_sql']}\n```\n"
    prompt += f"Result: {query_results_to_string(example['generated_result'])}\n\n"

    prompt += "Compare those incorrect results to this correct result:\n"
    prompt += f"{query_results_to_string(example['reference_result'])}\n"

    prompt += "The results should be identical, but they are not. So there is a mistake in the most recent query.\n\n"

    prompt += example["explanation"] + "\n\n"

    prompt += "Think about how to fix the query.\n"

    return prompt
