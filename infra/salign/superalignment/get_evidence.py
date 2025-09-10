from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.util.get_config import get_config

import random

import logging

logger = logging.getLogger(__name__)


def get_evidence(example, seed):

    if not "evidence" in example:
        return ""

    if len(example["evidence"]) == 0:
        return ""

    config = get_config()

    maximum_queries = config["maximum_research_queries"]

    gap = min(len(example["evidence"]), maximum_queries)

    prompt = "The analyst already wrote these helper queries to gather evidence to help write the final query:\n\n"

    evidence = []

    if "evidence" in example:
        evidence.extend(example["evidence"])

    population_size = max(len(evidence) - gap, 0)
    sample_count = min(population_size, maximum_queries - gap)
    sample_count = max(sample_count, 0)  # Ensure sample_count is not negative

    random.seed(seed)
    evidence_queries = random.sample(evidence[:-gap], sample_count)

    queries = evidence_queries + evidence[-gap:]

    for i, q in enumerate(queries, start=1):
        prompt += f"Query {i}:\n"
        prompt += f"```sql\n{q['sql']}\n```\n"
        prompt += f"Result: {query_results_to_string(q['result'])}\n\n"

    return prompt
