from infra.salign.superalignment.refine_queries import refine_queries
from infra.salign.superalignment.judge_query_alternatives import judge_query_alternatives

from infra.salign.util.get_config import get_config

import copy

import logging

logger = logging.getLogger(__name__)


def refine_queries_with_results(queries):
    queries_to_refine = copy.deepcopy(queries)
    refined_queries = []

    config = get_config()

    max_iterations = config["max_query_refinement_iterations"]

    for i in range(max_iterations):
        queries_to_refine = refine_queries(queries_to_refine)

        logger.info(f"Iteration {i}: {len(queries_to_refine)} queries refined.")

    refined_queries = select_best_queries(queries_to_refine)

    logger.info(
        f"Refined {len(refined_queries)} queries after {max_iterations} iterations."
    )

    # Remove failed queries or queries with empty results
    final_refined_queries = [
        query
        for query in refined_queries
        if not query["failed"] and len(query["reference_result"]) > 0
    ]

    # logger.info(f"FAILED: {[query['failed'] for query in final_refined_queries]}")

    logger.info(
        f"Final refined queries: {len(final_refined_queries)} queries after removing "
        f"{len(refined_queries) - len(final_refined_queries)} failed queries."
    )

    return final_refined_queries


def select_best_queries(queries):
    judged_queries = judge_query_alternatives(queries)

    best_queries = []

    for query in judged_queries:
        best_query = copy.deepcopy(query)

        score = best_query["score"]

        for alternate_query in best_query["alternate_queries"]:
            if alternate_query["score"] > score:
                best_query["reference_sql"], alternate_query["reference_sql"] = (
                    alternate_query["reference_sql"],
                    best_query["reference_sql"],
                )
                best_query["reference_result"], alternate_query["reference_result"] = (
                    alternate_query["reference_result"],
                    best_query["reference_result"],
                )
                best_query["failed"], alternate_query["failed"] = (
                    alternate_query["failed"],
                    best_query["failed"],
                )
                best_query["refinement"], alternate_query["refinement"] = (
                    alternate_query["refinement"],
                    best_query["refinement"],
                )
                best_query["score"], alternate_query["score"] = (
                    alternate_query["score"],
                    best_query["score"],
                )

        best_queries.append(best_query)

    return best_queries
