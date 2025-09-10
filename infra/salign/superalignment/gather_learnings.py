from infra.salign.superalignment.explain_errors import explain_errors
from infra.salign.sql.execute_query import execute_query

from infra.salign.superalignment.write_queries_to_research_dataset import (
    write_queries_to_research_dataset,
)

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import copy

import logging

logger = logging.getLogger(__name__)


def gather_learnings(eval_results, llm_info, seed):
    correct_results = [
        result for result in eval_results["results"] if result["score"] >= 1.0
    ]

    eval_explanations = explain_errors(eval_results, seed)

    queries = write_queries_to_research_dataset(eval_explanations, llm_info, seed)

    query_results = execute_queries(queries)

    learnings = extract_learnings_from_query_results(query_results, eval_explanations)

    cleanup(learnings)

    logger.info(
        f"Gathered {len(learnings)} learnings from eval results and {len(correct_results)} correct results."
    )

    final_results = {
        "results": correct_results + learnings,
        **{k: v for k, v in eval_results.items() if k != "results"},
    }

    return final_results


def execute_queries(queries):
    query_results = []

    for query in queries:
        query_result = copy.deepcopy(query)
        logger.debug(f"Gathering evidence for question: {query_result['question']}")
        for evidence in query_result["evidence"]:
            sql = evidence["sql"]
            if not "result" in evidence:
                result, failed = execute_query(query_result, sql)
                evidence["result"] = result
                evidence["failed"] = failed

        query_results.append(query_result)

    return query_results


def extract_learnings_from_query_results(query_results, eval_explanations):
    question_to_explanation_map = {
        explanation["question"]: copy.deepcopy(explanation)
        for explanation in eval_explanations
    }

    for query_result in query_results:
        question = query_result["question"]
        if question not in question_to_explanation_map:
            continue

        explanation = question_to_explanation_map[question]

        if not "evidence" in explanation:
            explanation["evidence"] = []

        for evidence in query_result["evidence"]:
            explanation["evidence"].append(evidence)

    return [
        question_to_explanation_map[question]
        for question in sorted(question_to_explanation_map.keys())
    ]


def cleanup(learnings):
    for query in learnings:
        if "evidence" in query:
            dedup(query["evidence"])

            config = get_config()
            max_evidence = config["maximum_evidence_history"]

            if len(query["evidence"]) > max_evidence:
                gap = len(query["evidence"]) - max_evidence
                remove_failed_evidence(query, gap)

            if len(query["evidence"]) > max_evidence:
                query["evidence"] = query["evidence"][-max_evidence:]


def dedup(evidence_list):
    seen = set()
    deduped_evidence = []
    for evidence in evidence_list:
        sql = evidence["sql"]
        if sql not in seen:
            seen.add(sql)
            deduped_evidence.append(evidence)
    evidence_list.clear()
    evidence_list.extend(deduped_evidence)


def remove_failed_evidence(query, gap):
    removed_count = 0

    # Remove up to gap items if they have failed
    new_evidence = []

    for evidence in query["evidence"]:
        if "failed" in evidence and evidence["failed"]:
            if removed_count < gap:
                removed_count += 1
                continue

        new_evidence.append(evidence)

    query["evidence"] = new_evidence
