from infra.salign.sql.get_full_db_profile import get_full_db_profile

from infra.salign.sql.execute_query import execute_query
from infra.salign.sql.extract_sql import extract_sql
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.util.get_inference_api_url import get_inference_api_url
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import copy
import sqlite3
import random

import logging

logger = logging.getLogger(__name__)

def refine_queries(queries):
    prompts, results, failed = make_refine_prompts(queries)

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=512)

    refined_queries = make_refined_queries(queries, responses, results, failed)

    return refined_queries


def make_refine_prompts(queries):
    prompts = []
    results = []
    failed = []

    random.seed(42)

    for index, query in enumerate(queries):
        prompt, result, fail = make_refine_prompt(query, seed=index)
        prompts.append(prompt)
        results.append(result)
        failed.append(fail)

    return prompts, results, failed


def make_refine_prompt(query, seed):
    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    prompt += get_full_db_profile(query["database"], seed) + "\n"

    prompt += "\n"
    prompt += "Now consider the following SQL query:\n"
    prompt += "```sql\n"
    prompt += query["reference_sql"] + "\n"
    prompt += "```\n"

    prompt += "It produces the following result:\n"

    result, failed = execute_query(query, query["reference_sql"])
    prompt += query_results_to_string(result) + "\n"

    prompt += (
        "Your task is to improve the query so that it produces more useful results.\n"
    )

    prompt += " e.g. Fix any syntax errors.\n"
    prompt += " e.g. Remove columns that are None or NULL.\n"
    prompt += " e.g. Remove columns that are not useful.\n"

    prompt += " Your target column count is {random.randint(1, 3)}.\n"

    prompt += "Think step by step.\n"
    prompt += "First, explain what analyst questions that the query and results could answer in at most 3 sentences.\n"
    prompt += "Second, assign a score to the query and results using the following criteria.\n"
    prompt += "  1. Assign a query that fails to execute a score of 0.\n"
    prompt += "  2. Assign a query that produces an empty result a score of 1.\n"
    prompt += (
        "  2. Assign a query that produces the wrong number of columns a score of 2.\n"
    )
    prompt += (
        "  3. Assign a query that produces answers a general question a score of 3.\n"
    )
    prompt += (
        "  4. Assign a query that produces answers a specific question a score of 4.\n"
    )
    prompt += "  5. Assign a query that produces answers a specific question with a high level of detail a score of 5.\n"
    prompt += "  6. Assign a query that produces answers a specific question with a high level of detail and is easy to understand a score of 6.\n"
    prompt += (
        "Third, explain how to improve the query's score in at most 3 sentences.\n"
    )
    prompt += "Fourth, write a new and improved SQL query.\n"
    prompt += " Write the query a ```sql``` code block.\n"
    prompt += PromptTemplate().assistant()

    return prompt, result, failed


def make_refined_queries(queries, responses, results, failed):
    new_queries = []

    existing_queries = set()

    for query, response, result, failed in zip(queries, responses, results, failed):
        new_query = copy.deepcopy(query)

        add_alternate_query(
            new_query,
            query["reference_sql"],
            response,
            result,
            failed,
            query.get("score", 0),
        )

        new_query["reference_sql"] = extract_sql(response)

        new_result, new_failed = execute_query(query, new_query["reference_sql"])
        new_query["reference_result"] = new_result
        new_query["failed"] = new_failed
        new_query["refinement"] = response

        if len(new_query["reference_sql"]) == 0:
            logger.info(f"Failed to extract SQL from response: {response}")
            continue

        if new_query["reference_sql"] not in existing_queries:
            existing_queries.add(new_query["reference_sql"])
            new_queries.append(new_query)
        else:
            logger.info(f"Duplicate query found: {new_query['reference_sql']}")

    logger.info(
        f"Generated {len(new_queries)} refined queries, after removing {len(queries) - len(new_queries)} duplicates."
    )

    return new_queries


def add_alternate_query(query, sql, refinement, result, failed, score):

    if not "alternate_queries" in query:
        query["alternate_queries"] = []

    alternate_query = {
        "reference_sql": sql,
        "refinement": refinement,
        "reference_result": result,
        "failed": failed,
        "score": score,
    }

    query["alternate_queries"].append(alternate_query)
