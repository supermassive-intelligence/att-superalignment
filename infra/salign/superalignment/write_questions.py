from infra.salign.sql.get_full_db_profile import get_full_db_profile

from infra.salign.sql.execute_query import execute_query
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.util.get_inference_api_url import get_inference_api_url
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import copy
import re

import logging

logger = logging.getLogger(__name__)


def write_questions(queries):
    prompts, results, failed = make_question_prompts(queries)

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=1024)

    new_questions = make_new_questions(queries, responses, results, failed)

    return new_questions


def make_question_prompts(queries):
    prompts = []
    results = []
    failed = []

    for index, query in enumerate(queries):
        prompt, result, fail = make_question_prompt(query, seed=index)
        prompts.append(prompt)
        results.append(result)
        failed.append(fail)

    return prompts, results, failed


def make_question_prompt(query, seed):
    prompt = PromptTemplate().user()
    prompt += "Consider a database with the following schema:\n"

    prompt += get_full_db_profile(query["database"], seed)

    prompt += "\n"
    prompt += "Now consider the following SQL query:\n"
    prompt += "```sql\n"
    prompt += query["reference_sql"] + "\n"
    prompt += "```\n"

    prompt += "It produces the following result:\n"

    result, failed = execute_query(query, query["reference_sql"])
    prompt += query_results_to_string(result) + "\n"

    prompt += "Your task is to write a question that could be answered by the SQL query and result above.\n"

    prompt += "Think step by step.\n"
    prompt += "First, explain what the query does and what the result means in at most 3 sentences.\n"
    prompt += "Second, note any specific details, names, or numbers used by the query in at most 3 sentences.\n"
    prompt += (
        "Third, write a question that could be answered by the query and result.\n"
    )
    prompt += " Write the question in plain English in a ```question\nYOUR_QUESTION_HERE\n``` block.\n"
    prompt += " Include specific details, names, or numbers from the query and result that would be necessary to write the query.\n"
    prompt += " If the query returns multiple columns, write a question that asks for all of them.\n"
    prompt += PromptTemplate().assistant()

    return prompt, result, failed


def make_new_questions(queries, responses, results, failed):
    new_queries = []

    existing_questions = set()

    for query, response, result, failed in zip(queries, responses, results, failed):
        new_query = copy.deepcopy(query)
        new_query["question"] = extract_question(response)
        new_query["result"] = result
        new_query["failed"] = failed

        if len(new_query["question"]) == 0:
            logger.info(f"Failed to extract question from response: {response}")
            continue

        if new_query["question"] not in existing_questions:
            existing_questions.add(new_query["question"])
            new_queries.append(new_query)
        else:
            logger.info(f"Duplicate question found: {new_query['question']}")

    logger.info(
        f"Generated {len(new_queries)} new questions, after removing {len(queries) - len(new_queries)} duplicates."
    )

    return new_queries


def extract_question(response):
    match = re.search(r"```question\n(.*?)\n```", response, re.DOTALL)

    if match:
        return match.group(1)

    return ""
