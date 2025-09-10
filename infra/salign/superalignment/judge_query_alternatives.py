from infra.salign.sql.get_full_db_profile import get_full_db_profile
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.util.get_inference_api_url import get_inference_api_url

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import copy
import json

import logging

logger = logging.getLogger(__name__)


def judge_query_alternatives(queries):
    logger.debug(f"Judging {len(queries)} query alternatives")

    prompts = make_judge_prompts(queries)

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=1024)

    judged_queries = make_judged_queries(queries, responses)

    return judged_queries


def make_judge_prompts(queries):
    prompts = []

    for index, query in enumerate(queries):
        prompt = make_judge_prompt(query, seed=index)
        prompts.append(prompt)

    return prompts


def make_judge_prompt(query, seed):
    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    prompt += get_full_db_profile(query["database"], seed=seed)
    prompt += "\n\n"

    queries = get_query_list(query)

    for i, q in enumerate(queries, start=1):
        prompt += f"Query {i}:\n"
        prompt += f"```sql\n{q['reference_sql']}\n```\n"
        prompt += f"Result: {query_results_to_string(q['reference_result'])}\n\n"

    prompt += "Your task is to assign a score to each query.\n"

    prompt += "Think step by step.\n"

    config = get_config()
    maximum_sentences = config["maximum_sentences"]

    prompt += f"First, explain the most important aspects of these queries in at most {maximum_sentences} sentences.\n"
    prompt += f"Second, explain the key differences between the queries in at most {maximum_sentences} sentences.\n"
    prompt += "Third, assign a score to each query based the criteria below.\n"
    prompt += " Consider the following criteria when selecting the query:\n"
    prompt += "  0. Assign a query that fails to execute a score of 0.\n"
    prompt += "  1. Assign a query that produces an empty result a score of 1.\n"
    prompt += (
        "  2. Assign a query that produces answers a general question a score of 2.\n"
    )
    prompt += (
        "  3. Assign a query that produces answers a specific question a score of 3.\n"
    )
    prompt += "  4. Assign a query that produces answers a specific question with a high level of detail a score of 4.\n"
    prompt += "  5. Assign a query that produces answers a specific question with a high level of detail and is easy to understand a score of 5.\n"
    prompt += " Format your final answer as a valid JSON array of scores, one for each query, in a ```json\n[<score1>, <score2>, ...]\n``` block.\n"

    prompt += PromptTemplate().assistant()

    return prompt


def get_query_list(query):
    return [query] + query["alternate_queries"]


def make_judged_queries(queries, responses):
    judged_queries = []

    for query, response in zip(queries, responses):
        judged_query = copy.deepcopy(query)

        scores = extract_scores_from_response(response)

        judged_query["score"] = scores[0] if scores else 0

        for alternative_query, score in zip(query["alternate_queries"], scores[1:]):
            alternative_query["score"] = score

        judged_queries.append(judged_query)

    return judged_queries


def extract_scores_from_response(response):
    try:
        response = response.split("```json")[-1]
        response = response.split("```")[0].strip()
        scores = json.loads(response)
        return scores
    except Exception as e:
        logger.error(f"Error extracting scores from response: {e}")
        return []
