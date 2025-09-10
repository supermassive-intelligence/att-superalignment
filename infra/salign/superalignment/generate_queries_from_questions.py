from infra.salign.sql.extract_sql import extract_sql
from infra.salign.sql.get_full_db_profile import get_full_db_profile

from infra.salign.util.get_inference_api_url import get_inference_api_url
from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import random

import os

import logging

logger = logging.getLogger(__name__)

def generate_queries_from_questions(query_logs, questions, target_query_count):
    prompts = make_generate_queries_prompts(
        query_logs=query_logs,
        questions=questions,
        target_query_count=target_query_count,
    )

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=512)

    database = query_logs[0]["database"]

    new_queries = make_new_queries(responses, database)

    return new_queries


def make_generate_queries_prompts(query_logs, questions, target_query_count):
    config = get_config()
    query_log_sample_size = config["query_log_sample_size"] if config["query_log_sample_size"] <= len(query_logs) else len(query_logs)
    question_sample_size = config["question_sample_size"]

    random.seed(42)

    prompts = []
    for index in range(target_query_count):

        selected_query_logs = random.sample(query_logs, query_log_sample_size)
        selected_questions = random.sample(questions, question_sample_size)

        prompt = make_prompt(
            questions=selected_questions, query_logs=selected_query_logs, seed=index
        )
        prompts.append(prompt)

    return prompts


def make_prompt(questions, query_logs, seed):
    prompt = PromptTemplate().user()
    prompt += f"You are a SQL expert.\n"

    prompt += get_full_db_profile(query_logs[0]["database"], seed)

    prompt += "\n\n"

    prompt += "Now consider the following questions that have been asked by analysts about the database.\n"
    prompt += "\n\n"

    for question in questions:
        prompt += f"Question: `{question}`\n"
        prompt += "\n\n"

    prompt += (
        "Consider the following SQL queries that have been executed on this database:\n"
    )
    prompt += "\n\n"

    for query in query_logs:
        prompt += f"```sql\n{query}`\n"
        prompt += "\n\n"

    prompt += (
        f"Your task is to write SQL queries to help understand the database better.\n"
    )

    prompt += "\n\n"

    prompt += "Think step by step.\n"
    prompt += "First, explain in plain english three different types of queries that an analyst should write to understand this database in at most 3 sentences.\n"
    prompt += "Second, determine one new type of query to write next in at most 3 sentences.\n"
    prompt += "Third, write your new query in a ```sql\nYOUR_QUERY_HERE\n``` block.\n"
    prompt += (
        f" Make sure your query is valid SQL and executable against the database.\n"
    )
    prompt += PromptTemplate().assistant()

    # print(prompt)

    return prompt


def make_new_queries(responses, database):
    new_queries = []

    for response in responses:
        sql_query = extract_sql(response)
        if sql_query:
            new_queries.append(
                {
                    "reference_sql": sql_query,
                    "database": database,
                }
            )

    return new_queries
