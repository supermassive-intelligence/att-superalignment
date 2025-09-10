from infra.salign.sql.get_db_profile import get_db_profile
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.util.get_inference_api_url import get_inference_api_url

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import copy
import random

import logging

logger = logging.getLogger(__name__)


def augment(dataset):
    config = get_config()

    question_variation_count = config["question_variation_count"]

    if question_variation_count > 0:
        varied_questions = write_question_variations(
            dataset, variation_count=question_variation_count
        )
        logger.info(f"Generated {len(varied_questions)} question variations")
    else:
        logger.info("No question variations requested")
        varied_questions = dataset

    return varied_questions

def write_question_variations(queries, variation_count):

    varied_questions = copy.deepcopy(queries)
    existing_questions = [[example["question"]] for example in queries]

    logger.info(f"Generating {variation_count} question variations for each SQL query.")
    for i in range(variation_count):
        prompts = make_question_variation_prompts(queries, existing_questions)

        llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

        responses = llm.generate(prompts, max_tokens=512)

        queries_with_varied_questions = copy.deepcopy(queries)

        varied_questions.extend(
            make_question_variations_from_responses(
                queries_with_varied_questions, responses
            )
        )

        logger.info(
            f"Generated {len(varied_questions)} question variations for the SQL queries."
        )

        for question, existing in zip(varied_questions, existing_questions):
            existing.append(question["question"])

    logger.info(
        f"Generated {len(varied_questions)} question variations for the SQL queries."
    )
    return varied_questions


def make_question_variation_prompts(queries, existing_questions):
    prompts = []

    random.seed(42)

    for i in range(len(queries)):
        random.shuffle(existing_questions[i])
        prompts.append(
            make_question_variation_prompt(queries[i], existing_questions[i], seed=i)
        )

    return prompts


def make_question_variation_prompt(query, existing_questions, seed):
    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    prompt += get_db_profile(query, seed=seed)

    prompt += "\n"
    prompt += "Now consider the following SQL query:\n"
    prompt += "```sql\n"
    prompt += query["reference_sql"] + "\n"
    prompt += "```\n"

    prompt += "It produces the following result:\n"

    prompt += "```result\n"
    prompt += query_results_to_string(query["reference_result"]) + "\n"
    prompt += "```\n"

    prompt += "Your task is to write a different question or different way of asking the question that could be answered by the SQL query and result above.\n"

    prompt += "You have already written the following questions:\n"
    for question in existing_questions:
        prompt += f"```question\n{question}\n```\n"

    prompt += "Think step by step.\n"
    prompt += "First, explain what the query does and what the result means in at most 3 sentences.\n"
    prompt += "Second, note any specific details, names, or numbers used by the query in at most 3 sentences.\n"
    prompt += "Third, consider different ways of phrasing the question in at most 3 sentences.\n"
    prompt += "Finally, write a different question that could be answered by the query and result.\n"
    prompt += " Write the question in plain English in a ```question\nYOUR_QUESTION_HERE\n``` block.\n"
    prompt += " Include specific details, names, or numbers from the query and result that would be necessary to write the query.\n"
    prompt += PromptTemplate().assistant()

    return prompt


def make_question_variations_from_responses(queries, responses):

    varied_questions = []

    for query, response in zip(queries, responses):
        q = copy.deepcopy(query)

        question = extract_question(response)
        q["question"] = question

        varied_questions.append(q)

    return varied_questions


def extract_question(response):
    response = response.split("```question")[-1]
    response = response.split("```")[0].strip()

    if not response:
        logger.warning("Empty question extracted from response.")

    return response
