from infra.salign.sql.get_db_profile import get_db_profile
from infra.salign.sql.execute_query import execute_query
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.util.get_inference_api_url import get_inference_api_url
from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import copy

import logging

logger = logging.getLogger(__name__)

def add_reasoning_trajectories(original_dataset):
    dataset = add_results(original_dataset)

    prompts = make_reasoning_prompts(dataset)

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=512)

    trajectories = make_new_trajectories(dataset, responses)

    return trajectories


def add_results(original_dataset):
    dataset = []

    logger.info(f"Adding results to {len(original_dataset)} examples.")

    for example in original_dataset:
        data = copy.deepcopy(example)
        reference_result, reference_failed = execute_query(
            example, example["reference_sql"]
        )

        data["reference_result"] = reference_result
        data["reference_failed"] = reference_failed

        dataset.append(data)

    return dataset


def make_reasoning_prompts(dataset):
    prompts = []

    for index, data in enumerate(dataset):
        prompt = make_reasoning_prompt(data)
        prompts.append(prompt)

    return prompts


def make_reasoning_prompt(example):
    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    prompt += get_db_profile(example)
    prompt += "\n\n"

    prompt += f"An analyst was asked to write a query to help answer this question: `{example['question']}`\n"
    prompt += f"The analyst wrote the following correct query:\n"
    prompt += f"```sql\n{example['reference_sql']}\n```\n\n"

    prompt += "When executed, it generated the following results:\n"
    prompt += f"{query_results_to_string(example['reference_result'])}\n\n"

    prompt += "Your task is to explain how to write this query in plain english.\n"

    prompt += "Think step by step.\n"

    config = get_config()
    maximum_sentences = config["maximum_sentences"]

    prompt += f"First, explain how the query and results answers the question in at most {maximum_sentences} sentences.\n"
    prompt += f"Second, explain to write this query in plain english in at most {maximum_sentences} sentences.\n"
    prompt += "Finally, write your explanation in a ```explanation\nYOU_EXPLANATION_HERE\n``` block.\n"
    prompt += PromptTemplate().assistant()

    return prompt


def make_new_trajectories(dataset, responses):
    trajectories = []

    for response, data in zip(responses, dataset):
        trajectory = copy.deepcopy(data)

        reasoning = extract_explanation(response)
        trajectory["reasoning"] = reasoning
        trajectories.append(trajectory)

    return trajectories


def extract_explanation(response):
    # If there is a ```sql block, grab the content
    if "```explanation" in response:
        response = response.split("```explanation")[-1]
        response = response.split("```")[0]

    return response.strip()


def truncate(text, max_length=100):
    if len(text) > max_length:
        truncated_text = text[:max_length]
        return truncated_text + "..."

    return text
