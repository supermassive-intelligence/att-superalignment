from infra.salign.sql.get_db_profile import get_db_profile
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.util.get_inference_api_url import get_inference_api_url
from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import random
import copy
import json

import logging

logger = logging.getLogger(__name__)


def identify_missing_skills_from_explanations(explanations, reasoners):
    prompts, selected_reasoners = get_identify_missing_skills_prompts(
        explanations, reasoners
    )

    responses = identify_missing_skills_from_explanations_op(prompts)

    missing_skills = make_missing_skills(
        explanations, responses, prompts, selected_reasoners, reasoners
    )

    return missing_skills

def identify_missing_skills_from_explanations_op(prompts):
    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=1024)

    return responses


def get_identify_missing_skills_prompts(explanations, reasoners):
    random.seed(42)

    config = get_config()
    maximum_reasoners = config["maximum_reasoners"]

    prompts = []
    all_selected_reasoners = []

    for explanation in explanations:

        selected_reasoners = random.sample(
            reasoners, min(maximum_reasoners, len(reasoners))
        )

        prompt = get_identify_missing_skills_prompt(explanation, selected_reasoners)
        prompts.append(prompt)

        all_selected_reasoners.append(selected_reasoners)

    return prompts, all_selected_reasoners


def get_identify_missing_skills_prompt(explanation, reasoners):

    config = get_config()

    maximum_sentences = config["maximum_sentences"]

    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    prompt += "Consider the database with schema as follows:\n"

    table_prompt = get_db_profile(explanation)

    prompt += table_prompt
    prompt += "\n\n"

    if len(reasoners) > 1:
        prompt += "Consider the following skills that an analyst should have:\n"
        for index, reasoner in enumerate(reasoners):
            prompt += f" {index}. - {reasoner.get_name()}\n"

    prompt += f"An analyst was asked to write a query to help answer this question: `{explanation['question']}`\n"
    prompt += f"The analyst wrote the following incorrect query:\n"
    prompt += f"```sql\n{explanation['generated_sql']}\n```\n\n"

    prompt += "When executed, it generated the following erroneous results:\n"
    prompt += f"{query_results_to_string(explanation['generated_result'])}\n\n"

    prompt += "Compare this query to the correct query:\n"
    prompt += f"```sql\n{explanation['reference_sql']}\n```\n\n"

    prompt += "When executed, it generated the following results:\n"
    prompt += f"{query_results_to_string(explanation['reference_result'])}\n\n"

    prompt += "The results should be identical. Since they aren't, there is an error in the analyst's query.\n"
    prompt += "Consider the following explanation of the error:\n"
    prompt += f"```text\n{explanation['explanation']}\n```\n\n"

    prompt += "Your task is to identify which skills the analyst is missing that would have allowed them to avoid making this error.\n"
    prompt += "Think step by step.\n"

    if len(reasoners) > 1:
        prompt += "First, consider how each of the listed skills could help the analyst avoid this error in at most {maximum_sentences} sentences.\n"
        prompt += "Second, identify the missing skills (from the list above, or other essential skills) that the analyst should have to avoid making these errors.\n"
        prompt += " Format the final list of skills as a JSON dict, with each skill as a string and the corresponding index as the key.\n"
        prompt += " If the required skill is not listed above, assign it the index of the last skill plus one.\n"
    else:
        prompt += "First, Identify the missing skills that the analyst should have to avoid making these errors in at most {maximum_sentences} sentences.\n"
        prompt += " Format the final list of skills as a JSON dict, with each skill as a string and the corresponding index as the key.\n"

    prompt += "For example, if the analyst should have skills in SQL joins and subqueries, the output should be:\n"
    prompt += '```json\n{"0": "SQL joins", "1": "subqueries"}\n```\n\n'
    prompt += PromptTemplate().assistant()

    return prompt


def make_missing_skills(
    explanations, responses, prompts, all_selected_reasoners, reasoners
):
    missing_skills = []

    for explanation, response, prompt, selected_reasoners in zip(
        explanations, responses, prompts, all_selected_reasoners
    ):
        missing_skill = copy.deepcopy(explanation)

        skills = parse_missing_skills(response, reasoners, selected_reasoners)

        missing_skill["missing_skills"] = skills

        missing_skill["trajectory"].append(prompt)
        missing_skill["trajectory"].append(response)

        missing_skills.append(missing_skill)

    return missing_skills


def parse_missing_skills(response, reasoners, selected_reasoners):
    json_string = extract_json_string(response)

    config = get_config()

    maximum_reasoners = config["maximum_reasoners"]

    try:
        json_object = json.loads(json_string)
    except json.JSONDecodeError:
        logger.warning(f"Failed to decode JSON from response: {response}")
        json_object = {}

    skills = []

    for index, skill in sorted(json_object.items()):
        index = int(index)
        reasoner, is_new = get_reasoner(index, skill, reasoners, selected_reasoners)

        logger.debug(
            f"Identified skill: {skill} (Reasoner: {reasoner}, Is new: {is_new})"
        )

        skills.append(
            {
                "name": reasoner,
                "is_new": is_new,
            }
        )

    return skills


def get_reasoner(index, skill_name, reasoners, selected_reasoners):

    config = get_config()
    available_reasoners = min(len(reasoners), config["maximum_reasoners"])

    if index >= available_reasoners:
        return skill_name, True

    selected_reasoner = selected_reasoners[index]

    return selected_reasoner.get_name(), False


def extract_json_string(response):
    # If there is a ```json block, grab the content
    if "```json" in response:
        response = response.split("```json")[-1]
        response = response.split("```")[0]

    return response.strip()
