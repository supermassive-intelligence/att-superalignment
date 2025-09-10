from infra.salign.sql.get_db_profile import get_db_profile
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.superalignment.merge_similar_reasoners import merge_similar_reasoners

from infra.salign.reasoning_prompts.learned_reasoning_prompt import LearnedReasoningPrompt

from infra.salign.util.get_inference_api_url import get_inference_api_url
from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import random
import copy

import logging

logger = logging.getLogger(__name__)


def identify_missing_reasoners(missing_skills):
    skills_with_missing_reasoners, prompts, responses = get_identify_reasoner_responses(
        missing_skills
    )

    reasoners = make_reasoners(skills_with_missing_reasoners, responses, prompts)

    merged_reasoners = merge_similar_reasoners(reasoners, missing_skills)

    return merged_reasoners


def get_identify_reasoner_responses(missing_skills):
    skills_with_missing_reasoners = identify_skills_with_missing_reasoners(
        missing_skills
    )

    if len(skills_with_missing_reasoners) == 0:
        return [], [], []

    prompts = make_identify_reasoner_prompts(skills_with_missing_reasoners)

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=1024)

    return skills_with_missing_reasoners, prompts, responses


def identify_skills_with_missing_reasoners(missing_skills):
    already_seen_skills = set()

    skills_with_missing_reasoners = []

    for skill in missing_skills:
        for missing_skill in skill["missing_skills"]:
            if missing_skill["is_new"]:
                skill_name = missing_skill["name"]

                if skill_name in already_seen_skills:
                    continue

                already_seen_skills.add(skill_name)

                skills_with_missing_reasoners.append(
                    {
                        "missing_skill": missing_skill,
                        "skill_name": skill_name,
                        "skill": skill,
                    }
                )

    return skills_with_missing_reasoners


def make_identify_reasoner_prompts(skills_with_missing_reasoners):
    prompts = []

    for skill in skills_with_missing_reasoners:
        prompt = make_identify_reasoner_prompt(skill)
        prompts.append(prompt)

    return prompts


def make_identify_reasoner_prompt(skill_description):

    skill = skill_description["skill"]

    config = get_config()
    maximum_sentences = config["maximum_sentences"]

    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    prompt += "Consider the database with schema as follows:\n"

    prompt += get_db_profile(skill)
    prompt += "\n\n"

    prompt += f"An analyst was asked to write a query to help answer this question: `{skill['question']}`\n"
    prompt += f"The analyst wrote the following incorrect query:\n"
    prompt += f"```sql\n{skill['generated_sql']}\n```\n\n"

    prompt += "When executed, it generated the following erroneous results:\n"
    prompt += f"{query_results_to_string(skill['generated_result'])}\n\n"

    prompt += "Compare this query to the correct query:\n"
    prompt += f"```sql\n{skill['reference_sql']}\n```\n\n"

    prompt += "When executed, it generated the following results:\n"
    prompt += f"{query_results_to_string(skill['reference_result'])}\n\n"

    prompt += "The results should be identical. Since they aren't, there is an error in the analyst's query.\n"
    prompt += "Consider the following explanation of the error:\n"
    prompt += f"```text\n{skill['explanation']}\n```\n\n"

    skill_description = skill_description["missing_skill"]["name"]

    prompt += "The analyst's mentor has provided the following skill that the analyst should have:\n"
    prompt += f"```text\n{skill_description}\n```\n\n"

    prompt += "Your task is to describe this skill in detail in at most {maximum_sentences} sentences.\n"
    prompt += "Format your response in a ```text\nYOUR_DESCRIPTION_HERE\n``` block.\n"

    prompt += PromptTemplate().assistant()

    return prompt


def make_reasoners(skills, responses, prompts):
    reasoners = []

    for skill, response, prompt in zip(skills, responses, prompts):
        reasoner = make_reasoner(skill, response, prompt)
        reasoners.append(reasoner)

    return reasoners


def make_reasoner(skill, response, prompt):
    skill_description = skill["skill_name"]

    logger.info(f"Creating reasoner for skill: {skill_description}")

    reasoner = LearnedReasoningPrompt(
        short_description=skill_description, long_description=get_text_block(response)
    )

    return reasoner


def get_text_block(response):
    # If there is a ```text block, grab the content
    if "```text" in response:
        response = response.split("```text")[-1]
        response = response.split("```")[0]

    return response.strip()
