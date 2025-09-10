from infra.salign.reasoning_prompts.learned_reasoning_prompt import LearnedReasoningPrompt

from infra.salign.util.get_inference_api_url import get_inference_api_url

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import json
import copy

import logging

logger = logging.getLogger(__name__)


def merge_similar_reasoners(reasoners, skills):
    config = get_config()

    max_new_reasoners = config["max_new_reasoners"]

    if len(reasoners) <= max_new_reasoners:
        return reasoners

    prompt = make_merge_reasoners_prompt(reasoners, skills)

    response = make_merge_reasoners_op(prompt)

    merged_reasoners, reasoner_to_merged_mapping = make_merged_reasoners(
        response, skills, reasoners
    )

    update_skills(merged_reasoners, skills, reasoner_to_merged_mapping)

    return merged_reasoners


def make_merge_reasoners_prompt(reasoners, skills):
    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    prompt += "Consider the following list of skills that an analyst should have to write queries against this database.\n"
    config = get_config()
    merged_reasoner_description_limit = config["merged_reasoner_description_limit"]

    for index, reasoner in enumerate(reasoners):
        prompt += f"{index + 1}. {reasoner.short_description}\n"
        prompt += f"   {truncate(reasoner.long_description, merged_reasoner_description_limit//len(reasoners))}\n"

    prompt += "\n\n"

    max_new_reasoners = config["max_new_reasoners"]
    maximum_sentences = config["maximum_sentences"]

    prompt += "\n"

    prompt += f"Your task is to merge these skills into at most {max_new_reasoners} new skills.\n"

    prompt += "Think step by step.\n"
    prompt += f"First, explain the similarities between skills in at most {maximum_sentences} sentences.\n"
    prompt += f"Second, explain the differences between skills in at most {maximum_sentences} sentences.\n"
    prompt += f"Third, explain the {max_new_reasoners} new skills that you would create by merging the existing skills in at most {maximum_sentences} sentences.\n"
    prompt += f"Fourth, explain which original skills belong to each new skill in at most {maximum_sentences} sentences.\n"
    prompt += "Finally, write your new skills in the following format:\n"
    prompt += " write it as a ```json``` block\n"
    prompt += (
        " The json block should include a list of dicts, one dict for each new skill.\n"
    )
    prompt += " Each dict should have three keys: `short_description`, `long_description`, and `reasoners`.\n"
    prompt += " `reasoners` should be a list of integers, each integer representing the index of an original skill that belongs to the new skill.\n"
    prompt += PromptTemplate().assistant()

    return prompt


def truncate(text, max_length):
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length] + "..."


def make_merge_reasoners_op(prompt):
    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    response = llm.generate([prompt], max_tokens=1024)

    return response[0]


def make_merged_reasoners(response, skills, reasoners):
    try:
        json_response = response.split("```json")[1].split("```")[0].strip()
        merged_reasoners = json.loads(json_response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}")
        logger.error(f"Response was: {response}")
        return reasoners, {
            reasoner.get_name(): reasoner.get_name() for reasoner in reasoners
        }

    reasoner_to_merged_mapping = {}

    for index, reasoner in enumerate(reasoners):
        name = reasoner.get_name()

        # Find the index in the merged reasoners
        for merged_index, merged_reasoner in enumerate(merged_reasoners):
            if index in merged_reasoner["reasoners"]:
                reasoner_to_merged_mapping[name] = merged_index
                break

        # If not found, create a new merged reasoner for this reasoner
        if name not in reasoner_to_merged_mapping:
            logger.warning(
                f"Reasoner {name} not found in merged reasoners, creating a new one."
            )
            reasoner_to_merged_mapping[name] = len(merged_reasoners)
            merged_reasoners.append(
                {
                    "short_description": reasoner.short_description,
                    "long_description": reasoner.long_description,
                    "reasoners": [index],
                }
            )

    # Convert merged reasoners to Reasoner objects
    merged_reasoners_objects = []

    for merged in merged_reasoners:
        new_reasoner = LearnedReasoningPrompt(
            short_description=merged["short_description"],
            long_description=merged["long_description"],
        )

        logger.info(f"Creating new merged reasoner: {new_reasoner.get_name()}")

        merged_reasoners_objects.append(new_reasoner)

    return merged_reasoners_objects, reasoner_to_merged_mapping


def update_skills(merged_reasoners, skills, reasoner_to_merged_mapping):
    for skill in skills:
        missing_skills = []
        for reasoner in skill["missing_skills"]:
            if not reasoner["is_new"]:
                continue

            merged_index = reasoner_to_merged_mapping[reasoner["name"]]
            logger.info(
                f"Merging reasoner {reasoner['name']} into merged index "
                f"{merged_index} {merged_reasoners[merged_index].get_name()}"
            )
            merged_reasoner = copy.deepcopy(reasoner)
            merged_reasoner["name"] = merged_reasoners[merged_index].get_name()
            missing_skills.append(merged_reasoner)

        skill["missing_skills"] = missing_skills

    logger.info("Updated skills with merged reasoners.")
