from infra.salign.sql.get_db_profile import get_db_profile

from infra.salign.superalignment.write_questions import write_questions

from infra.salign.util.get_inference_api_url import get_inference_api_url

import scalarlm

import logging

logger = logging.getLogger(__name__)


def update_reasoner_training_data(reasoners, missing_skills):
    reasoner_skill_pairs = get_reasoner_skill_pairs(reasoners, missing_skills)

    prompts = get_backward_prompts(reasoner_skill_pairs)

    responses = update_reasoner_training_data_operation(prompts)

    training_examples = update_reasoner_data(reasoner_skill_pairs, responses)

    training_examples_with_questions = write_questions(training_examples)

    trajectory_prompts = get_trajectory_prompts(
        reasoner_skill_pairs, training_examples_with_questions
    )

    trajectory_responses = update_reasoner_trajectory_data_operation(trajectory_prompts)

    update_reasoner_trajectory_data(
        reasoner_skill_pairs, training_examples_with_questions, trajectory_responses
    )


def update_reasoner_training_data_operation(prompts):

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=1024)

    return responses


def get_reasoner_skill_pairs(reasoners, missing_skills):
    reasoner_map = {reasoner.get_name(): reasoner for reasoner in reasoners}

    logger.info(f"Reasoners: {list(reasoner_map.keys())}")

    reasoner_skill_pairs = []

    for missing_skill in missing_skills:
        for skill in missing_skill["missing_skills"]:
            reasoner_name = skill["name"]

            reasoner = reasoner_map[reasoner_name]

            reasoner_skill_pairs.append(
                {
                    "reasoner": reasoner,
                    "missing_skill": missing_skill,
                }
            )

    return reasoner_skill_pairs


def get_backward_prompts(reasoner_skill_pairs):

    prompts = []

    for index, pair in enumerate(reasoner_skill_pairs):
        reasoner = pair["reasoner"]
        missing_skill = pair["missing_skill"]

        prompt = reasoner.backward_prompt(missing_skill, seed=index)

        prompts.append(prompt)

    return prompts


def update_reasoner_data(reasoner_skill_pairs, responses):
    training_examples = []

    for pair, response in zip(reasoner_skill_pairs, responses):
        reasoner = pair["reasoner"]
        missing_skill = pair["missing_skill"]

        training_example = reasoner.update_training_data(missing_skill, response)

        training_examples.append(training_example)

    return training_examples


def get_trajectory_prompts(reasoner_skill_pairs, training_examples_with_questions):
    prompts = []
    index = 0

    for pair, example in zip(reasoner_skill_pairs, training_examples_with_questions):
        reasoner = pair["reasoner"]
        missing_skill = pair["missing_skill"]

        prompt = reasoner.trajectory_prompt(example, seed=index)

        prompts.append(prompt)
        index += 1

    return prompts


def update_reasoner_trajectory_data_operation(prompts):

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=1024)

    return responses


def update_reasoner_trajectory_data(
    reasoner_skill_pairs, training_examples_with_questions, responses
):
    for pair, example, response in zip(
        reasoner_skill_pairs, training_examples_with_questions, responses
    ):
        reasoner = pair["reasoner"]
        missing_skill = pair["missing_skill"]

        reasoner.update_trajectory_data(example, response)
