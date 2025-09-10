from infra.salign.superalignment.add_reasoning_trajectories import add_reasoning_trajectories

import copy

import logging

logger = logging.getLogger(__name__)


def get_perfect_matches(explored_trajectories, inference_results):
    perfect_matches = []

    for result in explored_trajectories["results"]:
        if result["score"] >= 1.0:
            perfect_match = copy.deepcopy(result)
            perfect_match["reference_sql"] = result["generated_sql"]
            perfect_matches.append(perfect_match)

    for result in inference_results["results"]:
        if result["score"] >= 1.0:
            perfect_match = copy.deepcopy(result)
            perfect_match["reference_sql"] = result["generated_sql"]
            perfect_matches.append(perfect_match)

    training_dataset = perfect_matches
    logger.info(f"Linked schema for {len(training_dataset)} perfect matches")

    training_dataset = add_reasoning_trajectories(training_dataset)
    logger.info(
        f"Added reasoning trajectories to {len(training_dataset)} perfect matches"
    )

    return training_dataset
