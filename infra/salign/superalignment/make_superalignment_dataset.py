from infra.salign.superalignment.make_questions_from_alignment_prompt import (
    make_questions_from_alignment_prompt,
)
from infra.salign.superalignment.generate_queries_from_questions import (
    generate_queries_from_questions,
)
from infra.salign.superalignment.refine_queries_with_results import (
    refine_queries_with_results,
)
from infra.salign.superalignment.write_questions import write_questions
from infra.salign.superalignment.split_dataset import split_dataset
from infra.salign.superalignment.add_reasoning_trajectories import add_reasoning_trajectories
from infra.salign.superalignment.set_score import set_score

from infra.salign.util.get_config import get_config

import logging

logger = logging.getLogger(__name__)


def make_superalignment_dataset(
    query_logs, alignment_prompt=None, target_query_count=32
):

    questions = make_questions_from_alignment_prompt(
        alignment_prompt=alignment_prompt,
        query_logs=query_logs,
        target_question_count=target_query_count,
    )
    logger.info(f"Generated {len(questions)} questions from alignment prompt")

    queries = generate_queries_from_questions(
        query_logs=query_logs,
        questions=questions,
        target_query_count=target_query_count,
    )
    logger.info(f"Generated {len(queries)} new queries")

    refined_queries = refine_queries_with_results(queries)
    logger.info(f"Refined {len(refined_queries)} new queries")

    all_queries = query_logs + refined_queries
    logger.info(f"Total queries after refinement: {len(all_queries)}")

    initial_dataset = write_questions(all_queries)
    logger.info(f"Generated {len(initial_dataset)} initial dataset entries")

    set_score(initial_dataset)

    min_test_samples = get_config()["min_test_samples"]

    training_dataset, eval_dataset = split_dataset(
        initial_dataset, min_test_samples=min_test_samples
    )

    logger.info(f"Linked schema for {len(training_dataset)} training samples")

    training_dataset = add_reasoning_trajectories(training_dataset)
    logger.info(
        f"Added reasoning trajectories to {len(training_dataset)} training samples"
    )

    logger.info(
        f"Split dataset into {len(training_dataset)} training samples and {len(eval_dataset)} evaluation samples"
    )

    return training_dataset, eval_dataset
