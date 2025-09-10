from infra.salign.superalignment.explain_errors import explain_errors
from infra.salign.superalignment.identify_missing_reasoners import identify_missing_reasoners
from infra.salign.superalignment.identify_missing_skills_from_explanations import (
    identify_missing_skills_from_explanations,
)
from infra.salign.superalignment.update_reasoner_training_data import (
    update_reasoner_training_data,
)

import logging

logger = logging.getLogger(__name__)


def update_reasoners(reasoners, eval_results, seed):
    eval_explanations = explain_errors(eval_results, seed=seed)

    missing_skills = identify_missing_skills_from_explanations(
        eval_explanations, reasoners
    )

    new_reasoners = identify_missing_reasoners(missing_skills)

    logger.info(
        f"Identified {len(new_reasoners)} new reasoners based on missing skills"
    )

    reasoners.extend(new_reasoners)

    update_reasoner_training_data(reasoners, missing_skills)

    return reasoners
