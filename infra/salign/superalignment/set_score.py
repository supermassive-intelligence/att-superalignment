import logging

logger = logging.getLogger(__name__)


def set_score(dataset):
    """
    Set the score for each entry in the dataset.
    """
    for entry in dataset:
        if "score" not in entry:
            entry["score"] = 0.0

    logger.info("Set default score for all dataset entries")
