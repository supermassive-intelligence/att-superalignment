from infra.salign.util.get_config import get_config

import logging

logger = logging.getLogger(__name__)


def get_context(example, seed):

    if not "context" in example:
        return ""

    config = get_config()
    max_context_length = config["max_context_length"]

    context = example["context"]

    if len(context) > max_context_length:
        logger.warning(
            f"Context length {len(context)} exceeds max length {max_context_length}. Truncating context."
        )
        context = context[:max_context_length]

    return context + "\n\n"
