from infra.salign.util.get_config import get_config

import random

import logging

logger = logging.getLogger(__name__)


def get_insights(example, seed):

    if not "insights" in example:
        return ""

    config = get_config()
    gap = config["new_insight_count"]

    sample_count = min(len(example["insights"]) - gap, config["maximum_insights"] - gap)

    prompt = "\n\n"
    prompt += "The analyst has already learned these insights about the database\n"

    sampled_insights = []
    if sample_count > 0:
        random.seed(seed)
        sampled_insights = random.sample(example["insights"][:-gap], sample_count)

    insights = example["insights"][-gap:] + sampled_insights

    for i, insight in enumerate(insights, start=1):
        prompt += f"Insight {i}:\n"
        prompt += f"```insight\n{insight}\n```\n"

    return prompt
