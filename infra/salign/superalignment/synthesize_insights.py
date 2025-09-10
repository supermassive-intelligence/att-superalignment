from infra.salign.sql.get_db_profile import get_db_profile
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.superalignment.get_evidence import get_evidence
from infra.salign.superalignment.get_insights import get_insights
from infra.salign.superalignment.get_context import get_context

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import copy
import re

import logging

logger = logging.getLogger(__name__)

def synthesize_insights(eval_results, llm_info, seed):
    errors = [result for result in eval_results["results"] if result["score"] < 1.0]
    correct_results = [
        result for result in eval_results["results"] if result["score"] >= 1.0
    ]

    prompts = make_synthesize_insights_prompts(errors, seed)

    llm = scalarlm.SupermassiveIntelligence(api_url=llm_info["api_url"])

    responses = llm.generate(
        prompts, max_tokens=1024, model_name=llm_info["model_name"]
    )

    insights = make_insights(errors, responses, prompts)

    logger.info(
        f"Generated {len(insights)} insights from {len(errors)} errors and {len(correct_results)} correct results."
    )

    final_results = {
        "results": correct_results + insights,
        **{k: v for k, v in eval_results.items() if k != "results"},
    }

    return final_results


def make_synthesize_insights_prompts(errors, seed):
    prompts = []

    for error in errors:
        prompt = make_synthesize_insights_prompt(error, seed)
        prompts.append(prompt)

    return prompts


def make_synthesize_insights_prompt(error, seed):
    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    prompt += get_db_profile(error)
    prompt += "\n\n"

    prompt += f"An analyst was asked to write a query to help answer this question: `{error['question']}`\n"
    prompt += f"The analyst wrote the following incorrect query:\n"
    prompt += f"```sql\n{error['generated_sql']}\n```\n\n"

    prompt += "When executed, it generated the following erroneous results:\n"
    prompt += f"{query_results_to_string(error['generated_result'])}\n\n"

    prompt += "Compare these incorrect results to the correct results:\n"
    prompt += f"{query_results_to_string(error['reference_result'])}\n\n"

    prompt += "The query is incorrect because:\n"
    prompt += f"{error['explanation']}\n\n"

    prompt += get_evidence(error, seed=seed)

    prompt += get_insights(error, seed=seed)

    prompt += get_context(error, seed=seed)

    config = get_config()
    insight_count = config["new_insight_count"]

    prompt += "Your task is write down the top {insight_count} new insights that can be drawn from the above information.\n"
    prompt += "Focus on insights that would help the analyst fix the query.\n"
    prompt += "Think step by step.\n"

    maximum_sentences = config.get("maximum_sentences")

    prompt += "First, explain what new information is learned from the queries that were executed in at most {maximum_sentences} sentences.\n"
    prompt += "Second, consider how that could change any of the previous insights in at most {maximum_sentences} sentences.\n"
    prompt += "Third, explain how that new information could be used to fix the query in at most {maximum_sentences} sentences.\n"
    prompt += "Finally, write the most helpful new {insight_count} insights concisely. Use at most {maximum_sentences} sentences each if necessary.\n"
    prompt += " Write each insight in a separate ```insight\n...\n``` block.\n"
    prompt += (
        " Only write {insight_count} total insights in ```insight\n...\n``` blocks\n"
    )
    prompt += " Each new insight should be self contained.\n"
    prompt += " Each new insight should add different knowledge that is not contained in other insights.\n"
    prompt += " Include specific details, e.g. data values, column names, etc in each insight.\n"

    prompt += PromptTemplate().assistant()

    return prompt


def make_insights(errors, responses, prompts):
    insights = []

    for error, response, prompt in zip(errors, responses, prompts):
        insight = copy.deepcopy(error)
        extracted_insights = extract_insights_from_response(response)

        for extracted_insight in extracted_insights:
            add_insight(insight, extracted_insight)

        insights.append(insight)

    return insights


def extract_insights_from_response(response):
    # Insights are in ```insight\n...\n``` blocks
    pattern = r"```insight\n(.*?)\n```"

    matches = re.findall(pattern, response, re.DOTALL)

    extracted_insights = [match.strip() for match in matches if match.strip()]

    return extracted_insights


def add_insight(query, extracted_insight):
    query["insights"] = query.get("insights", [])
    query["insights"].append(extracted_insight)

    dedup(query["insights"])

    config = get_config()
    max_insights = config["maximum_insight_history"]

    if len(query["insights"]) > max_insights:
        query["insights"] = query["insights"][-max_insights:]


def dedup(insights):
    seen = set()
    unique_insights = []

    for insight in insights:
        if insight not in seen:
            seen.add(insight)
            unique_insights.append(insight)

    insights.clear()
    insights.extend(unique_insights)
