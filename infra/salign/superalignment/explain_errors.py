from infra.salign.sql.get_db_profile import get_db_profile
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.superalignment.get_insights import get_insights
from infra.salign.superalignment.get_context import get_context

from infra.salign.util.get_inference_api_url import get_inference_api_url
from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm
import copy


def explain_errors(results, seed):
    errors = [result for result in results["results"] if result["score"] < 1.0]

    prompts = make_explanation_prompts(errors, seed)

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=1024)

    explanations = make_explanations(errors, responses, prompts)

    return explanations


def make_explanation_prompts(errors, seed):
    prompts = []

    for error in errors:
        prompt = make_explanation_prompt(error, seed)
        prompts.append(prompt)

    return prompts


def make_explanation_prompt(error, seed):
    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    table_prompt = get_db_profile(error)

    prompt += table_prompt
    prompt += "\n\n"

    prompt += get_insights(error, seed=seed)

    prompt += get_context(error, seed=seed)

    prompt += f"An analyst was asked to write a query to help answer this question: `{error['question']}`\n"
    prompt += f"The analyst wrote the following incorrect query:\n"
    prompt += f"```sql\n{error['generated_sql']}\n```\n\n"

    prompt += "When executed, it generated the following erroneous results:\n"
    prompt += f"{query_results_to_string(error['generated_result'])}\n\n"

    if "reference_sql" in error:
        prompt += "Compare this query to the correct query:\n"
        prompt += f"```sql\n{error['reference_sql']}\n```\n\n"

        prompt += "When executed, it generated the following results:\n"
        prompt += f"{query_results_to_string(error['reference_result'])}\n\n"

    else:
        prompt += "Compare these incorrect results to the correct results:\n"
        prompt += f"{query_results_to_string(error['reference_result'])}\n\n"

    prompt += "Your task is to explain why the first query is incorrect, using the reference to help you.\n"
    prompt += "The results should be identical. Since they aren't, there is an error in the analyst's query.\n"
    prompt += "Think step by step.\n"

    config = get_config()
    maximum_sentences = config.get("maximum_sentences")

    prompt += f"First, describe the differences in results in at most {maximum_sentences} sentences.\n"
    prompt += f"Second, explain the errors in the analyst's query that result in the differences in at most {maximum_sentences} sentences.\n"

    prompt += PromptTemplate().assistant()

    return prompt


def make_explanations(errors, responses, prompts):

    config = get_config()
    maximum_trajectory_history = config.get("maximum_trajectory_history")

    explanations = []

    for error, response, prompt in zip(errors, responses, prompts):
        explanation = copy.deepcopy(error)
        explanation["explanation"] = response

        explanation["trajectory"].append(prompt)
        explanation["trajectory"].append(explanation["explanation"])

        if len(explanation["trajectory"]) > maximum_trajectory_history:
            explanation["trajectory"] = explanation["trajectory"][
                -maximum_trajectory_history:
            ]

        explanations.append(explanation)

    return explanations
