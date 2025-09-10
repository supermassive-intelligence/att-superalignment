from infra.salign.sql.get_db_profile import get_db_profile
from infra.salign.sql.query_results_to_string import query_results_to_string

from infra.salign.superalignment.get_insights import get_insights
from infra.salign.superalignment.get_evidence import get_evidence

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import re
import copy

def write_queries_to_research_dataset(results, llm_info, seed):

    prompts = make_research_prompts(results, seed)

    llm = scalarlm.SupermassiveIntelligence(api_url=llm_info["api_url"])

    responses = llm.generate(
        prompts, max_tokens=1024, model_name=llm_info["model_name"]
    )

    queries = make_queries(results, responses, prompts)

    return queries


def make_research_prompts(errors, seed):
    prompts = []

    for error in errors:
        prompt = make_research_prompt(error, seed)
        prompts.append(prompt)

    return prompts


def make_research_prompt(error, seed):
    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    table_prompt = get_db_profile(error)

    prompt += table_prompt
    prompt += "\n\n"

    prompt += get_evidence(error, seed=seed)

    prompt += get_insights(error, seed=seed)

    prompt += f"An analyst was asked to write a query to help answer this question: `{error['question']}`\n"
    prompt += f"The analyst wrote the following incorrect query:\n"
    prompt += f"```sql\n{error['generated_sql']}\n```\n\n"

    prompt += "When executed, it generated the following erroneous results:\n"
    prompt += f"{query_results_to_string(error['generated_result'])}\n\n"

    prompt += "The query is incorrect because:\n"
    prompt += f"{error['explanation']}\n\n"

    prompt += "Your task is to write a few queries to execute against the "
    prompt += (
        "database to get any missing required information that would be necessary "
    )
    prompt += "to write the correct query.\n"

    prompt += "Think step by step.\n"

    config = get_config()

    maximum_sentences = config["maximum_sentences"]
    maximum_queries = config["maximum_research_queries"]

    prompt += f"First, explain what new information would be most helpful to gather in at most {maximum_sentences} sentences.\n"
    prompt += f"Then, write at most {maximum_queries} new and different queries to gather that information.\n"
    prompt += " write the queries, each in a separate ```sql\n`` block.\n"
    prompt += " make sure that the number of results returned by each query is small enough to be manageable, e.g. less than 20 rows.\n"
    prompt += PromptTemplate().assistant()

    return prompt


def make_queries(results, responses, prompts):
    queries = []

    for error, response, prompt in zip(results, responses, prompts):
        exracted_queries = extract_queries_from_response(response)
        for extracted_query in exracted_queries:
            query = copy.deepcopy(error)
            add_evidence(query, extracted_query)
            queries.append(query)

    return queries


def extract_queries_from_response(response):
    # Queries are in ```sql\n ... \n``` blocks
    pattern = r"```sql\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)

    extracted_queries = [match.strip() for match in matches if match.strip()]

    return extracted_queries


def add_evidence(query, extracted_query):
    query["evidence"] = query.get("evidence", [])
    query["evidence"].append({"sql": extracted_query})
