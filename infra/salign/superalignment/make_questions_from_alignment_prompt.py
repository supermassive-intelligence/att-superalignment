from infra.salign.sql.get_full_db_profile import get_full_db_profile

from infra.salign.util.get_inference_api_url import get_inference_api_url

from infra.salign.util.get_config import get_config
from infra.salign.util.prompt_template import PromptTemplate

import random

import scalarlm


def make_questions_from_alignment_prompt(
    alignment_prompt,
    query_logs,
    target_question_count=32,
):
    prompts = make_prompts(alignment_prompt, query_logs, target_question_count)

    llm = scalarlm.SupermassiveIntelligence(api_url=get_inference_api_url())

    responses = llm.generate(prompts, max_tokens=1024)

    return make_questions(responses, prompts)


def make_prompts(alignment_prompt, query_logs, target_question_count):
    prompts = []

    config = get_config()
    query_log_sample_size = config["query_log_sample_size"] if config["query_log_sample_size"] <= len(query_logs) else len(query_logs)

    random.seed(42)

    for i in range(target_question_count):
        query_log_samples = random.sample(query_logs, query_log_sample_size)

        prompt = make_prompt(alignment_prompt, query_log_samples, seed=i)
        prompts.append(prompt)

    return prompts


def make_prompt(alignment_prompt, query_logs, seed):
    config = get_config()
    maximum_sentences = config["maximum_sentences"]

    prompt = PromptTemplate().user()
    prompt += "You are a SQL expert.\n"

    prompt += get_full_db_profile(query_logs[0]["database"], seed=seed)
    prompt += "\n\n"

    prompt += (
        "Consider the following queries that have been executed on this database:\n"
    )
    for query_log in query_logs:
        prompt += f"```sql\n{query_log['reference_sql']}\n```\n"
        # prompt += f"Result: {query_log['result']}\n\n" # TODO: execute the queries and get the result

    prompt += "Read the following instructions carefully:\n"
    prompt += alignment_prompt + "\n\n"

    prompt += f"Your task is to write a question based on the provided query log and instructions.\n"
    prompt += "Think step by step.\n"

    prompt += f"First, analyze the query log and the instructions to understand the context in at most {maximum_sentences} sentences.\n"
    prompt += "Second, generate a list of questions that are relevant to the query log and the instructions in at most {maximum_sentences} sentences.\n"
    prompt += " The questions should be diverse and cover different aspects of the query log and the instructions.\n"
    prompt += (
        "Finally, select the most relevant question from the list and return it.\n"
    )
    prompt += "Format your final question in a ```question\n<your question here>\n``` block.\n"

    prompt += PromptTemplate().assistant()

    return prompt


def make_questions(responses, prompts):
    questions = []

    for response, prompt in zip(responses, prompts):
        question = extract_question_from_response(response, prompt)
        questions.append(question)

    return questions


def extract_question_from_response(response, prompt):
    # If there is a ```question block, grab the content
    if "```question" in response:
        response = response.split("```question")[-1]
        response = response.split("```")[0]

    # Remove any leading or trailing whitespace
    response = response.strip()

    return response
