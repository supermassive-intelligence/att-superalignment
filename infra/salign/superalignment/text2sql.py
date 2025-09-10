from infra.salign.sql.execute_query import execute_query
from infra.salign.sql.extract_sql import extract_sql
from infra.salign.sql.extract_reasoning import extract_reasoning
from infra.salign.sql.get_db_profile import get_db_profile

from infra.salign.util.prompt_template import PromptTemplate

import scalarlm

import pandas as pd

import copy
import math

import logging

logger = logging.getLogger(__name__)


def text2sql(examples, model_name=None, api_url=None, seed=42):

    prompts = make_text2sql_prompts(examples, seed=seed)

    llm = scalarlm.SupermassiveIntelligence(api_url=api_url)

    responses = llm.generate(
        prompts,
        max_tokens=512,
        model_name=model_name,
    )

    results = make_results(examples, responses)

    results = add_metrics(results)

    return results


def make_text2sql_prompts(examples, seed=42):

    prompts = []

    for example in examples:
        prompt = make_prompt(example, seed=seed)
        prompts.append(prompt)

    return prompts


def make_prompt(example, seed=42):

    prompt = PromptTemplate().user()

    prompt += get_db_profile(example, seed=seed)

    prompt += "\n\n"

    prompt += f"Generate a SQL query to answer this question: `{example['question']}`\n"
    prompt += "Write the SQL query in a ```sql block.\n"
    prompt += PromptTemplate().assistant()

    return prompt


def make_results(examples, responses):
    results = []

    for example, response in zip(examples, responses):
        result = make_result(example, response)
        results.append(result)

    return results


def make_result(example, response):

    result = copy.deepcopy(example)

    result["generated_sql"] = extract_sql(response)
    result["reasoning"] = extract_reasoning(response)

    generated_result, generated_failed = execute_query(result, result["generated_sql"])

    result["generated_result"] = generated_result
    result["generated_failed"] = generated_failed

    return result


def add_metrics(results):
    results_with_metrics = []
    for result in results:
        result_with_metric = copy.deepcopy(result)
        add_metrics_to_result(result_with_metric)
        results_with_metrics.append(result_with_metric)

    return results_with_metrics


def add_metrics_to_result(result):
    metrics = query_result_match(result)

    result.update(metrics)


def query_result_match(example):
    results = {}

    generated_result = example["generated_result"]

    compute_score(results, example, generated_result)

    return results


def compute_score(results, example, generated_result):
    # Case 1: reference SQL exists, execute it
    if "reference_sql" in example:
        reference_result, reference_failed = execute_query(example, example["reference_sql"])
        results["reference_result"] = reference_result
        results["reference_failed"] = reference_failed

        match = not reference_failed and compare_results(reference_result, generated_result, example)
        results["score"] = 1.0 if match else 0.0
        return

    # Case 2: use provided reference results
    possible_refs = example.get("reference_results", [example["reference_result"]])

    for ref in possible_refs:
        if compare_results(ref, generated_result, example):
            results.update({"score": 1.0, "reference_result": ref})
            return

    results["score"] = 0.0
    results["reference_result"] = example["reference_result"]


def compare_results(reference_result, generated_result, example):
    if isinstance(reference_result, str) or isinstance(generated_result, str):
        # If either result is a string, it means the query failed
        return False

    reference_dataframe = pd.DataFrame(reference_result)
    generated_dataframe = pd.DataFrame(generated_result)

    condition_cols = []
    ignore_order = True

    if "eval_criteria" in example:
        condition_cols = example["eval_criteria"]["condition_cols"]
        ignore_order = example["eval_criteria"]["ignore_order"]

    return compare_pandas_table(
        generated_dataframe,
        reference_dataframe,
        condition_cols=condition_cols,
        ignore_order=ignore_order,
    )


def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
    """_summary_

    Args:
        pred (Dataframe): _description_
        gold (Dataframe): _description_
        condition_cols (list, optional): _description_. Defaults to [].
        ignore_order (bool, optional): _description_. Defaults to False.

    """

    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        try:
            if ignore_order_:
                v1, v2 = (
                    sorted(
                        v1,
                        key=lambda x: (x is None, str(x), isinstance(x, (int, float))),
                    ),
                    sorted(
                        v2,
                        key=lambda x: (x is None, str(x), isinstance(x, (int, float))),
                    ),
                )
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if pd.isna(a) and pd.isna(b):
                    continue
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if not math.isclose(float(a), float(b), abs_tol=tol):
                        return False
                elif a != b:
                    return False
            return True
        except Exception as e:
            return False

    score = 0
    try: 
        if condition_cols != []:
            gold_cols = gold.iloc[:, condition_cols]
        else:
            gold_cols = gold
        pred_cols = pred

        t_gold_list = gold_cols.transpose().values.tolist()
        t_pred_list = pred_cols.transpose().values.tolist()
        score = 1
        for _, gold in enumerate(t_gold_list):
            if not any(
                vectors_match(gold, pred, ignore_order_=ignore_order)
                for pred in t_pred_list
            ):
                score = 0
            else:
                for j, pred in enumerate(t_pred_list):
                    if vectors_match(gold, pred, ignore_order_=ignore_order):
                        break
    except Exception as e:
        pass

    return score
