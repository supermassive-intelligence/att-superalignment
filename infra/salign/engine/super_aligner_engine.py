from infra.salign.superalignment.explore_trajectories import explore_trajectories
from infra.salign.superalignment.gather_learnings import gather_learnings
from infra.salign.superalignment.synthesize_insights import synthesize_insights
from infra.salign.superalignment.set_score import set_score

from infra.salign.util.get_config import get_config
from infra.salign.util.get_inference_api_url import get_inference_api_url

import copy
import json
import os

from decimal import Decimal

import logging

logger = logging.getLogger(__name__)


class SuperAlignerEngine:
    def __init__(self, llm=None):
        self.database = None
        self.query_logs = []
        self.problems = []
        self.reasoners = []
        self.llm = llm
        self.alignment_prompt = ""

        if self.llm is None:
            self.llm = get_base_llm()

    def connect(self, database):
        self.database = database
        logger.info(f"Connected to database: {database}")

    def load_query_logs(self, query_logs):
        self.query_logs.extend(query_logs)
        logger.info(f"Loaded query logs of size: {len(query_logs)}")

    def load_problems(self, problems):
        self.problems.extend(problems)
        logger.info(f"Loaded problems of size: {len(problems)}")

    def learn_reasoners(self, reasoners):
        self.reasoners.extend(reasoners)
        logger.info(f"Learned reasoners: {[r.get_name() for r in reasoners]}")

    def align_prompt(self, prompt):
        self.alignment_prompt = prompt
        logger.info(f"Alignment prompt set.")

    def solve(self):

        set_score(self.problems)
        add_alignment_prompt(self.problems, self.alignment_prompt)

        max_iterations = get_max_solve_iterations()
        target_accuracy = get_target_accuracy()

        for iteration in range(max_iterations):
            logger.info(
                f"============= Superalignment Solver Iteration {iteration} ============="
            )

            llm = self.llm

            logger.info(f"Using LLM: {llm}")
            
            eval_results = explore_trajectories(
                make_results(self.problems), llm, self.reasoners, seed=iteration
            )

            if len(eval_results) == 0:
                logger.error("Unable to generate eval results for this solve step!")
                continue

            logger.info(f"Eval score: {eval_results['accuracy'] * 100:.2f}%")

            #learnings = gather_learnings(eval_results, llm, seed=iteration)

            #insights = synthesize_insights(learnings, llm, seed=iteration)

            #update_problems(self.problems, insights)

            overall_accuracy = get_accuracy(self.problems)

            logger.info(f"Overall Eval score: {overall_accuracy * 100:.2f}%")

            save_results(
                self.problems,
                llm["model_name"],
                self.database["db_id"],
                f"eval_results_{iteration}.json",
            )

            if overall_accuracy >= target_accuracy:
                logger.info(
                    f"Target accuracy of {target_accuracy * 100:.2f}% reached. Stopping training."
                )
                break

        logger.info("Superalignment solver process completed.")
        logger.info(f"Final eval score: {overall_accuracy * 100:.2f}%")

        return self.problems


def get_target_accuracy():
    config = get_config()

    return config["target_accuracy"]


def get_target_query_count():
    config = get_config()

    return config["target_query_count"]


def get_max_solve_iterations():
    config = get_config()

    return config["max_solve_iterations"]


def get_max_align_iterations():
    config = get_config()

    return config["max_align_iterations"]


def save_results(results, model_name, db_name, suffix):
    path = get_config()["results_path"]

    if not os.path.exists(path):
        os.makedirs(path)

    filename = f"{model_name}_{db_name}_{suffix}"
    full_path = os.path.join(path, filename)
    
    
    with open(full_path, "w") as f:
        json.dump(results, f, indent=4, default=decimal_serializer)
    logger.info(f"Results saved to {filename}")


def decimal_serializer(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError("Type not serializable")


def get_base_llm():
    config = get_config()

    return {"model_name": None, "api_url": get_inference_api_url()}

def update_problems(problems, eval_results):
    instance_id_map = {p["instance_id"]: p for p in problems}

    for result in eval_results["results"]:
        instance_id = result["instance_id"]
        assert instance_id in instance_id_map

        problem = instance_id_map[instance_id]

        # replace keys in the problem with those from the result
        problem.update(result)


def get_accuracy(problems):
    correct_count = sum(1 for p in problems if p.get("score", 0.0) >= 1.0)
    return correct_count / len(problems) if problems else 0.0


def make_results(problems):
    results = []

    correct_count = 0

    for problem in problems:
        result = copy.deepcopy(problem)
        if "score" not in result:
            result["score"] = 0.0  # Placeholder score
        else:
            if result["score"] >= 1.0:
                correct_count += 1
        results.append(result)

    return {
        "results": results,
        "accuracy": correct_count / len(results) if results else 0.0,
    }


def add_alignment_prompt(query_logs, alignment_prompt):
    for log in query_logs:
        log["alignment_prompt"] = alignment_prompt
