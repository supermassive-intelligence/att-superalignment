from infra.salign import SuperAligner
from infra.salign.reasoning_prompts.english_reasoning_prompt import EnglishReasoningPrompt

import json
import logging
import os

logger = logging.getLogger(__name__)

def run():
    solve()

def solve():
    setup_logging()
    
    db_name = "MGO"

    saligner = SuperAligner(llm=load_llm(db_name))
    saligner.connect(load_database(db_name))
    saligner.align_prompt(get_alignment_prompt())
    saligner.load_problems(load_problems(db_name))
    saligner.learn_reasoners(load_reasoners())

    model = saligner.solve()

    logger.info("Solve completed successfully.")
    save_llm(model, "MGO")


def load_reasoners():
    return [EnglishReasoningPrompt()]

def get_alignment_prompt():

    prompt = """
You are an expert database analyst with 20 years of experience writing SQL queries.
"""

    return prompt


def load_database(db_id):
    return {
        "type": "snowflake",
        "credential_path": "sdk/mgo/snowflake_credential.json",
        "db_id": db_id,
    }

def load_query_logs(db_name):

    path = f"infra/salign/data/mgo/solutions/{db_name}.json"

    if os.path.exists(path):

        with open(path, "r") as file:
            data = json.load(file)

        return [
            {
                "reference_sql": item["reference_sql"] if "reference_sql" in item else item["generated_sql"],
                "database": load_database(db_name),
            }
            for item in data
            if item["score"] >= 1.0
        ]

def setup_logging():
    logging.basicConfig(level=logging.INFO)


def load_problems(db_name):
    
    return [{"question": "What is the capital of Qatar?", 
             "reference_sql": "select * from table mgo", 
             "database": load_database(db_name)}]

def load_llm(db_name):
    model_path = f"infra/salign/data/mgo/llm/{db_name}.json"

    model_config = None

    if os.path.exists(model_path):
        logger.info(f"Loading LLM model configuration from {model_path}...")

        with open(model_path, "r") as file:
            model_config = json.load(file)
    else:
        logger.info(
            f"LLM model configuration not found at {model_path}. Using base model."
        )

    return model_config

def save_llm(model_config, db_name):
    model_path = f"infra/salign/data/mgo/llm/{db_name}.json"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        with open(model_path, "w") as file:
            json.dump(model_config, file, indent=4)
        logger.info(f"Saved LLM model configuration to {model_path}.")
    except Exception as e:
        logger.error(f"Failed to save LLM model configuration to {model_path}: {e}")