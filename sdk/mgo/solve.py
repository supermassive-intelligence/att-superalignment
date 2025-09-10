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
    save_llm(model, db_name)


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


def setup_logging():
    logging.basicConfig(level=logging.INFO)


def load_problems(db_name):
    
    path = f"sdk/mgo/mgo.json"

    with open(path, "r") as file:
        data = json.load(file)

    examples = []

    questions = data["Queries"]
    
    schema = format_schema(data["Schema"])
    
    for item in questions:
        example = {}
        
        example["question"] = item["Question"]
        example["reference_sql"] = item["Original Query"]
        example["db_profile"] = schema

        example["database"] = load_database(db_name)

        examples.append(example)
    
    return examples

def load_llm(db_name):
    return None

def save_llm(model_config, db_name):
    model_path = f"infra/salign/data/mgo/llm/{db_name}.json"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        with open(model_path, "w") as file:
            json.dump(model_config, file, indent=4)
        logger.info(f"Saved LLM model configuration to {model_path}.")
    except Exception as e:
        logger.error(f"Failed to save LLM model configuration to {model_path}: {e}")
        
def format_schema(schema_json):
    output = []
    for table in schema_json:
        table_str = f"TABLE: {table['table_name']}\n    COLUMNS:"
        column_lines = []
        for col in table["columns"]:
            col_desc = col.get("column_description", "")
            # Only include description if present/non-empty
            if col_desc:
                column_line = f"({col['column_name']}, {col['column_type']}, {col_desc})"
            else:
                column_line = f"({col['column_name']}, {col['column_type']})"
            column_lines.append(column_line)
        # Join columns, each on new line with indent
        table_str += "\n    " + ",\n    ".join(column_lines)
        output.append(table_str)
    return "\n\n".join(output)