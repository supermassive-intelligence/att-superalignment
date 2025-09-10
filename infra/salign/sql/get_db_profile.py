from infra.salign.util.get_config import get_config

import logging

logger = logging.getLogger(__name__)


def get_db_profile(example, seed=42):
    assert "db_profile" in example, "Example must contain 'db_profile' key"

    descriptions = list(example["db_profile"]["column_descriptions"])

    config = get_config()

    max_db_profile_length = config["db_profile_max_length"]

    prompt = "Consider the following database profile of some relevant columns:\n\n"
    prompt += "\nThe format is `Column: Table.Column`\n\n"

    total_selected_columns = 0
    for d in descriptions:
        description = d["profile"]
        if len(prompt) + len(description) + 2 > max_db_profile_length:
            break
        prompt += description + "\n"
        total_selected_columns += 1

    if len(descriptions) > total_selected_columns:
        logger.warning(
            f"Truncated data profile to {total_selected_columns} out of {len(descriptions)} columns."
        )

    selected_columns_per_table = {}

    total_selected_columns = 0
    create_table_max_length = config["create_table_max_length"]

    for d in descriptions:
        table = d["column"]["table"]
        column = d["column"]["column"]["column"]

        if table not in selected_columns_per_table:
            selected_columns_per_table[table] = []

        selected_columns_per_table[table].append(column)

        if (
            len(get_create_table_statements(selected_columns_per_table))
            > create_table_max_length
        ):
            break

        total_selected_columns += 1

    prompt += get_create_table_statements(selected_columns_per_table)

    if len(descriptions) > total_selected_columns:
        logger.warning(
            f"Truncated create table statements to {total_selected_columns} out of {len(descriptions)} columns."
        )

    return prompt


def get_create_table_statements(selected_columns_per_table):
    create_table_statements = []

    for table, columns in selected_columns_per_table.items():
        create_table_statement = f"CREATE TABLE {table} ({', '.join(columns)});"
        create_table_statements.append(create_table_statement)

    prompt = "\n\n" + "\n".join(create_table_statements)

    return prompt
