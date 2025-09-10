from infra.salign.util.get_db_adapter_from_config import get_db_adapter_from_config

def get_columns(database):
    db_adapter = get_db_adapter_from_config(database)
    table_info = db_adapter.get_table_info(database)

    column_info = []

    for table, columns in sorted(table_info.items()):
        for column in columns:
            column_info.append({"table": table, "column": column})

    return column_info
