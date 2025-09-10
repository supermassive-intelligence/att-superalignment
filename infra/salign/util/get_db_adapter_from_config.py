from infra.salign.sql.sqlite_adapter import SQLiteAdapter
from infra.salign.sql.snowflake_adapter import SnowflakeAdapter


def get_db_adapter_from_config(database_dict):
    # Try to get db_type, default to SQLiteAdapter
    db_type = database_dict.get("type", "SQLiteAdapter").lower()

    if db_type == "sqlite":
        return SQLiteAdapter()
    elif db_type == "snowflake":
        return SnowflakeAdapter()
    else:
        raise ValueError(f"Unknown database type: {db_type}")
