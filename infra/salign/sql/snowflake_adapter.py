from infra.salign.sql.database_adapter import DatabaseAdapter
from infra.salign.util.get_config import get_config
from contextlib import contextmanager
import snowflake.connector
import json
import pandas as pd

import logging

logging.getLogger("snowflake.connector").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class SnowflakeAdapter(DatabaseAdapter):

    @contextmanager
    def create_db_connection(self, database):
        """Context manager for Snowflake database connections."""

        config = get_config()
        query_execution_timeout = config["query_execution_timeout"]

        conn = None
        cursor = None
        try:
            cred_path = database["credential_path"]
            snowflake_credential = json.load(open(cred_path))

            conn = snowflake.connector.connect(
                database=database["db_id"],
                network_timeout=query_execution_timeout,
                **snowflake_credential,
            )
            cursor = conn.cursor()
            yield cursor

        except Exception as e:
            logger.error(f"Snowflake connection error: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_table_info(self, database):
        database_name = database["db_id"]

        with self.create_db_connection(database) as cursor:
            # Get all tables and their schemas
            cursor.execute("SHOW TABLES IN DATABASE {}".format(database_name))
            tables = cursor.fetchall()
            # Snowflake 'SHOW TABLES' returns: (table_name, schema_name, ...)

            table_info = {}
            for table in tables:
                table_name = table[1]  # table[1] = TABLE_NAME
                schema_name = table[3]  # table[3] = SCHEMA_NAME
                qualified_table_name = '{}."{}".{}'.format(
                    database_name, schema_name, table_name
                )

                # Now use fully qualified name for DESCRIBE
                cursor.execute(f"DESCRIBE TABLE {qualified_table_name}")
                columns = cursor.fetchall()

                col_list = []
                for column in columns:
                    # Snowflake DESCRIBE TABLE returns: name, type
                    col_name = '"' + str(column[0]) + '"'
                    col_type = column[1]
                    col_list.append({"column": col_name, "type": col_type})

                formatted_table_name = "{}.{}.{}".format(
                    database_name, schema_name, table_name
                )
                table_info[formatted_table_name] = col_list

        return table_info

    def convert_result(self, cursor):
        config = get_config()

        max_rows = config["max_rows_per_query"]

        results = cursor.fetchmany(max_rows)
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)

        return super().convert_decimals_to_floats(df.to_dict(orient="records"))
