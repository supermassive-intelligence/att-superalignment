from infra.salign.util.get_db_adapter_from_config import get_db_adapter_from_config

import logging

logger = logging.getLogger(__name__)


def execute_query(example, query):
    return do_execute_query(example, query)


def do_execute_query(example, query):

    database = example["database"]
    db_adapter = get_db_adapter_from_config(database)
    with db_adapter.create_db_connection(database) as cursor:

        falsed = False

        try:
            cursor.execute(query)
            result = db_adapter.convert_result(cursor)
        except Exception as e:
            logger.debug(f"Error executing query: {query}")
            logger.debug(f"Error message: {str(e)}")
            result = str(e)
            falsed = True

        return result, falsed
