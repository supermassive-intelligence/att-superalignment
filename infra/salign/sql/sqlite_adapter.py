import os
import sqlite3
import time
from infra.salign.sql.database_adapter import DatabaseAdapter
from infra.salign.util.get_config import get_config
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class SQLiteAdapter(DatabaseAdapter):

    @contextmanager
    def create_db_connection(self, database):
        """Context manager for SQLite database connections."""
        database_path = self.get_database_path(database)
        assert os.path.exists(
            database_path
        ), f"Database path does not exist: {database_path}"

        config = get_config()
        query_execution_timeout = config["query_execution_timeout"]

        conn = None
        cursor = None
        try:
            conn = self.create_sqlite_connection(
                database_path, timeout_seconds=query_execution_timeout
            )
            cursor = conn.cursor()
            yield cursor

        except Exception as e:
            logger.error(f"SQLite connection error: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_database_path(self, database):
        """Get database path from database configuration."""
        return database["path"]

    def create_sqlite_connection(self, db_file, timeout_seconds=30):
        """Create a SQLite connection with timeout handling."""
        # Create a connection to the SQLite database
        conn = sqlite3.connect(db_file)

        # Track when the query started
        start_time = time.time()

        # Define a progress handler function
        def progress_callback():
            # Check if we've exceeded our timeout
            if time.time() - start_time > timeout_seconds:
                # Returning non-zero will cause the query to abort
                return 1
            # Return 0 to continue execution
            return 0

        # Set the progress handler
        # The second parameter is the number of SQLite virtual machine instructions
        # to execute between invocations of the callback
        conn.set_progress_handler(progress_callback, 1000)

        return conn

    def get_table_info(self, database):
        # Get all table names and column names
        with self.create_db_connection(database) as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            table_info = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info('{table_name}')")
                columns = cursor.fetchall()

                for index, column in enumerate(columns):

                    columns[index] = list(column)

                    if column[1].find("-") != -1:
                        columns[index][1] = '"' + column[1] + '"'
                    elif column[1].find(" ") != -1:
                        columns[index][1] = '"' + column[1] + '"'

                table_info[table_name] = [
                    {"column": column[1], "type": column[2]} for column in columns
                ]

        return table_info

    def convert_result(self, cursor):
        result = list(cursor.fetchall())
        result = super().convert_decimals_to_floats(result)

        return result
