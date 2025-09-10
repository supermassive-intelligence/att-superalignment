from abc import ABC, abstractmethod
from decimal import Decimal
import datetime


class DatabaseAdapter(ABC):
    @abstractmethod
    def get_table_info(self, database):
        pass

    @abstractmethod
    def create_db_connection(self, database):
        pass

    @abstractmethod
    def convert_result(self, cursor):
        pass

    def convert_decimals_to_floats(self, obj):
        """
        Recursively convert all Decimal objects to floats in a nested data structure.

        Args:
            obj: The object to process (dict, list, or any other type)

        Returns:
            The processed object with Decimals converted to floats
        """
        if isinstance(obj, Decimal):
            return float(obj)
        # convert date or time to string
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, datetime.time):
            return obj.isoformat()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {
                key: self.convert_decimals_to_floats(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self.convert_decimals_to_floats(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_decimals_to_floats(item) for item in obj)
        else:
            return obj
