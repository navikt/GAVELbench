"""Module for managing data sources from a configuration file.

This module provides the DataSourceManager class, which is responsible for
loading data sources from a specified configuration file and retrieving
data source paths by key.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class DataSourceManager:
    """A class to manage data sources from a configuration file.

    This class initializes with a path to a configuration file, loads the
    data sources defined in that file, and provides a method to retrieve
    data source paths by their associated keys.

    Attributes:
        config_path (Path): The path to the configuration file.
        data_sources (Dict[str, str]): A dictionary containing data source paths.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the DataSourceManager with a configuration path.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.config_path = Path(config_path)
        self.data_sources = self._load_config()

    def _load_config(self) -> Dict[str, str] | Any:
        """Load data sources from a configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.

        Returns:
            Dict[str, str]: A dictionary of data sources.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r") as file:
            return json.load(file)

    def get_data_source(self, key: str) -> Optional[str]:
        """Retrieve a data source path by key.

        Args:
            key (str): The key for the desired data source.

        Returns:
            Optional[str]: The data source path or a not found message.
        """
        return self.data_sources.get(key, f"Data source not found for key: {key}")


# Example usage
if __name__ == "__main__":
    manager = DataSourceManager("config/data_sources.json")
    print(manager.get_data_source("norallm__normistral-7b-warm-instruct"))
