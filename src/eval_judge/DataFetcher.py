"""Module for fetching and processing data from specified sources.

This module provides the DataFetcher class, which is responsible for fetching
data from specified sources and converting it into a Dataset object for further
processing. It utilizes the DataSourceManager to manage data sources and
assumes the use of the `datasets` library for handling datasets.

Classes:
    DataFetcher: A class to fetch and process data from various sources.
"""

import json
from typing import Any

from datasets import Dataset  # Assuming you're using the `datasets` library

from eval_judge.DataManager import DataSourceManager


class DataFetcher:
    """A class to fetch and process data from specified sources."""

    def __init__(self, data_source_manager: DataSourceManager) -> None:
        """Initialize the DataFetcher with a DataSourceManager.

        Args:
            data_source_manager (DataSourceManager): The manager for data sources.
        """
        self.manager = data_source_manager

    def fetch_data(self, key: str) -> Any:
        """Fetch data from a specific source.

        Args:
            key (str): The key to identify the data source.

        Returns:
            list[dict]: The fetched data as a list of dictionaries.
        """
        data_source: str | Any = self.manager.get_data_source(key)
        with open(data_source, "r") as file:
            dataset = json.load(file)
        return dataset

    def get_dataset(self, key: str) -> Dataset:
        """Convert fetched data into a Dataset object.

        Args:
            key (str): The key to identify the generated answers data source.

        Returns:
            Dataset: A Dataset object containing user inputs, responses, and references.
        """
        bob_answers = self.fetch_data("bob_answers")
        generated_answers = self.fetch_data(key)

        dataset = []

        for item_bob in bob_answers:
            for item_gen in generated_answers:
                if item_bob["contextualized_question"] == item_gen["question"]:
                    dataset.append(
                        {
                            "user_input": item_bob["contextualized_question"],
                            "response": item_gen["answer"],
                            "reference": item_bob["answer_content"],
                        }
                    )

        # Create a Dataset object from the fetched data
        data_set = Dataset.from_dict(
            {
                "user_input": [item["user_input"] for item in dataset],
                "response": [item["response"] for item in dataset],
                "reference": [item["reference"] for item in dataset],
            }
        )
        return data_set


# Example usage
if __name__ == "__main__":
    manager = DataSourceManager("config/data_sources.json")
    fetcher = DataFetcher(manager)
    dataset = fetcher.get_dataset("NBAiLab__borealis-open-4b-gguf")
    print(dataset)
    print(dataset.to_pandas().head())
