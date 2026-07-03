"""Datafetcher Module.

This module provides the DataFetcher class, which is responsible for fetching and processing data
from various sources, including external categories and mappings. It utilizes a DataSourceManager
to retrieve data and supports loading JSON files into structured formats for further processing.
"""

import json
from typing import Any, Dict, List

from datasets import Dataset

from eval_judge.DataManager import DataSourceManager


class DataFetcher:
    """A class to fetch and process data from various sources including external categories and mappings."""

    def __init__(self, data_source_manager: DataSourceManager) -> None:
        """Initialize the DataFetcher with a DataSourceManager.

        Args:
            data_source_manager: An instance of DataSourceManager to manage data sources.
        """
        self.manager = data_source_manager

    def _load_json_list(self, file_path: str) -> List[Dict[str, Any]]:
        """Helper to load JSON file into a list of dicts.

        Args:
            file_path: The path to the JSON file to be loaded.

        Returns:
            A list of dictionaries loaded from the JSON file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the JSON structure is unexpected or invalid.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    return content
                elif isinstance(content, dict) and "data" in content:
                    return content["data"] if isinstance(content["data"], list) else []
                else:
                    raise ValueError(f"Unexpected JSON structure in {file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data source not found at: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")

    def fetch_data(self, key: str) -> Any:
        """Fetch data from a specific source.

        Args:
            key: The key identifying the data source to fetch.

        Returns:
            The data fetched from the specified source.
        """
        data_source_path: str | Any = self.manager.get_data_source(key)
        return self._load_json_list(data_source_path)

    def get_dataset(
        self,
        key: str,
        categories_key: str = "categories",
        mapping_key: str = "category_mapping",
    ) -> Dataset:
        """Construct the final dataset by merging answers, categories, and mapping to overcategories.

        Args:
            key: Key for generated answers.
            categories_key: Key for the detailed categories file (contains 'data_categories').
            mapping_key: Key for the mapping file (maps 'kategori' -> 'overkategori').

        Returns:
            A Dataset object constructed from the fetched data.
        """
        # 1. Fetch Base Data
        bob_answers = self.fetch_data("bob_answers")
        generated_answers = self.fetch_data(key)

        # 2. Fetch Categories (Detailed)
        try:
            categories_data = self.fetch_data(categories_key)
            has_cats_file = True
        except FileNotFoundError:
            print(
                f"Warning: '{categories_key}' not found. Proceeding without detailed categories."
            )
            categories_data = []
            has_cats_file = False

        # 3. Fetch Mapping (Kategori -> Overkategori)
        try:
            mapping_data = self.fetch_data(mapping_key)
            has_map_file = True
        except FileNotFoundError:
            print(f"Warning: '{mapping_key}' not found. Skipping overcategory mapping.")
            mapping_data = []
            has_map_file = False

        # --- BUILD LOOKUPS ---

        # Bob Lookup: Question -> Reference & Internal Cats
        bob_lookup: Dict[str, Dict[str, Any]] = {}
        for item in bob_answers:
            q = item.get("contextualized_question")
            if q:
                bob_lookup[q] = {
                    "reference": item.get("answer_content", ""),
                    "internal_cats": item.get("data_categories", []),
                }

        # Generated Lookup: Question -> Answer
        gen_lookup: Dict[str, str] = {}
        for item in generated_answers:
            q = item.get("question")
            if q and q not in gen_lookup:
                gen_lookup[q] = item.get("answer", "")

        # Categories Lookup: Question -> List of Categories
        cat_lookup: Dict[str, List[str]] = {}
        if has_cats_file:
            for item in categories_data:
                q = item.get("contextualized_question")
                if q:
                    if q not in cat_lookup:
                        cat_lookup[q] = []
                    entry_cats = item.get("data_categories", [])
                    if entry_cats:
                        cat_lookup[q].extend(entry_cats)

        # Mapping Lookup: Kategori -> List of Overkategori
        map_lookup: Dict[str, List[str]] = {}
        if has_map_file:
            for item in mapping_data:
                kategorie = item.get("kategori")
                overkategorier = item.get("overkategori", [])
                if kategorie:
                    if kategorie not in map_lookup:
                        map_lookup[kategorie] = []
                    # Extend in case multiple entries map same kategori to different overkategorier
                    map_lookup[kategorie].extend(overkategorier)

        # --- MERGE AND TRANSFORM ---

        dataset_rows = []

        # Primary iteration over Bob's questions (ensures we have a reference)
        for q, bob_info in bob_lookup.items():
            if q in gen_lookup:
                response = gen_lookup[q]
                reference = bob_info["reference"]

                # Get the list of categories for this question
                # Priority: External Category File > Internal Bob Data
                if has_cats_file and q in cat_lookup:
                    current_categories = cat_lookup[q]
                else:
                    current_categories = bob_info["internal_cats"]

                # Clean up duplicates in categories list
                current_categories = list(dict.fromkeys(current_categories))

                # --- MAP TO OVERKATEGORIER ---
                overkategorier_set = set()
                if has_map_file:
                    for cat in current_categories:
                        # Check if this exact category exists in our mapping
                        if cat in map_lookup:
                            overkategorier_set.update(map_lookup[cat])

                # Convert set back to sorted list for consistency
                final_overkategorier = sorted(list(overkategorier_set))

                dataset_rows.append(
                    {
                        "user_input": q,
                        "response": response,
                        "reference": reference,
                        "data_categories": current_categories,
                        "overkategori": final_overkategorier,
                    }
                )

        if not dataset_rows:
            return Dataset.from_dict(
                {
                    "user_input": [],
                    "response": [],
                    "reference": [],
                    "data_categories": [],
                    "overkategori": [],
                }
            )

        return Dataset.from_dict(
            {
                "user_input": [r["user_input"] for r in dataset_rows],
                "response": [r["response"] for r in dataset_rows],
                "reference": [r["reference"] for r in dataset_rows],
                "data_categories": [r["data_categories"] for r in dataset_rows],
                "overkategori": [r["overkategori"] for r in dataset_rows],
            }
        )


# Example Usage
if __name__ == "__main__":
    manager = DataSourceManager("config/data_sources.json")
    fetcher = DataFetcher(manager)

    # Ensure your config maps:
    # "categories": "path/to/categories.json"
    # "category_mapping": "path/to/category_mapping.json"

    try:
        ds = fetcher.get_dataset(
            key="NBAiLab__borealis-open-4b-gguf",
            categories_key="categories",
            mapping_key="category_mapping",
        )

        print("✅ Dataset created with 'overkategori' column!")
        print(ds.to_pandas().head())

        # Verify logic
        if "overkategori" in ds.column_names:
            sample_row = ds[0]
            print("\nSample Row:")
            print(f"Categories: {sample_row['data_categories']}")
            print(f"Mapped Overcategories: {sample_row['overkategori']}")

    except Exception as e:
        print(f"Error: {e}")
