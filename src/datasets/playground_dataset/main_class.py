import numpy as np
import pandas as pd

from src.datasets.base import BaseDataset
from src.datasets.playground_dataset.playground_utils import (
    generate_playground_dataset,
    generate_synthetic_training_dataset,
    get_evaluation_batches,
)


class PlaygroundDataset(BaseDataset):
    def __init__(self, config: dict):
        super().__init__(config)
        self.categories_taxonomy = self._get_categories_taxonomy()
        self.venues_taxonomy = self._get_venues_taxonomy()
        self.all_possible_arms = self._get_all_possible_arms()
        self._f_map, self._dim = self._prepare_vectorizer(
            self.categories_taxonomy, self.venues_taxonomy
        )

    def _get_categories_taxonomy(self) -> dict:
        """
        Looks for the saved categories_taxonomy in the project. If not found, it fetches the categories taxonomy
        from the database, including the weights for each category and subcategory based on bookings data.
        The weights are based on the number of bookings for each category and subcategory between 01/02/2026 and 01/04/2026.
        """  # noqa: E501, W505

        query = self.open_query_file("categories_taxonomy.sql")
        path = f"{self.path}/categories_taxonomy.parquet"
        results = self.runner.load_query_results(query=query, path=path)

        return self._get_dict(
            {
                row["offer_category_id"]: {
                    "weight": row["cat_bookings"],
                    "subcategories": dict(
                        zip(row["subcategories"], row["subcat_bookings"], strict=True)
                    ),
                }
                for _, row in results.iterrows()
            }
        )

    def _get_venues_taxonomy(self) -> dict:
        """
        Looks for the saved venues_taxonomy in the project. If not found, it fetches the venues taxonomy
        from the database, including the weights for each venue type based on bookings data.
        The weights are based on the number of bookings for each venue type between 01/02/2026 and 01/04/2026.
        """  # noqa: E501, W505

        query = self.open_query_file("venues_taxonomy.sql")
        path = f"{self.path}/venues_taxonomy.parquet"
        results = self.runner.load_query_results(query=query, path=path)

        return self._get_dict(
            {
                row["venue_type_label"]: {
                    "weight": row["bookings"] if row["bookings"] is not None else 0,
                    "count": row["venues"] if row["venues"] is not None else 0,
                }
                for _, row in results.iterrows()
            }
        )

    def _get_all_possible_arms(self) -> list:
        """
        Defines the action space as all unique combinations of categories, subcategories, and venue types from the item data.
        The action keys are formatted as "category__subcategory__venue_type".
        """  # noqa: E501, W505

        query = self.open_query_file("all_possible_arms.sql")
        path = f"{self.path}/arms_data.parquet"
        results = self.runner.load_query_results(query=query, path=path)

        return [
            f"{self._clean_string(row['cat'])}__{self._clean_string(row['sub'])}__{self._clean_string(row['lt'])}"
            for _, row in results.iterrows()
        ]

    def _prepare_vectorizer(
        self, categories_taxonomy: dict, venues_taxonomy: dict
    ) -> tuple:
        """Creates a mapping from strings to vector indices."""
        categories = list(categories_taxonomy.keys())
        subcategories = [
            s for c in categories_taxonomy.values() for s in c["subcategories"]
        ]
        locs = list(venues_taxonomy.keys())

        feature_names = categories + subcategories + locs
        feature_to_idx = {name: i for i, name in enumerate(feature_names)}
        return feature_to_idx, len(feature_names)

    def _vectorize_history(
        self, history: dict, feature_to_idx: dict, dim: int
    ) -> np.ndarray:
        """
        Converts {'categories': {...}, 'subcategories': {...}, 'venue_types': {...}} to a binary vector
        """  # noqa: E501, W505
        v = np.zeros(dim)
        for key in ["categories", "subcategories", "venue_types"]:
            for item in history.get(key, []):
                if item in feature_to_idx:
                    v[feature_to_idx[item]] = 1
        return v

    def generate_playground_dataset(
        self, num_users=100, playlist_length=40
    ) -> pd.DataFrame:
        return generate_playground_dataset(
            categories_taxonomy=self.categories_taxonomy,
            venues_taxonomy=self.venues_taxonomy,
            num_users=num_users,
            playlist_length=playlist_length,
        )

    def get_evaluation_batches(self, df: pd.DataFrame) -> pd.DataFrame:
        return get_evaluation_batches(df)

    def generate_synthetic_training_dataset(self, num_user=1000) -> pd.DataFrame:
        return generate_synthetic_training_dataset(
            categories_taxonomy=self.categories_taxonomy,
            venues_taxonomy=self.venues_taxonomy,
            all_possible_arms=self.all_possible_arms,
            num_users=num_user,
        )

    def add_vectorized_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a column "context_vec" to the dataframe,
        corresponding to vectorized user history"""
        df["context_vec"] = df["user_history"].apply(
            lambda x: self._vectorize_history(x, self._f_map, self._dim)
        )
        return df

    def get_outcome(
        self,
        row: pd.DataFrame,
        chosen_arm_key: str,
        boost: float | None = None,
        **kwargs,
    ) -> dict:
        if boost is not None:
            raise ValueError(
                f"boost {boost} given but playground_dataset does not accept boost"
            )
        return getattr(row, chosen_arm_key)
