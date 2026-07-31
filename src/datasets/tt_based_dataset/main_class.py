import json
import os
import random as rd

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.datasets.base import BaseDataset
from src.datasets.tt_based_dataset.TT_score_conversion import (
    fit_curve_to_score,
    function_to_fit,
)
from src.models.retrieval import get_lancedb_table, retrieve_items


class TtBasedDataset(BaseDataset):
    def __init__(self, config: dict):
        super().__init__(config)

        self._index_table = self.config.get("index_table", False)

        self.user_data = self._get_user_data()
        self.item_data = self._get_item_data()
        self.all_possible_arms = self._get_all_possible_arms()
        self.TT_score_mapping = self._get_TT_score_mapping()
        self.table = self._get_table()

    def _get_user_data(self) -> pd.DataFrame:
        query = self.open_query_file("user_data.sql")
        path = f"{self.path}/user_data.parquet"

        def user_data_cleanup(df):
            df["user_history"] = [
                {
                    key: [self._clean_string(s) for s in value]
                    for key, value in row.items()
                }
                for row in df["user_history"]
            ]
            embedding_tensor = torch.tensor(
                np.stack(df["user_embedding"].values), dtype=torch.float32
            )
            normalized_tensor = F.normalize(embedding_tensor, p=2, dim=1)
            df["user_embedding"] = normalized_tensor.tolist()
            return df

        return self.runner.load_query_results(
            query=query, path=path, data_formating_func=user_data_cleanup
        )

    def _get_item_data(self) -> pd.DataFrame:
        query = self.open_query_file("item_data.sql")
        path = f"{self.path}/item_data.parquet"

        def item_data_cleanup(df):
            for col in ["categories", "subcategories", "venue_types"]:
                df[col] = [self._clean_string(s) for s in df[col]]
            embedding_tensor = torch.tensor(
                np.stack(df["item_embedding"].values), dtype=torch.float32
            )
            normalized_tensor = F.normalize(embedding_tensor, p=2, dim=1)
            df["item_embedding"] = normalized_tensor.tolist()
            return df

        return self.runner.load_query_results(
            query=query, path=path, data_formating_func=item_data_cleanup
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

    def _get_TT_score_mapping(self) -> dict:
        """Load or compute the mapping from TT distance to interaction probabilities."""
        filepath = f"{self.path}/TT_score_mapping.json"
        if os.path.exists(filepath):
            with open(filepath, encoding="utf-8") as f:
                return json.load(f)
        else:
            mapping = fit_curve_to_score()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(mapping, f, indent=4)
            return mapping

    def _get_table(self):
        return get_lancedb_table(
            name=f"{self.name}_items", data=self.item_data, index=self._index_table
        )

    def get_outcome(
        self,
        row: pd.DataFrame,
        chosen_arm_key: str,
        boost: float | None = None,
        allowed_item_ids: list | None = None,  # Added to restrict pool
    ) -> dict:
        """
        Simulate user interaction based on the distance between user and item embeddings.
        The closer the item is to the user in the embedding space, the higher the probability of interaction (click, fave, book).
        Optionally limits the selection pool to a specific subset of allowed item IDs.
        """  # noqa: E501, W505
        outcome = {"clicked": 0, "faved": 0, "booked": 0}
        cat, sub, lt = chosen_arm_key.split("__")

        # Base structural arm filter
        base_filter = (
            f"categories = '{cat}' AND subcategories = '{sub}' AND venue_types = '{lt}'"
        )

        # If a restricted pipeline item pool exists, inject it into the SQL syntax
        if allowed_item_ids is not None:
            if (
                not allowed_item_ids
            ):  # Handle edge case where retrieval returned nothing
                # Warning already logged in the retrieval step
                return outcome
            id_str = ", ".join(
                [
                    f"'{idx}'" if isinstance(idx, str) else str(idx)
                    for idx in allowed_item_ids
                ]
            )
            final_filter = f"({base_filter}) AND item_id IN ({id_str})"
        else:
            final_filter = base_filter

        score = retrieve_items(
            table=self.table, user_vector=row.user_embedding, filter=final_filter, k=1
        )

        if not score:
            return outcome
        else:
            distance = score[0]["_distance"]

        p_click = function_to_fit(x=distance, **self.TT_score_mapping["clicked"])
        p_faved = function_to_fit(x=distance, **self.TT_score_mapping["faved"])
        p_booked = function_to_fit(x=distance, **self.TT_score_mapping["booked"])

        if boost is not None:
            p_click = min(1.0, boost * p_click)
            p_faved = min(1.0, boost * p_faved)
            p_booked = min(1.0, boost * p_booked)

        if rd.random() < p_click:
            outcome["clicked"] = 1

            if p_click > 0:
                if rd.random() < (min(1, p_faved / p_click)):
                    outcome["faved"] = 1
                if rd.random() < (min(1, p_booked / p_click)):
                    outcome["booked"] = 1

        return outcome
