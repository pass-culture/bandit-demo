import logging
from collections.abc import Callable

import pandas as pd
from tqdm import tqdm

from src.datasets.tt_based_dataset.TT_score_conversion import function_to_fit
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class OracleBaselineCalculator:
    def __init__(
        self, table, all_possible_arms, TT_score_mapping, boost: float | None = None
    ):
        self.table = table
        self.TT_score_mapping = TT_score_mapping
        self.boost = boost

        # STRUCTURAL DICTIONARIES
        # Maps a subcategory to a SET of all its valid parent categories
        self.subcat_to_cats = {}
        # Maps a venue_type to a SET of all its valid parent subcategories
        self.vt_to_subcats = {}

        for arm in all_possible_arms:
            cat, subcat, vt = arm.split("__")

            if subcat not in self.subcat_to_cats:
                self.subcat_to_cats[subcat] = set()
            self.subcat_to_cats[subcat].add(cat)

            if vt not in self.vt_to_subcats:
                self.vt_to_subcats[vt] = set()
            self.vt_to_subcats[vt].add(subcat)

    def _get_valid_bitmasks_for_user(
        self, h_cats: set, h_subcats: set, h_vts: set
    ) -> list[int]:
        """
        Determines valid bitmasks.
        Accounts for multi-parent nodes like 'other'.
        """
        valid_masks = [0, 1, 2, 3, 4, 5, 6, 7]

        # Evaluate Bit 2 == 0 (Cat NOT IN) and Bit 1 == 1 (Subcat IN)
        if h_subcats:
            # Gather ALL possible parent categories for ALL subcategories in user hist
            possible_parent_cats = set()
            for s in h_subcats:
                if s in self.subcat_to_cats:
                    possible_parent_cats.update(self.subcat_to_cats[s])

            # If the user has already interacted with EVERY potential parent category
            # for their current subcategories, then 'Subcat IN' AND 'Cat NOT IN'
            # is guaranteed to yield 0 results from LanceDB.
            if possible_parent_cats.issubset(h_cats):
                valid_masks.remove(2)  # 010
                valid_masks.remove(3)  # 011

        # Evaluate Bit 1 == 0 (Subcat NOT IN) and Bit 0 == 1 (VT IN)
        if h_vts:
            possible_parent_subcats = set()
            for vt in h_vts:
                if vt in self.vt_to_subcats:
                    possible_parent_subcats.update(self.vt_to_subcats[vt])

            if possible_parent_subcats.issubset(h_subcats):
                valid_masks.remove(1)  # 001
                valid_masks.remove(5)  # 101

        return valid_masks

    def _build_profile_filter(
        self, bitmask: int, h_categories: list, h_subcategories: list, h_vts: list
    ) -> str | None:
        """
        Construit le filtre SQL d'inclusion/exclusion le plus efficace
        pour un des 8 profils basés sur le bitmask (000 à 111).
        """
        filters = []

        # Python's bitwise operator : x & y does a "bitwise and".
        # Each bit of the output is 1 if the corresponding bit of x AND of y is 1
        # Otherwise it's 0.

        # Gestion des Categories (Bit 2)
        if bitmask & 4:  # 4 corresponds to a 1 on bit 2 -> IN
            if not h_categories:
                return None
            filters.append(
                f"categories IN ({', '.join(f"'{c}'" for c in h_categories)})"
            )
        else:  # -> NOT IN
            if h_categories:
                filters.append(
                    f"categories NOT IN ({', '.join(f"'{c}'" for c in h_categories)})"
                )

        # Gestion des Subcategories (Bit 1)
        if bitmask & 2:  # 2 corresponds to a 1 on bit 1 -> IN
            if not h_subcategories:
                return None
            filters.append(
                f"subcategories IN ({', '.join(f"'{s}'" for s in h_subcategories)})"
            )
        else:  # -> NOT IN
            if h_subcategories:
                filters.append(
                    f"subcategories NOT IN ({', '.join(f"'{s}'" for s in h_subcategories)})"  # noqa: E501
                )

        # Gestion des Venue Types (Bit 0)
        if bitmask & 1:  # 1 corresponds to a 1 on bit 0 -> IN
            if not h_vts:
                return None
            filters.append(f"venue_types IN ({', '.join(f"'{vt}'" for vt in h_vts)})")
        else:  # -> NOT IN
            if h_vts:
                filters.append(
                    f"venue_types NOT IN ({', '.join(f"'{vt}'" for vt in h_vts)})"
                )

        return " AND ".join(filters) if filters else None

    def _get_best_item_for_filter(
        self, user_vector: list, sql_filter: str | None
    ) -> tuple[str, float] | tuple[None, None]:
        """
        Exécute la requête de recherche vectorielle Top-1 avec le filtre de liste.
        """
        query = self.table.search(
            user_vector, vector_column_name="item_embedding"
        ).metric("dot")

        if sql_filter:
            query = query.where(sql_filter, prefilter=True)

        res = (
            query.select(["categories", "subcategories", "venue_types", "_distance"])
            .limit(1)
            .to_pandas()
        )

        if res.empty:
            return None, None

        row = res.iloc[0]
        arm_key = f"{row['categories']}__{row['subcategories']}__{row['venue_types']}"
        return arm_key, row["_distance"]

    def compute_expected_conversion_reward(
        self, distance: float, weights_dict: dict, boost: float | None = None
    ) -> dict:
        """
        Calcule l'espérance (la proba de conversion) de chaque événement pour une distance.
        """  # noqa: E501
        expected_p = {}
        for event in ["clicked", "faved", "booked"]:
            p = function_to_fit(x=distance, **weights_dict[event])
            if boost is not None:
                p = min(1.0, boost * p)
            expected_p[event] = p
        return expected_p

    def compute_dataset_baselines(
        self,
        df: pd.DataFrame,
        reward_func: Callable,
        reward_parameters: dict | None = None,
    ) -> dict:
        """
        Calcule les baselines Oracle via l'inversion des filtres (IN / NOT IN).
        Analyse au maximum 8 profils de requêtes par utilisateur.
        """
        logger.info(
            "Calcul des baselines théoriques de l'Oracle (Filtres IN/NOT IN)..."
        )

        max_rewards_global = []
        best_p = []

        for row in tqdm(df.itertuples(index=False), total=len(df)):
            user_vec = row.user_embedding
            h_u = row.user_history

            h_categories = (
                list(set(h_u.get("categories", []))) if isinstance(h_u, dict) else []
            )
            h_subcategories = (
                list(set(h_u.get("subcategories", []))) if isinstance(h_u, dict) else []
            )
            h_vts = (
                list(set(h_u.get("venue_types", []))) if isinstance(h_u, dict) else []
            )

            h_cats_set = set(h_categories)
            h_subcats_set = set(h_subcategories)
            h_vts_set = set(h_vts)

            best_global = -float("inf")
            best_probs = {"clicked": 0.0, "faved": 0.0, "booked": 0.0}

            # O(1) structural validation mapping
            valid_bitmasks = self._get_valid_bitmasks_for_user(
                h_cats_set, h_subcats_set, h_vts_set
            )

            for bitmask in valid_bitmasks:
                sql_filter = self._build_profile_filter(
                    bitmask, h_categories, h_subcategories, h_vts
                )

                arm_key, distance = self._get_best_item_for_filter(user_vec, sql_filter)

                if distance is None:
                    logger.warning(
                        f"No item found for filter {sql_filter} and user history {h_u}"
                    )
                    continue

                # Calcul de l'espérance de conversion
                exp_outcome = self.compute_expected_conversion_reward(
                    distance, self.TT_score_mapping, self.boost
                )

                # Évaluation du reward selectionné
                reward_global = reward_func(
                    exp_outcome,
                    h_u=h_u,
                    chosen_arm_key=arm_key,
                    **(reward_parameters or {}),
                )

                if reward_global > best_global:
                    best_global = reward_global
                    best_probs = exp_outcome

            # Sauvegarde des métriques pour cette ligne (ce user)
            if best_global == -float("inf"):
                logger.warning(
                    f"No possible gain found for user_vec {user_vec} and history {h_u} "
                )
                best_global = 0.0

            max_rewards_global.append(best_global)
            best_p.append(best_probs)

        return {
            "oracle_expected_reward": max_rewards_global,
            "oracle_conversion_probabilities": best_p,
        }
