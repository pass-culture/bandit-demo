import copy
import logging
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.retrieval import retrieve_items
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class Bandit:
    """
    Base Bandit class, arms defined by (category, sub category, venue type) combinations.
    This class can be extended to implement specific strategies like Random, Epsilon-Greedy, etc.
    """  # noqa: E501

    def __init__(self, boost: float | None = None):
        self.arm_counts = None
        self.arm_total_reward = None
        self.initialized = False
        self.all_arms = []
        self.arm_to_idx = {}
        self.logs = pd.DataFrame([])
        self.boost = boost

    def _init_arms(self, arms_pool: list) -> None:
        if not self.initialized:
            self.all_arms = arms_pool
            self.arm_to_idx = {arm: i for i, arm in enumerate(arms_pool)}
            n_arms = len(arms_pool)
            self.arm_counts = np.zeros(n_arms)
            self.arm_total_reward = np.zeros(n_arms)
            self.initialized = True

    def train(
        self,
        dataset: object,
        reward_func: Callable,
        context: str | None = None,
        mode: str = "classic",  # "classic" or "with_retrieval"
    ) -> None:
        if mode == "with_retrieval" and not hasattr(dataset, "table"):
            raise ValueError(
                "The dataset object must have a valid 'table' attribute when mode='with_retrieval'"  # noqa: E501
            )

        df = dataset.training_data
        self._init_arms(dataset.all_possible_arms)

        contexts = np.stack(df[context].values) if context else None
        user_embeddings = (
            np.stack(df["user_embedding"].values) if mode == "with_retrieval" else None
        )

        results = []
        logger.info(f"Training {type(self).__name__} (mode: {mode})...")

        for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
            context_vec = contexts[i] if contexts is not None else None

            if mode == "with_retrieval":
                retrieved_records = retrieve_items(
                    table=dataset.table,
                    user_vector=user_embeddings[i].tolist(),
                    k=100,
                    columns=[
                        "item_id",
                        "categories",
                        "subcategories",
                        "venue_types",
                    ],
                )
                if not retrieved_records:
                    continue  # Skip if retrieval yields no candidates
                    # Warning already logged in the retrieval step

                allowed_item_ids = [rec["item_id"] for rec in retrieved_records]

                arms_pool = list(
                    {
                        f"{rec['categories']}__{rec['subcategories']}__{rec['venue_types']}"
                        for rec in retrieved_records
                    }
                )
            elif mode == "classic":
                arms_pool = dataset.all_possible_arms
                allowed_item_ids = None

            else:
                raise ValueError(
                    f"Invalid mode '{mode}'. Supported modes are 'classic' and 'with_retrieval'."  # noqa: E501
                )

            candidate_idx = self.select_arm_idx(arms_pool, context_vec)
            chosen_arm_key = arms_pool[candidate_idx]
            global_arm_idx = self.arm_to_idx[chosen_arm_key]

            outcome = dataset.get_outcome(
                row, chosen_arm_key, boost=self.boost, allowed_item_ids=allowed_item_ids
            )

            current_reward = reward_func(
                outcome, h_u=row.user_history, chosen_arm_key=chosen_arm_key
            )

            self.update_by_idx(global_arm_idx, current_reward, context_vec)

            logger.debug(
                f"Step {i}: User {row.user_id}, Chosen Arm: {chosen_arm_key}, Reward: {current_reward:.4f}, Arm pool size: {len(arms_pool)}"  # noqa: E501
            )

            results.append(
                {
                    "user_id": row.user_id,
                    "chosen_arm": chosen_arm_key,
                    "outcome": outcome,
                    "reward": current_reward,
                }
            )

        self.logs = pd.concat([self.logs, pd.DataFrame(results)], ignore_index=True)
        logger.info("Bandit training completed.")

    def update_by_idx(
        self, arm_idx: int, reward: float, context: np.ndarray | list | None = None
    ):
        self.arm_total_reward[arm_idx] += reward
        self.arm_counts[arm_idx] += 1

    def select_arm_idx(self, arms_pool: list, context: np.ndarray | list | None = None):
        raise NotImplementedError

    def copy_bandit(self):
        """
        Create a deep copy of the bandit instance.
        """
        return copy.deepcopy(self)

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved {type(self).__name__} instance to {filepath}!")

    @classmethod
    def load(cls, filepath: str) -> "Bandit":
        instance = joblib.load(filepath)
        print(f"Loaded {type(instance).__name__} from {filepath}!")
        return instance

    def evaluate_bandit(
        self, dataset: object, context: str | None = None, len_eval: int = 1000
    ):
        """
        Get the chosen arms for a given dataset without updating the bandit.
        Shows the variety of arms chosen for various contexts.
        Returns a Dataframe with every arm chosen and how many times they were chosen.
        """
        eval_bandit = self.copy_bandit()
        df = dataset.training_data.head(len_eval)
        eval_bandit._init_arms(dataset.all_possible_arms)

        contexts = np.stack(df[context].values) if context else None

        results = []
        logger.info(f"Evaluating {type(self).__name__}...")

        for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
            context = contexts[i] if contexts is not None else None

            arm_idx = eval_bandit.select_arm_idx(dataset.all_possible_arms, context)
            chosen_arm_key = eval_bandit.all_arms[arm_idx]

            results.append(
                {
                    "user_id": row.user_id,
                    "chosen_arm": chosen_arm_key,
                }
            )

        logger.info("Bandit evaluation completed.")
        eval_logs = (
            pd.DataFrame(results).groupby("chosen_arm").size().reset_index(name="count")
        )
        return eval_logs.sort_values("count", ascending=False)
