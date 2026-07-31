import random

import numpy as np
import pandas as pd


def get_exponential_count(max_val: int, scale: float = 1.0) -> int:
    """
    Uses exponential distribution to pick a count.
    Most results will be low (1-3), few will be high.
    """
    val = int(np.random.exponential(scale)) + 1
    return min(val, max_val)


def get_weibull_count(
    max_val: int, shape_k: float = 1.5, scale_lambda: float = 1.0
) -> int:
    """
    Uses Weibull distribution to pick a count.
    Takes the shape (k) as an argument.
    The shape is multiplied by scale_lambda to stretch the distribution.
    """
    val = max(1, round(np.random.weibull(shape_k) * scale_lambda))
    return min(val, max_val)


def generate_user_history(
    categories_taxonomy: dict,
    venues_taxonomy: dict,
    categories: list,
    venue_types: list,
    categories_weight: float,
    venue_types_weight: float,
    venue_ids: bool = True,  # noqa: FBT001
) -> dict:
    num_known_categories = get_weibull_count(
        len(categories), shape_k=1.8, scale_lambda=2.9
    )
    known_categories = np.random.choice(
        categories,
        size=num_known_categories,
        replace=False,
        p=np.array([c.get("weight", 0) for c in categories_taxonomy.values()])
        / categories_weight,
    )

    subcategories = []
    for c in known_categories:
        num_known_subcategories = get_weibull_count(
            len(categories_taxonomy[c]["subcategories"]), shape_k=1.6, scale_lambda=3.2
        )
        known_subcategories = np.random.choice(
            list(categories_taxonomy[c]["subcategories"].keys()),
            size=num_known_subcategories,
            replace=False,
            p=np.array(list(categories_taxonomy[c]["subcategories"].values()))
            / categories_taxonomy[c]["weight"],
        )
        subcategories.extend(known_subcategories)

    num_known_venue_types = get_weibull_count(
        len(venue_types), shape_k=1.8, scale_lambda=3.2
    )
    known_venue_types = np.random.choice(
        venue_types,
        size=num_known_venue_types,
        replace=False,
        p=np.array([venues_taxonomy[lt].get("weight", 0) for lt in venues_taxonomy])
        / venue_types_weight,
    )

    if venue_ids:
        known_venue_ids = []
        for lt in known_venue_types:
            num_known_venue_ids = get_weibull_count(
                venues_taxonomy[lt]["count"], shape_k=1.4, scale_lambda=5
            )
            # We don't want to generate the same venue_id twice for the same user
            existing_ids = set(known_venue_ids)
            new_ids = set()
            while len(new_ids) < num_known_venue_ids:
                new_id = f"venue_{lt}_{random.randint(1, venues_taxonomy[lt]['count'])}"
                if new_id not in existing_ids:
                    new_ids.add(new_id)
                    existing_ids.add(new_id)
            known_venue_ids.extend(new_ids)

    user_history = {
        "categories": set(known_categories),
        "subcategories": set(subcategories),
        "venue_types": set(known_venue_types),
        "venue_ids": set(known_venue_ids) if venue_ids else None,
    }

    return user_history


def get_conversion(
    user_history: dict,
    cat: str,
    sub: str,
    lt: str,
    venue_id: str | None = None,
    rank: int | None = None,
) -> dict:
    p_click = 0.02
    if cat in user_history["categories"]:
        p_click += 0.05
    if sub in user_history["subcategories"]:
        p_click += 0.12
    if lt in user_history["venue_types"]:
        p_click += 0.08
    if venue_id and venue_id in user_history["venue_ids"]:
        p_click += 0.4

    if rank:
        p_click = p_click / np.sqrt(rank + 1)  # Position bias

    clicked = 1 if random.random() < p_click else 0
    faved = 1 if clicked and random.random() < 0.3 else 0
    booked = 1 if clicked and random.random() < 0.1 else 0

    return {"clicked": clicked, "faved": faved, "booked": booked}


def get_arm_conversion(arm_key: str, user_history: dict) -> dict:
    cat, sub, lt = arm_key.split("__")
    return get_conversion(user_history, cat, sub, lt)


def generate_playground_dataset(
    categories_taxonomy: dict,
    venues_taxonomy: dict,
    num_users: int = 100,
    playlist_length: int = 40,
) -> pd.DataFrame:
    """
    Generates a synthetic dataset for the recommendation system playground.
     - Each user has a history of known categories, subcategories, venue types, and venue IDs, with counts determined by a Weibull distribution to create a long-tail effect.
     - Each user has a random number of playlists with the same long tail effect (usually 1-2 playlists per user), and each playlist contains a ranked list of offers with associated features.
     - Click probabilities are influenced by whether the offer's category, subcategory, or venue type matches the user's history, and also by the offer's rank in the playlist to simulate position bias.
    """  # noqa: E501, W505

    categories = list(categories_taxonomy.keys())
    venue_types = list(venues_taxonomy.keys())

    categories_weight = sum(c.get("weight", 0) for c in categories_taxonomy.values())
    venue_types_weight = sum(
        venues_taxonomy[lt].get("weight", 0) for lt in venues_taxonomy
    )

    data = []

    for user_id in range(num_users):
        user_history = generate_user_history(
            categories_taxonomy=categories_taxonomy,
            venues_taxonomy=venues_taxonomy,
            categories=categories,
            venue_types=venue_types,
            categories_weight=categories_weight,
            venue_types_weight=venue_types_weight,
        )

        num_playlists = get_exponential_count(4, scale=1.0)

        for p_idx in range(num_playlists):
            reco_id = f"u{user_id}_p{p_idx}"

            for rank in range(1, playlist_length + 1):
                cat = random.choice(categories)
                sub = random.choice(
                    list(categories_taxonomy[cat]["subcategories"].keys())
                )

                lt = random.choice(venue_types)
                venue_id = (
                    f"venue_{lt}_{random.randint(1, venues_taxonomy[lt]['count'])}"
                )

                conversion = get_conversion(user_history, cat, sub, lt, venue_id, rank)

                data.append(
                    {
                        "playlist_reco_id": reco_id,
                        "user_id": user_id,
                        "offer_rank": rank,
                        "offer_category": cat,
                        "offer_subcategory": sub,
                        "offer_venue_type": lt,
                        "offer_venue_id": venue_id,
                        # "offer_genre": None,
                        "user_history_categories": user_history["categories"],
                        "user_history_subcategories": user_history["subcategories"],
                        "user_history_venue_types": user_history["venue_types"],
                        "user_history_venue_ids": user_history["venue_ids"],
                        "clicked": conversion["clicked"],
                        "faved": conversion["faved"],
                        "booked": conversion["booked"],
                    }
                )

    return pd.DataFrame(data)


def get_evaluation_batches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the synthetic dataset into logical units for reward evaluation.
    Each unit = (Recommended Playlist + User History + Ground Truth Outcomes)
    """
    df = df.sort_values(["playlist_reco_id", "offer_rank"])

    df["offer_data"] = df[
        [
            "offer_rank",
            "offer_category",
            "offer_subcategory",
            "offer_venue_type",
            "offer_venue_id",
        ]
    ].to_dict(orient="records")

    df["outcome_data"] = df[["clicked", "faved", "booked"]].to_dict(orient="records")

    scenarios = (
        df.groupby("playlist_reco_id")
        .agg(
            {
                "user_id": "first",
                "user_history_categories": "first",
                "user_history_subcategories": "first",
                "user_history_venue_types": "first",
                "user_history_venue_ids": "first",
                "offer_data": list,
                "outcome_data": list,
            }
        )
        .reset_index()
    )

    scenarios["user_history"] = scenarios.apply(
        lambda x: {
            "categories": x["user_history_categories"],
            "subcategories": x["user_history_subcategories"],
            "venue_types": x["user_history_venue_types"],
            "venue_ids": x["user_history_venue_ids"],
        },
        axis=1,
    )

    scenarios = scenarios.rename(
        columns={
            "playlist_reco_id": "reco_id",
            "offer_data": "playlist",
            "outcome_data": "outcomes",
        }
    )

    return scenarios[["reco_id", "user_id", "user_history", "playlist", "outcomes"]]


def generate_synthetic_training_dataset(
    categories_taxonomy: dict,
    venues_taxonomy: dict,
    all_possible_arms: list,
    num_users: int = 1000,
) -> pd.DataFrame:
    """
    Generates a synthetic training dataset with the same parameters as the playground dataset
    for user history and conversion probabilities, but organized for the training of bandit algorithms
    where each arm is a triplet of (category, subcategory, venue type).
    Each row represents a user interaction with all of the possible arms.
    """  # noqa: E501, W505

    categories = list(categories_taxonomy.keys())
    venue_types = list(venues_taxonomy.keys())

    categories_weight = sum(c.get("weight", 0) for c in categories_taxonomy.values())
    venue_types_weight = sum(
        venues_taxonomy[lt].get("weight", 0) for lt in venues_taxonomy
    )

    data = []

    for user_id in range(num_users):
        user_history = generate_user_history(
            categories_taxonomy=categories_taxonomy,
            venues_taxonomy=venues_taxonomy,
            categories=categories,
            venue_types=venue_types,
            categories_weight=categories_weight,
            venue_types_weight=venue_types_weight,
            venue_ids=False,
        )

        data.append(
            {
                "user_id": user_id,
                "user_history": user_history,
                **{
                    arm_key: get_arm_conversion(arm_key, user_history)
                    for arm_key in all_possible_arms
                },
            }
        )

    return pd.DataFrame(data)
