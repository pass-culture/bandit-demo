# This module provides functionality to convert Two-Tower (TT) model scores into estimated conversion probabilities.  # noqa: E501, W505
# These probabilities are then used to compute reward for the TT-based dataset.

import logging
from collections.abc import Callable

import gcsfs
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.optimize import curve_fit

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


fs = gcsfs.GCSFileSystem()

dataset = ds.dataset(
    "mlflow-bucket-prod/benchmark_sandbox/TT_score_conversion_logs/", filesystem=fs
)


def load_filtered_rows(
    dataset: ds.Dataset, extra_filter: ds.Expression | None = None, n: int | None = None
) -> pd.DataFrame:
    """
    Loads rows from the dataset with a base filter and an optional extra filter, then processes the data into a Pandas DataFrame. <br>
    The base filter ensures that only valid scores and user IDs are included, and that the context is specific to user-based recommendations. <br>
    The extra filter can be used to further narrow down the data, for example by offer recommendation rank. <br>
    If 'n' is provided, only the top 'n' rows will be returned.

    - Base filter: score <= 1, user_id != "-1", context == "recommendation:user_based"
       - score <= 1: Ensures that the cold starts are excluded.
       - user_id != "-1": Ensures that the "similar_offer" route is excluded.
       - context == "recommendation:user_based": Ensures that the Top Offers are excluded.

    Please note that the score is a distance metric from the two-tower model, where a lower score indicates a closer match between the user and the item. <br>
    It is computed over normalized embeddings with a dot product, which is equivalent to a cosine similarity under the present conditions.

    - The source dataset contains the columns :
        - item_id
        - offer_id
        - offer_reco_rank
        - offer_category
        - context
        - user_id
        - conversion
        - score
    It was built with data from `past_offer_context_sink` for the recommendation logs and `reco_offer_conversion`for the conversion events
    """  # noqa: E501, W505
    base_filter = (
        (ds.field("score") <= 1)
        & (ds.field("user_id") != "-1")
        & (ds.field("context") == "recommendation:user_based")
    )

    combined_filter = (
        base_filter & extra_filter if extra_filter is not None else base_filter
    )

    logger.info(f"Loading from dataset with filter {combined_filter}...")

    projection = {
        "offer_reco_rank": ds.field("offer_reco_rank"),
        "offer_category": ds.field("offer_category"),
        "score": ds.field("score"),
        "clicked": ds.field(("conversion", "clicked")),
        "faved": ds.field(("conversion", "faved")),
        "booked": ds.field(("conversion", "booked")),
    }

    scanner = dataset.scanner(columns=projection, filter=combined_filter)

    logger.info("Loaded rows from GCS. Converting to Pandas...")

    pyarrow_table = scanner.head(n) if n else scanner.to_table()

    df = pyarrow_table.to_pandas()

    if "score" in df.columns and len(df) > 0:
        bins = pd.qcut(df["score"], q=50, duplicates="drop")
        df["score_bin"] = pd.IntervalIndex(bins).mid

    logger.info(f"Processed DataFrame with {len(df)} rows.")
    return df


def function_to_fit(x: float, k: float, a: float, b: float) -> float:
    """Decreasing exponential.
    x should be a distance, the probability decreases when the distance increases.
    The parameters k, a, b are fitted to the data.
    k controls the initial value of the function when x=0,
    a controls the rate of decay as x increases,
    and b controls the asymptotic value of the function as x approaches infinity."""
    return k * np.exp(-a * x) + b


def fit_curve_to_score(function_to_fit: Callable = function_to_fit) -> dict:
    """
    Fits a curve to the conversion data to find the parameters that best map the TT score to conversion probabilities for different events (clicked, faved, booked). <br>
    The function_to_fit is a decreasing exponential function defined as k * exp(-a * x) + b, where:
    - k controls the initial value of the function when x=0
    - a controls the rate of decay as x increases
    - b controls the asymptotic value of the function as x approaches infinity
    The fitting process uses the curve_fit function from scipy.optimize to find the optimal parameters for each event type based on the conversion data grouped by score bins. <br>
    The resulting parameters are returned in a dictionary format, which can be used to convert TT scores into estimated conversion probabilities for each event type. <br>
    The source dataset contains the columns :
        - item_id
        - offer_id
        - offer_reco_rank
        - offer_category
        - context
        - user_id
        - conversion
        - score
    It was built with data from `past_offer_context_sink` for the recommendation logs and `reco_offer_conversion`for the conversion events. <br>
    It is filtered through the `load_filtered_rows` function. <br>
    Filter: score <= 1, user_id != "-1", context == "recommendation:user_based, offer_reco_rank < 2"
        - score <= 1: Ensures that the cold starts are excluded.
        - user_id != "-1": Ensures that the "similar_offer" route is excluded.
        - context == "recommendation:user_based": Ensures that the Top Offers are excluded.
        - offer_reco_rank < 2: Keeps only the first recommendation of every playlists to ensure that the conversion probabilities are computed on viewed reco.

    Please note that the score is a distance metric from the two-tower model, where a lower score indicates a closer match between the user and the item.
    It is computed over normalized embeddings with a dot product, which is equivalent to a cosine similarity under the present conditions.
    """  # noqa: E501, W505
    df_conv = load_filtered_rows(dataset, (ds.field("offer_reco_rank") < 2))
    events = ["clicked", "faved", "booked"]  # "viewed"
    agg = df_conv.groupby("score_bin", observed=False)[events].agg("mean").reset_index()
    score_bin_numeric = agg["score_bin"].astype(float).values

    event_params = {}

    for event in events:
        params, _ = curve_fit(
            function_to_fit,
            score_bin_numeric,
            agg[event].values,
            p0=[0.02, 15, 0],
            bounds=(0, np.inf),
        )

        event_params[event] = {"k": params[0], "a": params[1], "b": params[2]}

    if event_params["clicked"]["a"] > event_params["faved"]["a"]:
        logger.warning(
            f"The decay parameter for 'clicked' events ({event_params['clicked']['a']}) is higher than for 'faved' ({event_params['faved']['a']}). This may indicate an issue with the data or fitting process."  # noqa: E501
        )
    if event_params["clicked"]["a"] > event_params["booked"]["a"]:
        logger.warning(
            f"The decay parameter for 'clicked' events ({event_params['clicked']['a']}) is higher than for 'booked' ({event_params['booked']['a']}). This may indicate an issue with the data or fitting process."  # noqa: E501
        )

    return event_params
