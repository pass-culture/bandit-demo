import logging

import lancedb
import pandas as pd

from src.config import LANCE_DB_DIR
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


db = lancedb.connect(LANCE_DB_DIR)


def get_lancedb_table(
    name: str,
    data: pd.DataFrame = None,
    index: bool = False,  # noqa: FBT001
    update: bool = False,  # noqa: FBT001
) -> lancedb.Table:
    """Create the lanceDB table if it does not exists.
    Updates it if asked. Returns the table"""

    table_exists = name in db.table_names()

    if update or not table_exists:
        if data is None:
            logger.error(f"Data must be provided to create or update the table {name}.")
            raise ValueError(
                f"Data must be provided to create or update the table {name}."
            )
        else:
            table = db.create_table(name, data=data, mode="overwrite")
            logger.info(f"Table '{name}' created or updated.")
        if index:
            table.create_index(
                metric="dot",
                num_partitions=256,
                num_sub_vectors=32,
                vector_column_name="item_embedding",
            )
    else:
        table = db.open_table(name)
        logger.info(f"Table '{name}' opened.")

    return table


def retrieve_items(
    table: lancedb.table.Table,
    user_vector: list,
    filter=None,
    nprobes: int | None = None,
    k: int = 20,
    columns: list[str] | None = None,
) -> list[dict]:
    """Fetch top-k items for a given user embedding vector with optional filtering.
    The filter must be a valid LanceDB filter string, e.g.
    "category = 'Music' AND venue_id = 123".
    If no results match the filter, the function returns an empty list.
    The columns parameter allows specifying which columns to return in the results.
    The function defaults to returning 'item_id' and '_distance'.
    """
    # The choice of the metric is key, the finale score must be similar to TT's score
    query = table.search(user_vector, vector_column_name="item_embedding").metric("dot")

    if filter is not None:
        query = query.where(filter, prefilter=True)

    if nprobes is not None and table.list_indices() is not None:
        query = query.nprobes(nprobes)
    elif nprobes is not None:
        logger.warning(
            f"nprobes specified but no index found for table {table.name}. Ignoring nprobes parameter."  # noqa: E501
        )

    # Fallback to defaults if no columns specified to protect old calls
    select_cols = columns if columns is not None else ["item_id", "_distance"]
    if "_distance" not in select_cols:
        select_cols.append("_distance")
    results = query.select(select_cols).limit(k).to_pandas()

    if results.empty:
        logger.warning(f"No items matched the provided filter : {filter}")
        return []

    # Return all chosen columns + _distance which is auto-appended by LanceDB
    return results.to_dict(orient="records")


def retrieve_items_from_user_id(
    table: lancedb.table.Table,
    user_id: str,
    user_data: pd.DataFrame,
    filter=None,
    nprobes: int | None = None,
    k: int = 20,
) -> dict:
    """Fetch top-k similar items for a given user ID with optional filtering.
    The filter must be a valid LanceDB filter string, e.g.
    "category = 'Music' AND venue_id = 123".
    If no results match the filter, the function returns an empty list.
    """
    # Get the user embedding from a pre-loaded dict for O(1) speed
    # user_vector = user_emb_dict.get(user_id)
    user = user_data[user_data["user_id"] == user_id]
    if user.empty:
        logger.warning(f"User not found for user id : {user_id}")
        return []
    user_vector = user["user_embedding"].values[0]

    return retrieve_items(
        table=table, user_vector=user_vector, filter=filter, nprobes=nprobes, k=k
    )
