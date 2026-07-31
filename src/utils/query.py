# This module provides functions to execute SQL queries against BigQuery and handle the results efficiently.  # noqa: E501, W505

import os
from collections.abc import Callable

import pandas as pd
from google.cloud import bigquery
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


class QueryRunner:
    def __init__(self):
        self.console = Console()
        self.client = bigquery.Client()

    def dry_run_query(self, query: str) -> None:
        """Performs a dry run of the query with a clean, formatted output."""
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = self.client.query(query, job_config=job_config)
        gb_size = query_job.total_bytes_processed / 10**9

        color = "red" if gb_size > 50 else "orange3" if gb_size > 20 else "green"

        sql_syntax = Syntax(
            query, "sql", theme="monokai", line_numbers=True, word_wrap=True
        )

        panel = Panel(
            sql_syntax,
            title=f"[{color}]Estimated Scan: {gb_size:.2f} GB[/{color}]",
            subtitle="Dry Run Result",
            border_style=color,
            padding=(1, 2),
            expand=False,
        )

        self.console.print(panel)

    def run_query(self, query: str, safe_config: bool = True) -> pd.DataFrame:  # noqa: FBT001
        """Runs the query and returns the results as a DataFrame.
        If safe_config is True, it sets a maximum bytes billed limit at 25 GB."""
        limit = 25 * 10**9 if safe_config else None
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=limit)

        query_job = self.client.query(query, job_config=job_config)

        df = query_job.to_dataframe(
            progress_bar_type="tqdm",
        )

        return df

    def load_query_results(
        self, query: str, path: str, data_formating_func: Callable | None = None
    ):
        """
        Checks if the file exists in queries/file_name.
        If it does, loads the dile to a pandas dataframe.
        Otherwise, the query is executed and stored in queries/file_name as a parquet file.
        ``````data_formating_func : function that takes for only argument a dataframe "df",
        and gives back a dataframe
        """  # noqa: E501
        if os.path.exists(path=path):
            results = pd.read_parquet(path=path)

        else:
            folder_path = os.path.dirname(path)

            if folder_path and not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
                print(f"Created missing directory tree: [yellow]{folder_path}[/yellow]")

            self.dry_run_query(query)

            results = self.run_query(query)

            if data_formating_func:
                results = data_formating_func(df=results)
            results.to_parquet(path, index=False)

        return results
