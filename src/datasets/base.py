import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.config import DATA_DIR, SQL_DIR
from src.utils.query import QueryRunner


class BaseDataset(ABC):
    """
    Classe de base abstraite pour tous los datasets du projet.
    Force une interface commune et gère la logique partagée.
    """

    def __init__(self, config: dict):
        """
        Initialisation commune à tous les datasets.
        """
        self._config = config
        self._name = config.get("name", "unknown_dataset")
        self._data_path = f"{DATA_DIR}/{self._name}"
        self._query_path = f"{SQL_DIR}/{self._name}"
        self.runner = QueryRunner()

        self.training_data: Any = None

        self._RE_SPACES = re.compile(r"[\s\-/]")
        self._RE_NON_ALPHA = re.compile(r"[^\w]")
        self._RE_MULTI_UNDERSCORE = re.compile(r"_{2,}")

    @property
    def config(self) -> dict:
        return self._config

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._data_path

    @property
    def query_path(self) -> str:
        return self._query_path

    def open_query_file(self, filename: str) -> str:
        with open(f"{self.query_path}/{filename}", encoding="utf-8") as f:
            return f.read()

    @abstractmethod
    def get_outcome(self, row: pd.DataFrame, chosen_arm_key: str) -> dict:
        """
        Retourne l'outcome d'une recommendation.
        Chaque sous-classe DOIT implémenter cette méthode.
        """
        pass

    def _clean_string(self, text: str) -> str:
        """
        Cleans a string by normalizing unicode characters,
        replacing spaces and hyphens with underscores,
        removing non-alphanumeric characters,
        and collapsing multiple underscores.
        """
        if not isinstance(text, str):
            return text

        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

        text = self._RE_SPACES.sub("_", text)
        text = self._RE_NON_ALPHA.sub("", text)
        text = self._RE_MULTI_UNDERSCORE.sub("_", text)

        return text.strip("_")

    def _get_dict(self, input_dict: dict) -> dict:
        """
        Cleans the keys and values of a dictionary by normalizing unicode characters,
        replacing spaces and hyphens with underscores,
        removing non-alphanumeric characters,
        and collapsing multiple underscores.
        """
        for k, v in input_dict.items():
            if isinstance(v, dict):
                input_dict[k] = self._get_dict(v)
            elif isinstance(v, str):
                input_dict[k] = self._clean_string(v)
        return {self._clean_string(k): input_dict[k] for k in input_dict}
