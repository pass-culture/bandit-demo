from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent

SQL_DIR = ROOT_DIR / "sql"
SQL_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LANCE_DB_DIR = ROOT_DIR / "lance_db"
LANCE_DB_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_config(dataset_name: str) -> dict:
    """
    Charge le fichier de configuration YAML associé à un dataset.
    Exemple : load_dataset_config("dataset_b")
    """

    config_path = ROOT_DIR / "configs" / f"{dataset_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Le fichier de configuration n'existe pas : {config_path}"
        )

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
