import importlib

from src.config import load_dataset_config


def get_dataset(dataset_name: str) -> object:
    """
    The dataset name must be in snake_case, and the dataset class must in PascalCase
    """
    config = load_dataset_config(dataset_name)
    dataset_type = config.get("type")
    class_name = dataset_type.title().replace("_", "")

    try:
        module_path = f"src.datasets.{dataset_type}"
        module = importlib.import_module(module_path)

        dataset_class = getattr(module, class_name)

        return dataset_class(config)

    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Impossible de charger le dataset '{dataset_type}'. "
            f"Vérifiez que le fichier src/datasets/{dataset_type}.py et la classe {class_name} existent."  # noqa: E501
        ) from e
