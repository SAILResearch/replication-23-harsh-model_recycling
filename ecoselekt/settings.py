import math
from pathlib import Path
from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    # Path to the directory where the data will be stored
    DATA_DIR: Path = Path(__file__).parent / "data"
    # Path to the directory where the models will be stored (e.g. KNN, RF, etc.)
    MODELS_DIR = Path(__file__).parent / "models"
    RANDOM_SEED: int = 42
    # sliding window splits
    WINDOW_SIZE: int = 1000
    SHIFT: int = 200
    TEST_SIZE: int = 200
    EXP_ID: str = "apch"
    MODEL_HISTORY: int = 15
    # featuer selection
    K_FEATURES: int = 100
    # wilcoxon test
    ALPHA: float = 0.05 / MODEL_HISTORY

    # apachejit
    PROJECTS = [
        "activemq",
        "camel",
        "cassandra",
        "flink",
        "groovy",
        "hbase",
        "hive",
        "ignite",
    ]

    # derived constants
    C_TEST_WINDOWS: int = math.ceil(TEST_SIZE / SHIFT)
    F_TEST_WINDOWS: int = TEST_SIZE // SHIFT


settings = Settings()
