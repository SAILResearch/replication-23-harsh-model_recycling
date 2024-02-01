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
    MAX_KNEIGHBOURS: int = 20
    CURRENT_KNEIGHBOURS: int = 20
    KNN_ALGO: str = "ball_tree"
    # featuer selection
    K_FEATURES: int = 100
    # wilcoxon test
    ALPHA: float = 0.05 / MODEL_HISTORY

    # deepjit settings
    MSG_LENGTH: int = 256
    CODE_LENGTH: int = 512
    CODE_LINE: int = 10
    FILTER_SIZES: List[int] = [1, 2, 3]
    NUM_FILTERS: int = 16
    NUM_EPOCHS: int = 50
    EMBEDDING_DIM: int = 16
    DROP_OUT: float = 0.5
    HIDDEN_UNITS: int = 512
    BATCH_SIZE: int = 64
    CLASS_NUM: int = 1
    LR: float = 1e-5

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
