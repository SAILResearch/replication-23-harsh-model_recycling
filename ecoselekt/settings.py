import math
from pathlib import Path
from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    # Path to the directory where the data will be stored
    DATA_DIR: Path = Path(__file__).parent / "data"
    # Path to the directory where the models will be stored (e.g. KNN, RF, etc.)
    MODELS_DIR = Path(__file__).parent / "models"
    PROJECTS: List[str] = ["openstack", "qt"]
    SAMPLING_METHOD: str = "DE_SMOTE_min_df_3"
    # bootstrap sampling
    # NSAMPLES: int = 10
    RANDOM_SEED: int = 42
    # sliding window splits
    WINDOW_SIZE: int = 1000
    SHIFT: int = 200
    TEST_SIZE: int = 200
    EXP_ID: str = "apch"
    # MAX_WINDOW: int = 11
    # knn
    MODEL_HISTORY: int = 15
    MAX_KNEIGHBOURS: int = 20
    CURRENT_KNEIGHBOURS: int = 20
    KNN_ALGO: str = "ball_tree"
    # featuer selection
    K_FEATURES: int = 100
    K_FEATURES_FOR_SELEKT: int = 100
    # wilcoxon test
    ALPHA: float = 0.05 / MODEL_HISTORY

    # apachejit
    PROJECTS = [
        "activemq",
        "camel",
        "cassandra",
        "flink",
        "groovy",
        # "hadoop", #!: no positive labels
        # "hadoop-hdfs",
        # "hadoop-mapreduce",
        "hbase",
        "hive",
        "ignite",
        # "kafka",
        # "spark",
        # "zeppelin",
        # "zookeeper",
    ]

    # derived constants
    C_TEST_WINDOWS: int = math.ceil(TEST_SIZE / SHIFT)
    F_TEST_WINDOWS: int = TEST_SIZE // SHIFT

    # inference settings
    SELEKT_TOLERANCE: float = 0.2
    GRAIN_SIZE: int = 42


settings = Settings()
