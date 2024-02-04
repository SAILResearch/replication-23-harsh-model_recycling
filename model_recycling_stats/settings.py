from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    # Path to the directory where the data will be stored
    DATA_DIR: Path = Path(__file__).parent.parent / "results"
    RANDOM_SEED: int = 42
    EXP_ID: str = "apch"
    MODEL_HISTORY: int = 15

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

    # experiments
    MODEL_SELECTION_EXPERIMENTS: list = [
        "ms-exp1",
        "ms-exp2",
        "ms-exp3",
        "ms-exp4",
        "ms-exp5",
        "ms-exp6",
        "ms-exp7",
        "ms-exp8",
    ]
    MODEL_REUSE_EXPERIMENTS: list = [
        "mr-exp1",
        "mr-exp2",
        "mr-exp3",
        "mr-exp4",
    ]
    MODEL_STACKING_EXPERIMENTS: list = [
        "mst-exp1",
        "mst-exp2",
        "mst-exp3",
        "mst-exp4",
        "mst-exp5",
        "mst-exp6",
        "mst-exp7",
        "mst-exp8",
        "mst-exp9",
        "mst-exp10",
        "mst-exp11",
    ]
    MODEL_VOTING_EXPERIMENTS: list = [
        "mv-exp1",
        "mv-exp2",
        "mv-exp3",
        "mv-exp4",
    ]
    CLUSTERING_EXPERIMENTS: list = [
        "c-exp1",
        "c-exp2",
    ]
    EXPERIMENTS = (
        MODEL_SELECTION_EXPERIMENTS
        + MODEL_REUSE_EXPERIMENTS
        + MODEL_STACKING_EXPERIMENTS
        + MODEL_VOTING_EXPERIMENTS
        + CLUSTERING_EXPERIMENTS
    )

    RF_EXTRA_EXPERIMENTS: list = [
        "mst-window",
        "mst-shift",
        "mr-window",
        "mr-shift",
        "mv-window",
        "mv-shift",
        "ms-window",
        "ms-shift",
    ]

    LR_EXTRA_EXPERIMENTS: list = [
        "mst-window",
        "mst-shift",
        "mv-window",
        "mv-shift",
        "ms-window",
        "ms-shift",
    ]

    NN_EXTRA_EXPERIMENTS: list = [
        "mst-window",
        "mst-shift",
    ]


settings = Settings()
