## How to Run

### Pre-requisites

- [Python >= 3.9](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/)
### Steps

1. Download the commit metrics from this [link](https://github.com/harsh8398/apachejit/blob/14e29628584160037139a7d111fcfdfb593fa700/dataset/apachejit_total.csv) and put it in `/ecoselekt/data/` directory.

2. Download the "ApacheJIT Commits" zip from the [releases](https://github.com/SAILResearch/replication-23-harsh-model_recycling/releases) and extract it in `/ecoselekt/data/` directory.

3. Install dependencies:

```shellscript
poetry install
```

4. Create time windows and train RFS model versions:

```shellscript
make mlms
```

5. Evaluate RFS models and create a baseline:

```shellscript
make eval_mlms
```

7. Calculate SHAP values:

```shellscript
make calc_shap
```

8. Calculate feature survival rate:

```shellscript
make calc_surv
```

9. Calculate feature survival rates residing outside confidence interval (OCI):

```shellscript
make check_oci
```

The output results are stored in `/ecoselekt/data/` directory.