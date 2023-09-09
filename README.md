# Model Recycling

- RQ1 – The results for RQ1 are available under `/rq1.ipynb` notebook [here](https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/rq1).
- RQ2 – To look at the analysis of all variant results, please refer to `/analysis.ipynb` notebook.
- RQ3 - For latency evaluation of best variants, please refer to `/latency.ipynb` notebook under their branch listed in [variants](#variants) table.

## Variants

The results for each project are available in `/experiment-apache-<<project_name_here>>.ipynb` notebook under each recycling strategy variant, which resides in its independent branch as listed below.

| **Strategy**       | **# Variant** | **Link to Variant Branch**                                                  |
| ------------------ | ------------- | --------------------------------------------------------------------------- |
| Model Selection    | 1             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp1   |
|                    | 2             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp2   |
|                    | 3             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp3   |
|                    | 4             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp4   |
|                    | 5             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp5   |
|                    | 6             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp6   |
|                    | 7             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp7   |
|                    | 8             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp8   |
| Single Model Reuse | 1             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mr-exp1   |
|                    | 2             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mr-exp2   |
|                    | 3             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mr-exp3   |
|                    | 4             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mr-exp4   |
| Model Stacking     | 1             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp1  |
|                    | 2             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp2  |
|                    | 3             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp3  |
|                    | 4             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp4  |
|                    | 5             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp5  |
|                    | 6             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp6  |
|                    | 7             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp7  |
|                    | 8             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp8  |
|                    | 9             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp9  |
|                    | 10            | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp10 |
|                    | 11            | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp11 |
| Model Voting       | 1             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mv-exp1   |
|                    | 2             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mv-exp2   |
|                    | 3             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mv-exp3   |
|                    | 4             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mv-exp4   |
| Clustering         | 1             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/c-exp1    |
|                    | 2             | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/c-exp2    |

## Extra Variants

The variants with different window size and shift size are listed below.

| **Strategy**       | **# Variant** | **Link to Variant Branch**                                                  |
| ------------------ | ------------- | --------------------------------------------------------------------------- |
| Model Selection    | w=2000        | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp6w  |
|                    | s=300         | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/ms-exp6s  |
| Single Model Reuse | w=2000        | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mr-exp2w  |
|                    | s=300         | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mr-exp2s  |
| Model Stacking     | w=2000        | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp5w |
|                    | s=300         | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mst-exp5s |
| Model Voting       | w=2000        | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mv-exp1w  |
|                    | s=300         | https://github.com/SAILResearch/wip-23-harsh-model_recycling/tree/mv-exp1s  |

## How to Run Experiment

### Pre-requisites

- [Python >= 3.9](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/)
### Steps

1. Checkout the variant branch:

```shellscript
git checkout <<branch_name_here>>
```
2. Download the commit metrics from this [link](https://github.com/harsh8398/apachejit/blob/14e29628584160037139a7d111fcfdfb593fa700/dataset/apachejit_total.csv) and put it in `/ecoselekt/data/` directory.

3. Download the "ApacheJIT Commits" zip from the [releases](https://github.com/SAILResearch/wip-23-harsh-model_recycling/releases) and extract it in `/ecoselekt/data/` directory.

4. Install dependencies:

```shellscript
poetry install
```

5. Create time windows and train RFS model versions:

```shellscript
make mlms
```

6. Evaluate RFS models and create a baseline:

```shellscript
make eval_mlms
```

7. [Required only for Model Stacking and Model Voting variants] Install ScottKnottESD R package

```shellscript
poetry run setup_r
```

8. Prepare for recycling:

```shellscript
make prep_reuse
```

[Optional] – To evaluate inference latency results (script only present under the best experiments branches):

```shellscript
poetry run inf_latency
```

9. Inference with recycling:

```shellscript
make inf_reuse
```

10. Evaluate recycling:

```shellscript
make eval_reuse
```

11. Evaluate historical models over time windows:

```shellscript
poetry run process_selekt_vs_best
```

12. Finally, run statistical tests on results:

```shellscript
poetry run stat_test
```

By default, each step stores output results in the `/ecoselekt/data/` directory, while the models are stored in `ecoselekt/models/` directory.