# Model Recycling

*Abstract* – Once a Machine Learning (ML) model is deployed, the same model is typically retrained from scratch, either on a scheduled interval or as soon as model drift is detected, to make sure the model reflects current data distributions and performance experiments. As such, once a new model is available, the old model typically is discarded. This paper challenges the notion of older models being useless by showing that old models still have substantial value compared to newly trained models, and by proposing novel post-deployment model recycling techniques that help make informed decisions on which old models to reuse and when to reuse. In an empirical study on eight long-lived Apache projects comprising a total of 84,343 commits, we analyze the performance of five model recycling strategies on Just-In-Time defect prediction models. Comparison against traditional model retraining from scratch (RFS) shows that our approach significantly outperforms RFS in terms of recall, g-mean and AUC by up to a median of $26\%$, $21\%$ and $1\%$, respectively, with the best recycling strategy (Model Stacking) outperforming the baseline in over $50\%$ of the projects. Our recycling strategies provide this performance improvement at the cost of a median of $2$x to $6$-$10$x slower time-to-inference compared to RFS, depending on the selected strategy and variant.

## Results

- RQ1 – The results for RQ1 are available under `/rq1.ipynb` notebook [here](https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/rq1).
- RQ2 – To look at the analysis of all variant results, please refer to the following notebooks
  - `/analysis.ipynb` notebook for random forest variants
  - `/analysis-lr.ipynb` notebook for logistic regression variants
  - `/analysis-nn.ipynb` notebook for deep neural network variants
- RQ3 - For latency evaluation of best variants, please refer to `/latency.ipynb` notebook under their branch listed in [variants](#variants) table.

## Variants

The source code to replicate the results for each project under each recycling strategy variant is available in its independent branch as listed below.

| **Strategy**       | **# Variant** | **Link to Variant Branch**                                                          |
| ------------------ | ------------- | ----------------------------------------------------------------------------------- |
| Model Selection    | 1             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp1   |
|                    | 2             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp2   |
|                    | 3             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp3   |
|                    | 4             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp4   |
|                    | 5             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp5   |
|                    | 6             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp6   |
|                    | 7             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp7   |
|                    | 8             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp8   |
| Single Model Reuse | 1             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mr-exp1   |
|                    | 2             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mr-exp2   |
|                    | 3             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mr-exp3   |
|                    | 4             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mr-exp4   |
| Model Stacking     | 1             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp1  |
|                    | 2             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp2  |
|                    | 3             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp3  |
|                    | 4             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp4  |
|                    | 5             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp5  |
|                    | 6             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp6  |
|                    | 7             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp7  |
|                    | 8             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp8  |
|                    | 9             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp9  |
|                    | 10            | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp10 |
|                    | 11            | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp11 |
| Model Voting       | 1             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mv-exp1   |
|                    | 2             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mv-exp2   |
|                    | 3             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mv-exp3   |
|                    | 4             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mv-exp4   |
| Clustering         | 1             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/c-exp1    |
|                    | 2             | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/c-exp2    |

## Extra Variants

The variants with different window sizes and shift sizes are listed below.

| **Strategy**       | **# Variant** | **Link to Variant Branch**                                                          |
| ------------------ | ------------- | ----------------------------------------------------------------------------------- |
| Model Selection    | w=2000        | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp6w  |
|                    | s=300         | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/ms-exp6s  |
| Single Model Reuse | w=2000        | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mr-exp2w  |
|                    | s=300         | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mr-exp2s  |
| Model Stacking     | w=2000        | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp5w |
|                    | s=300         | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mst-exp5s |
| Model Voting       | w=2000        | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mv-exp1w  |
|                    | s=300         | https://github.com/SAILResearch/replication-23-harsh-model_recycling/tree/mv-exp1s  |

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

3. Download the "ApacheJIT Commits" zip from the [releases](https://github.com/SAILResearch/replication-23-harsh-model_recycling/releases) and extract it in `/ecoselekt/data/` directory.

4. Install dependencies:

```shellscript
poetry install
```

5. Create time windows and train RFS model versions:

```shellscript
# train random forest models
make mlms
# train logistic regression models
make lr
# train neural network models
make nn
```

6. Evaluate RFS models and create a baseline:

```shellscript
# evaluate random forest models
poetry run eval_models
# evaluate logistic regression models
poetry run eval_lr
# evaluate neural network models
poetry run eval_nn
```

7. [Required only for Model Stacking and Model Voting variants] Install ScottKnottESD R package

```shellscript
poetry run setup_r
```

8. Prepare for recycling:

```shellscript
# prepare for random forest models
make prep_reuse
# prepare for logistic regression models
make prep_reuse_lr
# prepare for neural network models
make prep_reuse_nn
```

[Optional] – To evaluate inference latency results (script only present under the best experiments branches):

```shellscript
# for rf
poetry run inf_latency
# for lr
poetry run inf_latency_lr
# for nn
poetry run inf_latency_nn
```

9. Inference with recycling:

```shellscript
# for rf
make inf_reuse
# for lr
make inf_reuse_lr
# for nn
make inf_reuse_nn
```

10. Evaluate recycling:

```shellscript
# for rf
make eval_reuse
# for lr
make eval_reuse_lr
# for nn
make eval_reuse_nn
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

## Authors

- Harsh Patel [@harsh8398](https://github.com/harsh8398)
- Dr. Bram Adams
- Dr. Ahmed E. Hassan