mlms: pyproject.toml
	caffeinate poetry run train_models

lr: pyproject.toml
	caffeinate poetry run train_lr

eval_mlms: pyproject.toml
	caffeinate poetry run eval_models

prep_reuse: pyproject.toml
	caffeinate poetry run train_selekt

prep_reuse_lr: pyproject.toml
	caffeinate poetry run train_selekt_lr

eval_reuse: pyproject.toml
	caffeinate poetry run eval_selekt

eval_reuse_lr: pyproject.toml
	caffeinate poetry run eval_selekt_lr

inf_reuse: pyproject.toml
	caffeinate poetry run inference_selekt

inf_reuse_lr: pyproject.toml
	caffeinate poetry run inference_selekt_lr

copy: pyproject.toml
	cp -r ecoselekt/data/apch_activemq_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_camel_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_flink_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_groovy_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_ignite_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_hbase_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_hive_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_cassandra_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_stats.csv results/exp_apch