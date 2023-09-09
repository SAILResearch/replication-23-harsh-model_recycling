mlms: pyproject.toml
	caffeinate poetry run train_models

eval_mlms: pyproject.toml
	caffeinate poetry run eval_models

calc_shap: pyproject.toml
	caffeinate poetry run calc_drift_imp

calc_surv: pyproject.toml
	caffeinate poetry run calc_surv_shap

check_oci: pyproject.toml
	caffeinate poetry run oci_check

copy: pyproject.toml
	cp -r ecoselekt/data/apch_activemq_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_camel_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_flink_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_groovy_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_ignite_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_hbase_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_hive_*.csv results/exp_apch && \
	cp -r ecoselekt/data/apch_cassandra_*.csv results/exp_apch