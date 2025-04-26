from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.exceptions import AirflowException
from datetime import datetime, timedelta

def ensure_var(key: str, default: str) -> str:
    """
    Ensure that the Airflow Variable `key` exists.
    If it does not, set it to `default` and return that value.
    """
    try:
        return Variable.get(key)
    except (AirflowException, KeyError):
        Variable.set(key, default)
        return default

# On import, guarantee the defaults exist; tasks will pick up the latest via Jinja
for key, default in [
    (
        'train_data_path',
        'hdfs://management:9000/data/train_data/*'
    ),
    (
        'train_csv_path',
        'hdfs://management:9000/data/train.csv'
    ),
    (
        'sample_data_path',
        'hdfs://management:9000/data/sample.csv'
    ),
    ('tuning_time', '6.5'),
    (
        'images_base_path',
        'hdfs://management:9000/data'
    ),
    (
        'test_csv_path',
        'hdfs://management:9000/data/test.csv'
    )
]:
    ensure_var(key, default)

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='ai_vs_human_classification_dag',
    default_args=default_args,
    description='A DAG that triggers spark-submit jobs, collate results, chooses best result and publishes an event via BashOperator',
    schedule_interval='@once',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    tags=['spark', 'resnet', 'ai_vs_human'],
) as dag:

    spark_submit_for_tuning = BashOperator(
        task_id='spark_submit_for_tuning',
        bash_command="""
        export PYSPARK_PYTHON=/usr/bin/python3
        export PYSPARK_DRIVER_PYTHON=/usr/bin/python3
        export PATH=/home/almalinux/hadoop-3.4.0/bin:$PATH
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        /home/almalinux/spark-3.5.3-bin-hadoop3-scala2.13/bin/spark-submit \
            --master spark://management:7077 \
            --deploy-mode client \
            --conf spark.pyspark.python=/usr/bin/python3 \
            --conf spark.pyspark.driver.python=/usr/bin/python3 \
            --conf spark.executor.instances=4 \
            --conf spark.executor.cores=3 \
            --conf spark.executor.memory=24G \
            --conf spark.driver-memory=4G \
            tune_resnet.py \
            --tune_time {{var.value.tuning_time}}  \
            --csv_path {{var.value.train_csv_path}} \
            --image_path {{var.value.train_data_path}} \

        """
    )

    publish_model_tuning_event = BashOperator(
        task_id='publish_model_tuning_event',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python publisher.py "Tuning completed for ResNet"
        """
    )

    collate_model_partitions = BashOperator(
        task_id='collate_model_partitions',
        bash_command="""
        export PATH=/home/almalinux/hadoop-3.4.0/bin:$PATH && \
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python collate.py resnet50
        """
    )

    publish_model_collate_event = BashOperator(
        task_id='publish_model_collate_event',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python publisher.py "Collate model partitions completed"
        """
    )

    spark_submit_evaluate_model_for_train_data = BashOperator(
        task_id='spark_submit_evaluate_model_for_train_data',
        bash_command="""
        export PYSPARK_PYTHON=/usr/bin/python3
        export PYSPARK_DRIVER_PYTHON=/usr/bin/python3
        export PATH=/home/almalinux/hadoop-3.4.0/bin:$PATH
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        /home/almalinux/spark-3.5.3-bin-hadoop3-scala2.13/bin/spark-submit \
        --master spark://management:7077 \
        --deploy-mode client \
        --conf spark.pyspark.python=/usr/bin/python3 \
        --conf spark.pyspark.driver.python=/usr/bin/python3 \
        --conf spark.executor.instances=4 \
        --conf spark.executor.cores=3 \
        --conf spark.executor.memory=28G \
        --conf spark.myApp.numPartitions=32 \
        evaluate_train.py \
        --csv_path {{var.value.train_csv_path}} \
        --images_base_path {{var.value.images_base_path}} \
        """
    )

    publish_evaluate_train_event = BashOperator(
        task_id='publish_evaluate_train_event',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python publisher.py "Evaluate train data completed"
        """
    )

    spark_submit_evaluate_model_for_test_data = BashOperator(
        task_id='spark_submit_evaluate_model_for_test_data',
        bash_command="""
        export PYSPARK_PYTHON=/usr/bin/python3
        export PYSPARK_DRIVER_PYTHON=/usr/bin/python3
        export PATH=/home/almalinux/hadoop-3.4.0/bin:$PATH
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        /home/almalinux/spark-3.5.3-bin-hadoop3-scala2.13/bin/spark-submit \
        --master spark://management:7077 \
        --deploy-mode client \
        --conf spark.pyspark.python=/usr/bin/python3 \
        --conf spark.pyspark.driver.python=/usr/bin/python3 \
        --conf spark.executor.instances=4 \
        --conf spark.executor.cores=3 \
        --conf spark.executor.memory=28G \
        --conf spark.myApp.numPartitions=16 \
        evaluate_test.py \
        --csv_path {{var.value.test_csv_path}} \
        --images_base_path {{var.value.images_base_path}} \
        """
    )

    publish_evaluate_test_event = BashOperator(
        task_id='publish_evaluate_test_event',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python publisher.py "Evaluate test data completed"
        """
    )


    publish_result_event = BashOperator(
        task_id='publish_result',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python publisher.py "Model tuning completed"
        """
    )

    delete_request_session = BashOperator(
        task_id='delete_request_session',
        bash_command="""
        sudo -u almalinux bash -lc "for session in \$(/usr/bin/screen -ls | awk '/request/ {print \$1}'); do
        /usr/bin/screen -S \"\$session\" -X quit
        done"
        """
    )

    start_request_session = BashOperator(
        task_id='start_request_session',
        bash_command="""
        sudo -u almalinux bash -lc "cd /home/almalinux/Image-Classification-Model-Tuning && \
        /usr/bin/screen -dmS request /usr/bin/python /home/almalinux/Image-Classification-Model-Tuning/request.py && \
        echo \"Started new screen session 'request'\""
        """
    )

    spark_submit_for_tuning >> publish_model_tuning_event >> collate_model_partitions >> publish_model_collate_event
    publish_model_collate_event >> spark_submit_evaluate_model_for_train_data >> publish_evaluate_train_event >> delete_request_session
    publish_model_collate_event >> spark_submit_evaluate_model_for_test_data >> publish_evaluate_test_event >> delete_request_session
    delete_request_session >> start_request_session >> publish_result_event
