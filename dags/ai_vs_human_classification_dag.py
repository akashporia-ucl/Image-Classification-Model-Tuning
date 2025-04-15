from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

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
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['spark', 'resnet', 'vgg16'],
) as dag:

    spark_submit_for_tuning = BashOperator(
        task_id='spark_submit_for_tuning',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
          tune_resnet.py 2
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
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
        evaluate_train.py
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
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
          evaluate_test.py
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

    spark_submit_for_tuning >> publish_model_tuning_event >> collate_model_partitions >> publish_model_collate_event >> \
    [spark_submit_evaluate_model_for_train_data >> publish_evaluate_train_event,
     spark_submit_evaluate_model_for_test_data >> publish_evaluate_test_event] >> publish_result_event
