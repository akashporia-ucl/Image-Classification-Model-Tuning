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

    run_spark_submit_resnet = BashOperator(
        task_id='run_spark_submit_resnet',
        bash_command="""
        cd /home/almalinux/Image-Classification/backend && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
          tune_resnet.py 1
        """
    )

    run_spark_submit_vgg16 = BashOperator(
        task_id='run_spark_submit_vgg16',
        bash_command="""
        cd /home/almalinux/Image-Classification/backend && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
          tune_vgg16.py 1
        """
    )

    run_spark_submit_inception = BashOperator(
        task_id='run_spark_submit_inception',
        bash_command="""
        cd /home/almalinux/Image-Classification/backend && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
          tune_inception.py 1
        """
    )

    run_collate_results = BashOperator(
        task_id='run_collate_results',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning/backend && \
        python collate.py
        """
    )

    run_evaluate_model_vgg16 = BashOperator(
        task_id='run_evaluate_model_vgg6',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning/backend && \
        python evaluate_vgg16.py
        """
    )

    run_evaluate_model_inception = BashOperator(
        task_id='run_evaluate_model_inception',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning/backend && \
        python evaluate_inception.py
        """
    )

    run_evaluate_model_resnet = BashOperator(
        task_id='run_evaluate_model_inception',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning/backend && \
        python evaluate_resnet.py
        """
    )

    publish_result_event = BashOperator(
        task_id='publish_result',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning/backend && \
        python publisher.py
        """
    )

    [run_spark_submit_resnet, run_spark_submit_vgg16, run_spark_submit_inception] >> run_collate_results >> [run_evaluate_model_vgg16, run_evaluate_model_inception, run_evaluate_model_resnet] >> publish_result_event
