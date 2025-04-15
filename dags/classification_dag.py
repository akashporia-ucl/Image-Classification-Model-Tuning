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
    dag_id='classification_dag',
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
        cd /home/almalinux/Image-Classification-Model-Tuning && \
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
        cd /home/almalinux/Image-Classification-Model-Tuning && \
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
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
          tune_inception.py 1
        """
    )

    run_collate_results_for_resnet = BashOperator(
        task_id='run_collate_results_for_resnet',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python collate.py resnet50
        """
    )

    run_collate_results_for_inception = BashOperator(
        task_id='run_collate_results_for_inception',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python collate.py inceptionv3
        """
    )

    run_collate_results_for_vgg16 = BashOperator(
        task_id='run_collate_results_for_vgg16',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python collate.py vgg16
        """
    )

    run_evaluate_model_vgg16 = BashOperator(
        task_id='run_evaluate_model_vgg16',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
        evaluate_vgg16.py
        """
    )

    run_evaluate_model_inception = BashOperator(
        task_id='run_evaluate_model_inception',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
        evaluate_inception.py
        """
    )

    run_evaluate_model_resnet = BashOperator(
        task_id='run_evaluate_model_resnet',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning && \
        spark-submit \
          --master spark://management:7077 \
          --deploy-mode client \
          --conf spark.executor.instances=4 \
          --conf spark.executor.cores=2 \
          --conf spark.executor.memory=4G \
        evaluate_resnet.py
        """
    )

    publish_update_for_resnet = BashOperator(
        task_id='publish_result_for_resnet',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python publisher.py "Tuning completed for ResNet"
        """
    )

    publish_update_for_inception = BashOperator(
        task_id='publish_result_for_inception',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python publisher.py "Tuning completed for Inception"
        """
    )

    publish_update_for_vgg16 = BashOperator(
        task_id='publish_result_for_vgg16',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python publisher.py "Tuning completed for VGG16"
        """
    )

    publish_result_event = BashOperator(
        task_id='publish_result',
        bash_command="""
        cd /home/almalinux/Image-Classification-Model-Tuning &&
        /usr/bin/python publisher.py "Model tuning completed"
        """
    )

    #[run_spark_submit_resnet, run_spark_submit_vgg16, run_spark_submit_inception] >> run_collate_results >> [run_evaluate_model_vgg16, run_evaluate_model_inception, run_evaluate_model_resnet] >> publish_result_event
    #[run_spark_submit_resnet, run_spark_submit_vgg16, run_spark_submit_inception] >> run_collate_results >> publish_result_event

    # [
    #     [run_spark_submit_resnet >> run_collate_results_for_resnet >> run_evaluate_model_resnet >> publish_update_for_resnet],
    #     [run_spark_submit_inception >> run_collate_results_for_inception >> run_evaluate_model_inception >> publish_update_for_inception],
    #     [run_spark_submit_vgg16 >> run_collate_results_for_vgg16 >> run_evaluate_model_vgg16 >> publish_update_for_vgg16]
    # ] >> publish_result_event

    [
        run_spark_submit_resnet >> run_collate_results_for_resnet >> run_evaluate_model_resnet >> publish_update_for_resnet,
        run_spark_submit_inception >> run_collate_results_for_inception >> run_evaluate_model_inception >> publish_update_for_inception,
        run_spark_submit_vgg16 >> run_collate_results_for_vgg16 >> run_evaluate_model_vgg16 >> publish_update_for_vgg16
    ] >> publish_result_event
