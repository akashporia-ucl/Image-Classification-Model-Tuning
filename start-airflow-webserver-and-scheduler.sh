#!/usr/bin/env bash
set -euo pipefail

cd /home/almalinux/Image-Classification-Model-Tuning
source py_env/bin/activate

export AIRFLOW__CORE__SQL_ALCHEMY_CONN="sqlite:////tmp/airflow.db"
export AIRFLOW_HOME="."

airflow db init

# ignore “user already exists” errors
airflow users create \
  --username admin \
  --firstname firstname \
  --lastname lastname \
  --role Admin \
  --email admin@airflow.com \
  --password admin || true

# Launch webserver in a detached screen session (and log to a file)
screen -dmS webserver bash -c \
  'airflow webserver --port 7777'

# Launch scheduler in a detached screen session
screen -dmS scheduler bash -c \
  'airflow scheduler'
