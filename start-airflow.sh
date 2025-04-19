#!/usr/bin/env bash
set -euo pipefail

# 1. Change to your project directory
cd /home/almalinux/Image-Classification-Model-Tuning

# 2. Activate the Python virtual environment
source py_env/bin/activate

# 3. Set Airflow environment variables
export AIRFLOW__CORE__SQL_ALCHEMY_CONN="sqlite:////tmp/airflow.db"
export AIRFLOW_HOME="."

# 4. Initialise the Airflow metadata database
airflow db init

# 5. Create the admin user non‑interactively
airflow users create \
  --username admin \
  --firstname firstname \
  --lastname lastname \
  --role Admin \
  --email admin@airflow.com \
  --password admin

# 6. Launch the webserver on port 7777 (in background)
airflow webserver --port 7777 &

# 7. Launch the scheduler (in foreground)
airflow scheduler &