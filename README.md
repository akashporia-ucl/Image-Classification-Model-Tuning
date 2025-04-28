# Image Classification Model Tuning

## Overview

This repository presents a Python-based framework dedicated to the fine-tuning and evaluation of image classification models. It encompasses scripts for hyperparameter tuning, model evaluation, and auxiliary utilities, facilitating streamlined experimentation and performance assessment.

- **dags/**: Presumably contains Directed Acyclic Graphs for workflow management tools such as Apache Airflow.
- **collate.py**: Handles data collation and preprocessing tasks.
- **evaluate_test.py**: Evaluates model performance on the test dataset.
- **evaluate_train.py**: Evaluates model performance on the training dataset.
- **publisher.py**: Manages the publishing or logging of results.
- **request.py**: Handles HTTP requests or API interactions.
- **tune_resnet.py**: Fine-tunes a ResNet model architecture.

## Prerequisites

- Python 3.x
- Common libraries such as:
  - `torch`
  - `torchvision`
  - `numpy`
  - `pandas`
  - `requests`
  - `sklearn`
  - `matplotlib`

*Please install missing packages manually using pip, as no `requirements.txt` file is included.*

## Installation and Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/akashporia-ucl/Image-Classification-Model-Tuning.git
    cd Image-Classification-Model-Tuning
    ```

2. (Optional) Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install required packages:

    ```bash
    pip install torch torchvision numpy pandas requests scikit-learn matplotlib
    ```
