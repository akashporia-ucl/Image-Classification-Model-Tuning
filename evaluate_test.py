import os
import sys
import io
import subprocess
import tempfile
from pyspark.sql import SparkSession
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

def load_image_from_hdfs_local(hdfs_path):
    """
    Loads image bytes from HDFS using a command-line call and returns a PIL Image.
    This function is intended for use inside the worker.
    """
    try:
        output = subprocess.check_output(["hdfs", "dfs", "-cat", hdfs_path])
        image = Image.open(io.BytesIO(output)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image from {hdfs_path}: {e}")
        return None

def evaluate_partition(partition_index, rows, images_base_path, hdfs_model_path):
    """
    This function runs on each worker, processing the rows in its partition.
    For each row (which is expected to be a Spark Row with a single key 'id' holding the filename):
      - Loads the image from HDFS
      - Applies the same transform as during training/inference
      - Loads the model from HDFS
      - Runs inference
      - Prints a log statement with the file name and predicted label
      - Returns an iterator over tuples: (id, predicted_label)
    """
    # Import libraries locally within this function.
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image
    import os, subprocess, tempfile, io

    def load_image(path):
        return load_image_from_hdfs_local(path)

    # Define the transformation.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a ResNet50 model and adjust for binary classification.
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    model.eval()

    # Load the model state from HDFS.
    local_tmp = tempfile.NamedTemporaryFile(delete=False)
    local_tmp.close()
    try:
        subprocess.check_call(["hdfs", "dfs", "-get", "-f", hdfs_model_path, local_tmp.name])
        state_dict = torch.load(local_tmp.name, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Partition {partition_index}: Error loading model from HDFS: {e}")
        os.remove(local_tmp.name)
        return iter([])  # Return an empty iterator on error.
    os.remove(local_tmp.name)

    results = []
    for row in rows:
        try:
            # Extract the expected column. In test.csv, the filename is in the column 'id'
            file_name = row["id"]
        except Exception as e:
            print(f"Partition {partition_index}: Error parsing row {row} - {e}")
            continue

        # Construct the full HDFS path for the image.
        hdfs_image_path = os.path.join(images_base_path, file_name)
        image = load_image(hdfs_image_path)
        if image is None:
            # Create a dummy image if loading fails.
            image = Image.new("RGB", (224, 224))
        image = transform(image)
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        pred_label = int(predicted.cpu().numpy()[0])
        print(f"Partition {partition_index}, File {file_name}: Predicted = {pred_label}")
        results.append((file_name, pred_label))
    return iter(results)

def main():
    # Initialize a Spark session.
    spark = SparkSession.builder.appName("Distributed_Evaluation_Test").getOrCreate()

    # File paths and HDFS paths.
    test_csv_path = "test.csv"  # Local CSV with a single column "id" containing filenames.
    images_base_path = "hdfs://management:9000/data"  # Base path where images reside.
    hdfs_model_path = "hdfs://management:9000/data/model_collated/resnet50_final.pt"  # Model file on HDFS.
    hdfs_output_csv = "hdfs://management:9000/data/distributed_evaluation_test_results.csv"

    # Read the CSV as a Spark DataFrame.
    df = spark.read.csv(test_csv_path, header=True, inferSchema=True)
    print(f"Loaded CSV with {df.count()} records. Columns: {df.columns}")

    # Convert the DataFrame to an RDD.
    rdd = df.rdd

    # Apply the evaluation function to each partition.
    rdd_results = rdd.mapPartitionsWithIndex(
        lambda idx, rows: evaluate_partition(idx, rows, images_base_path, hdfs_model_path)
    )

    # Collect results back to the driver.
    results = rdd_results.collect()

    # Convert collected results to a Pandas DataFrame.
    results_df = pd.DataFrame(results, columns=["id", "predicted_label"])

    # Save the per-image prediction results to a CSV locally.
    local_output_csv = "distributed_evaluation_test_results.csv"
    results_df.to_csv(local_output_csv, index=False)
    print(f"Saving distributed evaluation results locally to {local_output_csv}")

    # Upload the CSV to HDFS.
    try:
        output_dir = os.path.dirname(hdfs_output_csv)
        subprocess.check_call(["hdfs", "dfs", "-mkdir", "-p", output_dir])
        subprocess.check_call(["hdfs", "dfs", "-put", "-f", local_output_csv, hdfs_output_csv])
        print(f"Distributed evaluation results saved to {hdfs_output_csv}")
    except Exception as e:
        print("Error uploading evaluation results to HDFS:", e)
    os.remove(local_output_csv)

    spark.stop()

if __name__ == "__main__":
    main()
