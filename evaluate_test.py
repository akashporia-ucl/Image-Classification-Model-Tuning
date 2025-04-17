import os
import io
import subprocess
from pyspark.sql import SparkSession
import pandas as pd
from PIL import Image
import tempfile
import torch
import torch.nn as nn
from torchvision import transforms, models


def load_image_from_hdfs(path):
    """
    Fetches image bytes from HDFS and returns a PIL Image in RGB.
    """
    try:
        data = subprocess.check_output(["hdfs", "dfs", "-cat", path])
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None


def evaluate_partition(partition_index, rows, images_base_path, hdfs_model_path):
    """
    Processes a partition of rows: loads images, applies model inference,
    and yields (id, predicted_label).
    """
    # Local imports inside executors
    import torch
    from torchvision import transforms, models
    from PIL import Image
    import os

    # Device and preprocessing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Build model and load broadcasted weights
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    #model.load_state_dict(bc_state.value)
    model.to(device)
    model.eval()

    local_tmp = tempfile.NamedTemporaryFile(delete=False)
    local_tmp.close()

    try:
        subprocess.check_call(["hdfs", "dfs", "-get", "-f", hdfs_model_path, local_tmp.name])
        state_dict = torch.load(local_tmp.name, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model from HDFS: {e}")
        os.remove(local_tmp.name)
        return iter([])
    os.remove(local_tmp.name)

    results = []
    for row in rows:
        try:
            img_id = row["id"]  # matches test.csv column
        except Exception as e:
            print(f"Partition {partition_index}: invalid row {row} - {e}")
            continue

        hdfs_img_path = os.path.join(images_base_path, img_id)
        image = load_image_from_hdfs(hdfs_img_path) or Image.new("RGB", (224, 224))
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
        pred_label = int(pred.cpu().item())

        print(f"Partition {partition_index}, {img_id}: Predicted={pred_label}")
        results.append((img_id, pred_label))

    return iter(results)


def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Distributed_Test_Evaluation") \
        .getOrCreate()

    # Number of partitions (default 32)
    num_partitions = int(spark.conf.get("spark.myApp.numPartitions", "32"))

    # HDFS paths
    test_csv_path   = "hdfs://management:9000/data/test.csv"
    images_base     = "hdfs://management:9000/data"
    hdfs_model_path = "hdfs://management:9000/data/model_collated/resnet50_final.pt"
    hdfs_output_csv = "hdfs://management:9000/data/distributed_evaluation_test_results.csv"

    # Broadcast the model state_dict once
    #state_dict = torch.load(hdfs_model_path, map_location="cpu")
    #bc_state = spark.sparkContext.broadcast(state_dict)

    # Read test.csv and repartition
    df = spark.read.csv(test_csv_path, header=True, inferSchema=True)
    print(f"Loaded test CSV with {df.count()} rows. Columns: {df.columns}")
    df = df.repartition(num_partitions)

    # Apply inference
    rdd = df.rdd.mapPartitionsWithIndex(
        lambda idx, rows: evaluate_partition(idx, rows, images_base, hdfs_model_path)
    )

    # Collect and save locally
    results = rdd.collect()
    results_df = pd.DataFrame(results, columns=["id", "predicted_label"])
    local_out = "test_predictions.csv"
    results_df.to_csv(local_out, index=False)
    print("Saved local predictions to {}".format(local_out))

    # Upload back to HDFS
    try:
        subprocess.check_call(["hdfs", "dfs", "-mkdir", "-p", os.path.dirname(hdfs_output_csv)])
        subprocess.check_call(["hdfs", "dfs", "-put", "-f", local_out, hdfs_output_csv])
        print(f"Uploaded predictions to {hdfs_output_csv}")
    except Exception as e:
        print(f"Error uploading to HDFS: {e}")
    finally:
        os.remove(local_out)

    spark.stop()


if __name__ == "__main__":
    main()
