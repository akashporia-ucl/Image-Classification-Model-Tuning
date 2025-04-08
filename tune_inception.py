import os
import io
import sys
import time
import tempfile
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, trim, col

def process_partition(partition_index, partition_data, mapping_bc, tune_time_hours):
    """
    Each executor processes its partition:
      - Loads a pretrained InceptionV3 model (with aux_logits disabled) and adapts its final layer for binary classification.
      - Repeatedly iterates over its partition data until the specified tuning time (in hours) has elapsed.
        For each image:
            * Extracts the image’s base name.
            * Looks up its label from the broadcast mapping.
            * Loads and preprocesses the image using InceptionV3-specific transforms.
            * Performs one training step (forward pass, loss computation, backward pass, and optimizer update).
      - After tuning, the model’s state dictionary is saved to a temporary file
        and then uploaded to HDFS under /data/model_partitions/inceptionv3.
    """
    # Import necessary libraries on the worker.
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image

    # Load pretrained InceptionV3 with aux_logits disabled.
    model = models.inception_v3(pretrained=True, aux_logits=False)
    # Replace the final fully connected layer.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # InceptionV3 expects 299x299 images.
    # Recommended preprocessing: Resize to 342, then center crop to 299.
    transform = transforms.Compose([
        transforms.Resize(342),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mapping = mapping_bc.value

    # Convert tuning time (hours) to seconds.
    tuning_time_sec = tune_time_hours * 3600
    start_time = time.time()

    # Since the iterator may be exhausted, collect partition data into a list.
    data_list = list(partition_data)

    # Loop until tuning time is reached.
    while time.time() - start_time < tuning_time_sec:
        for file_path, file_content in data_list:
            if time.time() - start_time >= tuning_time_sec:
                break
            try:
                base_name = os.path.basename(file_path).strip().lower()
                if base_name not in mapping:
                    print(f"Partition {partition_index}: Label not found for {base_name}, skipping.")
                    continue

                label_val = mapping[base_name]
                label_tensor = torch.tensor([label_val])
                
                image = Image.open(io.BytesIO(file_content)).convert("RGB")
                image = transform(image)
                image = image.unsqueeze(0)
                
                output = model(image)
                loss = criterion(output, label_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"Partition {partition_index}: Processed {base_name} with loss {loss.item()}")
            except Exception as e:
                print(f"Partition {partition_index}: Error processing {file_path}: {e}")

    # Save the tuned model parameters into an in-memory buffer.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model_bytes = buffer.getvalue()

    # Write the model to a temporary file.
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(model_bytes)
        tmp_path = tmp_file.name

    # Ensure the target HDFS directory exists.
    try:
        subprocess.check_call(["hdfs", "dfs", "-mkdir", "-p", "/data/model_partitions/inceptionv3"])
    except Exception as e:
        print(f"Partition {partition_index}: Error ensuring HDFS directory exists: {e}")

    # Define the target HDFS path.
    hdfs_output_path = f"/data/model_partitions/inceptionv3/inceptionv3_model_partition_{partition_index}.pt"
    try:
        subprocess.check_call(["hdfs", "dfs", "-put", "-f", tmp_path, hdfs_output_path])
        print(f"Partition {partition_index}: Successfully wrote model to {hdfs_output_path}")
    except Exception as e:
        print(f"Partition {partition_index}: Error writing model to HDFS: {e}")
    finally:
        os.remove(tmp_path)

    yield partition_index

def main():
    # Read tuning time (in hours) from command-line argument; default is 1 hour.
    tune_time_hours = 1.0
    if len(sys.argv) > 1:
        try:
            tune_time_hours = float(sys.argv[1])
        except ValueError:
            print("Invalid tuning time provided. Using default value 1.0 hour.")

    # Create a SparkSession.
    spark = SparkSession.builder.appName("TuneInceptionV3Model").getOrCreate()
    sc = spark.sparkContext

    # -------------------------------------------
    # Step 1: Read the CSV file with image labels.
    # -------------------------------------------
    csv_path = "hdfs://management:9000/data/train.csv"
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # Build a mapping from the base file name to a numerical label.
    def convert_label(label):
        if isinstance(label, str) and label.strip().lower().startswith("h"):
            return 0
        else:
            return 1

    mapping = { os.path.basename(row['file_name']).strip().lower(): convert_label(row['label'])
                for row in df.collect() }
    print("Broadcast mapping:", mapping)
    mapping_bc = sc.broadcast(mapping)

    # -------------------------------------------
    # Step 2: Read images from HDFS.
    # -------------------------------------------
    images_rdd = sc.binaryFiles("hdfs://management:9000/data/train_data/*")

    # -------------------------------------------
    # Step 3: Process each partition to tune the model.
    # Pass the tuning time (in hours) to each partition.
    # -------------------------------------------
    results = images_rdd.mapPartitionsWithIndex(
        lambda idx, it: process_partition(idx, list(it), mapping_bc, tune_time_hours)
    ).collect()

    print("Completed processing partitions:", results)
    spark.stop()

if __name__ == "__main__":
    main()
