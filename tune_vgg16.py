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
      - Loads a pretrained VGG16 model and adapts its classifier for binary classification.
      - Repeatedly iterates over its partition data until the specified tuning time has elapsed.
            * For each image in its partition:
                - Extracts the image’s base name.
                - Looks up its label from the broadcast mapping.
                - Loads and preprocesses the image.
                - Performs one training step (forward, loss computation, backward, and optimizer update).
      - After tuning, the tuned model parameters are saved to a temporary file
        and then written to HDFS using the HDFS CLI command (under /data/model_partitions/vgg16).
    """
    # Import necessary libraries on the worker.
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image

    # Load the pretrained VGG16 model.
    model = models.vgg16(pretrained=True)
    # Replace the final classifier layer.
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 2)
    model.train()  # Set to training mode.

    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Define image transformations (using ImageNet normalization).
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Retrieve the broadcasted mapping (base filename -> numerical label).
    mapping = mapping_bc.value

    # Convert tuning time from hours to seconds.
    tuning_time_sec = tune_time_hours * 3600
    start_time = time.time()

    # Since the iterator can be exhausted, store the partition data in a list.
    data_list = list(partition_data)

    # Loop until the tuning time has elapsed.
    while time.time() - start_time < tuning_time_sec:
        for file_path, file_content in data_list:
            # If we've exceeded the tuning time, break out.
            if time.time() - start_time >= tuning_time_sec:
                break
            try:
                # Extract the normalized base file name.
                base_name = os.path.basename(file_path).strip().lower()
                if base_name not in mapping:
                    print(f"Partition {partition_index}: Label not found for {base_name}, skipping.")
                    continue

                # Get the label and create a tensor.
                label_val = mapping[base_name]
                label_tensor = torch.tensor([label_val])

                # Open the image, convert to RGB, apply transformations, and add a batch dimension.
                image = Image.open(io.BytesIO(file_content)).convert("RGB")
                image = transform(image)
                image = image.unsqueeze(0)

                # Forward pass, compute loss, and update the model.
                output = model(image)
                loss = criterion(output, label_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Partition {partition_index}: Processed {base_name} with loss {loss.item()}")
            except Exception as e:
                print(f"Partition {partition_index}: Error processing {file_path}: {e}")

    # After tuning, save the tuned model parameters to an in-memory buffer.
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
        subprocess.check_call(["hdfs", "dfs", "-mkdir", "-p", "/data/model_partitions/vgg16"])
    except Exception as e:
        print(f"Partition {partition_index}: Error ensuring HDFS directory exists: {e}")

    # Define the target HDFS path.
    hdfs_output_path = f"/data/model_partitions/vgg16/vgg16_model_partition_{partition_index}.pt"
    try:
        # Upload the temporary file to HDFS using the CLI.
        subprocess.check_call(["hdfs", "dfs", "-put", "-f", tmp_path, hdfs_output_path])
        print(f"Partition {partition_index}: Successfully wrote model to {hdfs_output_path}")
    except Exception as e:
        print(f"Partition {partition_index}: Error writing model to HDFS: {e}")
    finally:
        os.remove(tmp_path)

    yield partition_index

def main():
    # Read tuning time from command-line argument (in hours) with default value of 1.
    tune_time_hours = 1.0
    if len(sys.argv) > 1:
        try:
            tune_time_hours = float(sys.argv[1])
        except ValueError:
            print("Invalid tuning time provided. Using default value 1.0 hour.")
    
    # Create a SparkSession.
    spark = SparkSession.builder.appName("TuneVGG16Model").getOrCreate()
    sc = spark.sparkContext

    # -------------------------------------------
    # Step 1: Read the CSV file with image labels.
    # -------------------------------------------
    csv_path = "hdfs://management:9000/data/train.csv"
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    
    # Build a mapping from the base filename to a numerical label.
    def convert_label(label):
        if isinstance(label, str) and label.strip().lower().startswith("h"):
            return 0
        else:
            return 1

    mapping = { os.path.basename(row['file_name']).strip().lower(): convert_label(row['label'])
                for row in df.collect() }
    print("Broadcast mapping:", mapping)
    
    # Broadcast the mapping.
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
