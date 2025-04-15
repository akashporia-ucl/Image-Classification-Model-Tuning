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
    Each executor processes its partition as follows:
      - Loads a pretrained ResNet50 model.
      - Adapts its final layer for binary classification and freezes all parameters
        except for the last residual block (layer4) and final fully connected layer.
      - Uses an AdamW optimiser with a StepLR scheduler.
      - The training is organised into epochs. In each epoch the entire partition’s
        data is used to perform training steps; average loss and training accuracy are
        printed at the end of each epoch.
      - Training is repeated until the specified tuning time (in hours) has elapsed.
      - After training, the model’s state dictionary is saved to a temporary file and
        then uploaded to HDFS under /data/model_partitions/resnet50.
    """
    # Import necessary libraries on the worker.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image

    # Determine the computing device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet50 and modify the final layer for binary classification.
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Freeze all layers first.
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last residual block and the new fully connected layer.
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)
    model.train()

    # Define loss function, AdamW optimiser and learning rate scheduler.
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0005,
        weight_decay=1e-4
    )
    scheduler = StepLR(optimiser, step_size=5, gamma=0.1)

    # Define the transformation for training (resize to 224x224 and normalise).
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Retrieve the mapping of image file base names to labels.
    mapping = mapping_bc.value

    # Convert tuning time from hours to seconds.
    tuning_time_sec = tune_time_hours * 3600
    start_time = time.time()
    epoch = 0

    # Since the iterator may be exhausted, collect partition data into a list.
    data_list = list(partition_data)

    # Training loop: iterate over epochs until the allotted tuning time has elapsed.
    while time.time() - start_time < tuning_time_sec:
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for file_path, file_content in data_list:
            # Check if tuning time has elapsed.
            if time.time() - start_time >= tuning_time_sec:
                break
            try:
                base_name = os.path.basename(file_path).strip().lower()
                if base_name not in mapping:
                    print(f"Partition {partition_index}: Label not found for {base_name}, skipping.")
                    continue

                label_val = mapping[base_name]
                # Move label tensor to the correct device.
                label_tensor = torch.tensor([label_val]).to(device)
                
                # Load and preprocess the image.
                image = Image.open(io.BytesIO(file_content)).convert("RGB")
                image = transform(image)
                image = image.unsqueeze(0).to(device)
                
                # Forward pass.
                outputs = model(image)
                loss = criterion(outputs, label_tensor)
                
                # Backward pass and optimisation.
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                
                running_loss += loss.item()

                # Calculate training accuracy for this sample.
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == label_tensor).sum().item()
                total_train += 1
                
                print(f"Partition {partition_index}: Processed {base_name} with loss {loss.item():.4f}")
            except Exception as e:
                print(f"Partition {partition_index}: Error processing {file_path}: {e}")

        if total_train > 0:
            avg_loss = running_loss / total_train
            training_accuracy = 100 * correct_train / total_train
            print(f"Partition {partition_index}: Epoch {epoch+1} complete: Avg Loss: {avg_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%")
            print(f"Partition {partition_index}: Current Learning Rate: {scheduler.get_last_lr()[0]}")
        else:
            print(f"Partition {partition_index}: No training data processed in epoch {epoch+1}")

        # Step the learning rate scheduler after the epoch.
        scheduler.step()
        epoch += 1

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
        subprocess.check_call(["hdfs", "dfs", "-mkdir", "-p", "/data/model_partitions/resnet50"])
    except Exception as e:
        print(f"Partition {partition_index}: Error ensuring HDFS directory exists: {e}")

    # Define the target HDFS path.
    hdfs_output_path = f"/data/model_partitions/resnet50/resnet50_model_partition_{partition_index}.pt"
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
    spark = SparkSession.builder.appName("TuneResNet50Model").getOrCreate()
    sc = spark.sparkContext

    # -------------------------------------------
    # Step 1: Read the CSV file with image labels.
    # -------------------------------------------
    csv_path = "hdfs://management:9000/data/train.csv"
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # Build a mapping from the base file name to the label (using labels from CSV directly).
    mapping = {
        os.path.basename(row['file_name']).strip().lower(): int(row['label'])
        for row in df.collect()
    }
    print("Broadcast mapping:", mapping)
    mapping_bc = sc.broadcast(mapping)

    # -------------------------------------------
    # Step 2: Read images from HDFS.
    # -------------------------------------------
    images_rdd = sc.binaryFiles("hdfs://management:9000/data/train_data/*")

    # -------------------------------------------
    # Step 3: Process each partition to tune the model.
    # The tuning time (in hours) is passed to each partition.
    # -------------------------------------------
    results = images_rdd.mapPartitionsWithIndex(
        lambda idx, it: process_partition(idx, list(it), mapping_bc, tune_time_hours)
    ).collect()

    print("Completed processing partitions:", results)
    spark.stop()

if __name__ == "__main__":
    main()