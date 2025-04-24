import os
import io
import sys
import time
import random
import tempfile
import subprocess
from pyspark.sql import SparkSession
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import  argparse


class PartitionImageDataset(Dataset):
    def __init__(self, data_list, mapping, transform):
        self.data_list = data_list
        self.mapping = mapping
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path, file_content = self.data_list[idx]
        base_name = os.path.basename(file_path).strip().lower()
        label = self.mapping.get(base_name)
        image = Image.open(io.BytesIO(file_content)).convert("RGB")
        image = self.transform(image)
        return image, label


def process_partition(partition_index, partition_data, mapping_bc, tune_time_hours):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = mapping_bc.value

    # Collect and shuffle data
    data_list = list(partition_data)
    if not data_list:
        print(f"Partition {partition_index}: No data in this partition")
        return
    random.shuffle(data_list)

    # Split into train/validation (80/20)
    split_idx = int(0.8 * len(data_list))
    train_list = data_list[:split_idx]
    val_list   = data_list[split_idx:]

    # Data augmentation and transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_dataset = PartitionImageDataset(train_list, mapping, train_transform)
    val_dataset   = PartitionImageDataset(val_list, mapping, val_transform)

    # Compute weights for imbalanced sampling
    labels = [mapping.get(os.path.basename(fp).strip().lower()) for fp, _ in train_list]
    label_counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    sample_weights = [1.0 / label_counts[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Model setup (fine-tune layer3, layer4, fc)
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)

    # Loss, optimizer with parameter groups, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 5e-4},
        {'params': model.fc.parameters(),    'lr': 1e-3},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # Training loop (time‑based)
    start_time = time.time()
    tuning_time_sec = tune_time_hours * 3600
    epoch = 0

    while time.time() - start_time < tuning_time_sec:
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            if time.time() - start_time >= tuning_time_sec:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc  = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss / val_total
        val_acc  = 100 * val_correct / val_total

        print(f"Partition {partition_index}: Epoch {epoch+1} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()
        epoch += 1

    # Save the trained model state
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model_bytes = buffer.getvalue()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(model_bytes)
        tmp_path = tmp_file.name

    try:
        subprocess.check_call(["hdfs", "dfs", "-mkdir", "-p", "/data/model_partitions/resnet50"] )
    except Exception as e:
        print(f"Partition {partition_index}: Error ensuring HDFS dir: {e}")
    hdfs_output = f"/data/model_partitions/resnet50/resnet50_model_partition_{partition_index}.pt"
    try:
        subprocess.check_call(["hdfs", "dfs", "-put", "-f", tmp_path, hdfs_output])
        print(f"Partition {partition_index}: Model saved to {hdfs_output}")
    except Exception as e:
        print(f"Partition {partition_index}: Error writing model: {e}")
    finally:
        os.remove(tmp_path)

    yield partition_index


def main():
    # 1. Set up argument parser
    parser = argparse.ArgumentParser(
        description="Tune ResNet50 model on HDFS image data"
    )
    parser.add_argument(
        "--tune_time",
        type=float,
        default=1.0,
        help="Tuning time per partition in hours (default: 1.0)"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="hdfs://management:9000/data/train.csv",
        help="Path to the labels CSV file"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="hdfs://management:9000/data/train_data/*",
        help="Path to the image data files"
    )
    args = parser.parse_args()
    tune_time_hours = args.tune_time
    csv_path =  args.csv_path
    image_path = args.image_path

    # Parse tuning time
    # tune_time_hours = 1.0
    # if len(sys.argv) > 1:
    #     try:
    #         tune_time_hours = float(sys.argv[1])
    #     except ValueError:
    #         print("Invalid tuning time; defaulting to 1.0 hour")

    # Initialize Spark
    spark = SparkSession.builder.appName("TuneResNet50Model").getOrCreate()
    sc    = spark.sparkContext

    # Read labels CSV and broadcast mapping
    #csv_path = "hdfs://management:9000/data/train.csv"
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    mapping = {os.path.basename(r['file_name']).strip().lower(): int(r['label']) for r in df.collect()}
    mapping_bc = sc.broadcast(mapping)

    # Read image data and tune per partition
    #image_path = "hdfs://management:9000/data/train_data/*"
    images_rdd = sc.binaryFiles(image_path)
    results = (images_rdd
               .mapPartitionsWithIndex(
                   lambda idx, it: process_partition(idx, it, mapping_bc, tune_time_hours)
               )
               .collect())

    print("Completed processing partitions:", results)
    spark.stop()


if __name__ == "__main__":
    main()