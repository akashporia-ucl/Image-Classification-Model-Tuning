import os
import io
import sys
import tempfile
import subprocess
import torch
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import lower, trim, col
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

def load_collated_model_state(hdfs_model_path="/data/model_collated/resnet50_final.pt"):
    """
    Downloads the collated model file from HDFS to a temporary file, loads its state dictionary,
    and then removes the temporary file.
    """
    # Create a temporary file path.
    tmp_path = tempfile.mktemp()
    try:
        # Download the model file from HDFS.
        subprocess.check_call(["hdfs", "dfs", "-get", hdfs_model_path, tmp_path])
        state_dict = torch.load(tmp_path, map_location="cpu")
    except Exception as e:
        print("Error loading collated model from HDFS:", e)
        state_dict = None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return state_dict

def process_partition(iterator, mapping_bc, model_state_dict_bc):
    """
    For each image in the partition:
      - Load and preprocess the image.
      - Run inference using the ResNet50 model (built and loaded with the broadcast state dict).
      - Look up the true label from the broadcast test mapping.
      - Return a tuple: (file_name, true_label, predicted_label, correct_flag)
    """
    # Import torch and torchvision libraries within the executor.
    import torch
    import torch.nn as nn
    from torchvision import models
    import torchvision.transforms as transforms
    from PIL import Image
    import os
    import io

    # Build the ResNet50 model.
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    # Load the broadcasted state dictionary.
    state_dict = model_state_dict_bc.value
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode.

    # Define the transform (as used for ResNet50).
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Get the broadcast mapping of true labels.
    mapping = mapping_bc.value

    results = []
    for file_path, file_content in iterator:
        try:
            # Extract the base file name.
            base_name = os.path.basename(file_path).strip().lower()
            # Get the true label from the mapping; if not found, skip.
            if base_name not in mapping:
                continue
            true_label = mapping[base_name]
            # Load the image.
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0)
            # Run inference.
            with torch.no_grad():
                output = model(image)
                pred_label = torch.argmax(output, dim=1).item()
            correct = 1 if pred_label == true_label else 0
            results.append((base_name, true_label, pred_label, correct))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return iter(results)

def main():
    spark = SparkSession.builder.appName("EvaluateResNet50Model").getOrCreate()
    sc = spark.sparkContext

    # -------------------------------------------
    # Step 1: Read the test CSV file.
    # -------------------------------------------
    test_csv_path = "hdfs://management:9000/data/test.csv"
    test_df = spark.read.csv(test_csv_path, header=True, inferSchema=True)

    # Normalize file names and build a mapping: base file name -> true label.
    def convert_label(label):
        # Example conversion: labels starting with "h" become 0, otherwise 1.
        if isinstance(label, str) and label.strip().lower().startswith("h"):
            return 0
        else:
            return 1

    mapping = {
        os.path.basename(row['file_name']).strip().lower(): convert_label(row['label'])
        for row in test_df.collect()
    }
    print("Test mapping:", mapping)
    mapping_bc = sc.broadcast(mapping)

    # -------------------------------------------
    # Step 2: Load the collated model state dictionary from HDFS.
    # -------------------------------------------
    model_state_dict = load_collated_model_state("/data/model_collated/resnet50_final.pt")
    if model_state_dict is None:
        print("Failed to load the collated model. Exiting.")
        sys.exit(1)
    model_state_dict_bc = sc.broadcast(model_state_dict)

    # -------------------------------------------
    # Step 3: Read test images from HDFS.
    # -------------------------------------------
    test_images_rdd = sc.binaryFiles("hdfs://management:9000/data/test_data_v2/*")

    # -------------------------------------------
    # Step 4: Process each partition to run inference.
    # -------------------------------------------
    results_rdd = test_images_rdd.mapPartitions(
        lambda it: process_partition(it, mapping_bc, model_state_dict_bc)
    )
    
    # Convert the RDD to a DataFrame.
    results_rows = results_rdd.map(lambda x: Row(file_name=x[0],
                                                  true_label=x[1],
                                                  pred_label=x[2],
                                                  correct=x[3]))
    results_df = spark.createDataFrame(results_rows)
    results_df.show()

    # Optionally, compute overall accuracy.
    agg_df = results_df.groupBy().sum("correct").withColumnRenamed("sum(correct)", "total_correct")
    count_df = results_df.groupBy().count().withColumnRenamed("count", "total")
    overall_df = agg_df.crossJoin(count_df).withColumn("accuracy", col("total_correct")/col("total"))
    overall_df.show()

    # -------------------------------------------
    # Step 5: Write the detailed results to a CSV file in HDFS.
    # -------------------------------------------
    output_path = "/data/model_collated/test_results"
    results_df.write.csv(output_path, header=True, mode="overwrite")
    print(f"Test results saved to HDFS at {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()

# spark-submit \
#   --master spark://<spark-master-host>:7077 \
#   --deploy-mode cluster \
#   --conf spark.executor.instances=4 \
#   --conf spark.executor.cores=2 \
#   --conf spark.executor.memory=4G \
#   evaluate_model.py
