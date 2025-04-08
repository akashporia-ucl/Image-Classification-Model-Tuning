import os
import io
import sys
import tempfile
import subprocess
import torch
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

def load_collated_model_state(hdfs_model_path="/data/model_collated/inceptionv3_final.pt"):
    """
    Downloads the collated InceptionV3 model from HDFS to a temporary file,
    loads its state dictionary, and then removes the temporary file.
    """
    tmp_path = tempfile.mktemp()
    try:
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
      - Loads and preprocesses the image (InceptionV3 requires 299x299 input).
      - Runs inference using the InceptionV3 model loaded with the broadcast state dict.
      - Compares the predicted label with the true label from the broadcast mapping.
      - Returns a tuple: (file_name, true_label, predicted_label, correct_flag)
    """
    import torch
    from torchvision import models
    import torchvision.transforms as transforms
    from PIL import Image
    import os
    import io

    # Build InceptionV3 with aux_logits disabled.
    model = models.inception_v3(pretrained=False, aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    state_dict = model_state_dict_bc.value
    model.load_state_dict(state_dict)
    model.eval()

    # InceptionV3 transformation: Resize to 342, then center crop to 299.
    transform = transforms.Compose([
        transforms.Resize(342),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mapping = mapping_bc.value
    results = []
    for file_path, file_content in iterator:
        try:
            base_name = os.path.basename(file_path).strip().lower()
            if base_name not in mapping:
                continue
            true_label = mapping[base_name]
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0)
            with torch.no_grad():
                output = model(image)
                pred_label = torch.argmax(output, dim=1).item()
            correct = 1 if pred_label == true_label else 0
            results.append((base_name, true_label, pred_label, correct))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return iter(results)

def main():
    spark = SparkSession.builder.appName("EvaluateInceptionV3Model").getOrCreate()
    sc = spark.sparkContext

    # Step 1: Read the test CSV from HDFS.
    test_csv_path = "hdfs://management:9000/data/test.csv"
    test_df = spark.read.csv(test_csv_path, header=True, inferSchema=True)

    # Build mapping: base file name -> true label.
    def convert_label(label):
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

    # Step 2: Load the collated InceptionV3 model state dictionary.
    model_state_dict = load_collated_model_state("/data/model_collated/inceptionv3_final.pt")
    if model_state_dict is None:
        print("Failed to load the collated InceptionV3 model. Exiting.")
        sys.exit(1)
    model_state_dict_bc = sc.broadcast(model_state_dict)

    # Step 3: Read test images from HDFS.
    test_images_rdd = sc.binaryFiles("hdfs://management:9000/data/test_data_v2/*")

    # Step 4: Process each partition to run inference.
    results_rdd = test_images_rdd.mapPartitions(lambda it: process_partition(it, mapping_bc, model_state_dict_bc))
    
    # Convert results to DataFrame.
    results_rows = results_rdd.map(lambda x: Row(file_name=x[0], true_label=x[1], pred_label=x[2], correct=x[3]))
    results_df = spark.createDataFrame(results_rows)
    results_df.show()

    # Compute overall accuracy.
    agg_df = results_df.groupBy().sum("correct").withColumnRenamed("sum(correct)", "total_correct")
    count_df = results_df.groupBy().count().withColumnRenamed("count", "total")
    overall_df = agg_df.crossJoin(count_df).withColumn("accuracy", col("total_correct")/col("total"))
    overall_df.show()

    # Write detailed results to CSV in HDFS.
    output_path = "/data/model_collated/test_results_inception"
    results_df.write.csv(output_path, header=True, mode="overwrite")
    print(f"Test results saved to HDFS at {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()
