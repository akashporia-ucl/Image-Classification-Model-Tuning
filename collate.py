import os
import subprocess
import torch

def get_model_files(hdfs_dir):
    """
    Uses the HDFS CLI to list all .pt files in the specified HDFS directory.
    Returns a list of HDFS file paths.
    """
    try:
        output = subprocess.check_output(["hdfs", "dfs", "-ls", hdfs_dir]).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print("Error listing HDFS directory:", e)
        return []
    
    model_files = []
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 8:  # typical ls output: permissions, replication, owner, group, size, date, time, path
            file_path = parts[-1]
            if file_path.endswith(".pt"):
                model_files.append(file_path)
    return model_files

def download_models(model_files, local_dir):
    """
    Downloads the HDFS model files to a local directory.
    Returns a list of local file paths.
    """
    os.makedirs(local_dir, exist_ok=True)
    local_files = []
    for hdfs_file in model_files:
        local_path = os.path.join(local_dir, os.path.basename(hdfs_file))
        try:
            subprocess.check_call(["hdfs", "dfs", "-get", hdfs_file, local_path])
            local_files.append(local_path)
            print(f"Downloaded {hdfs_file} to {local_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {hdfs_file}: {e}")
    return local_files

def average_state_dicts(state_dicts):
    """
    Given a list of state dictionaries (from torch.load),
    returns a new state dictionary where each tensor is the element‐wise mean of the corresponding tensors.
    """
    if not state_dicts:
        return None

    keys = state_dicts[0].keys()
    avg_state = {}
    num_models = len(state_dicts)
    for key in keys:
        sum_tensor = None
        for sd in state_dicts:
            tensor = sd[key]
            if sum_tensor is None:
                sum_tensor = tensor.clone()
            else:
                sum_tensor.add_(tensor)
        avg_state[key] = sum_tensor / num_models
    return avg_state

def collate_models(model_folder, output_filename, local_download_dir="downloaded_models"):
    """
    For a given model folder in HDFS, downloads all .pt files,
    averages their state dictionaries, and saves the final model to HDFS under /data/model_collated.
    """
    print(f"Processing folder: {model_folder}")
    model_files = get_model_files(model_folder)
    if not model_files:
        print("No model files found in", model_folder)
        return False
    print("Found model files:", model_files)
    
    # Create a subfolder for this model type locally.
    model_local_dir = os.path.join(local_download_dir, os.path.basename(model_folder).lower())
    local_model_files = download_models(model_files, model_local_dir)
    
    state_dicts = []
    for local_file in local_model_files:
        try:
            sd = torch.load(local_file, map_location="cpu")
            state_dicts.append(sd)
            print(f"Loaded {local_file}")
        except Exception as e:
            print(f"Error loading {local_file}: {e}")
    
    if not state_dicts:
        print("No valid state dictionaries loaded for", model_folder)
        return False
    
    avg_state_dict = average_state_dicts(state_dicts)
    if avg_state_dict is None:
        print("Failed to average models for", model_folder)
        return False
    
    # Save the averaged model locally.
    final_model_local = output_filename
    torch.save(avg_state_dict, final_model_local)
    print(f"Averaged model saved locally as {final_model_local}")
    
    # Ensure the target HDFS output directory exists.
    try:
        subprocess.check_call(["hdfs", "dfs", "-mkdir", "-p", "/data/model_collated"])
    except Exception as e:
        print("Error ensuring HDFS output directory exists:", e)
    
    # Define the target HDFS path.
    hdfs_final_path = f"/data/model_collated/{os.path.basename(final_model_local)}"
    try:
        subprocess.check_call(["hdfs", "dfs", "-put", "-f", final_model_local, hdfs_final_path])
        print(f"Averaged model uploaded to HDFS at {hdfs_final_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error uploading final model to HDFS: {e}")
        return False
    return True

def main():
    # Define the HDFS model partition folders to process.
    model_folders = {
        "inceptionv3": "/data/model_partitions/inceptionv3",
        "resnet50": "/data/model_partitions/resnet50",
        "vgg16": "/data/model_partitions/vgg16"
    }
    
    # Process each folder and collate the models.
    for model_name, folder in model_folders.items():
        output_filename = f"{model_name}_final.pt"
        success = collate_models(folder, output_filename)
        if not success:
            print(f"Failed to collate models for {model_name}")
    
if __name__ == "__main__":
    main()
