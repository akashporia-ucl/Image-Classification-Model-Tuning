import os
import subprocess
import json
import logging
import time
from PIL import Image
import pika
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import io
import torch.nn as nn

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RabbitMQ configuration
RABBITMQ_HOST = 'worker1'
RABBITMQ_USERNAME = 'myuser'
RABBITMQ_PASSWORD = 'mypassword'
credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)

# Exchange and queue settings
EXCHANGE_NAME = 'direct_logs'
REQUEST_QUEUE = 'request_queue'
REQUEST_ROUTING_KEY = 'request_key'
RESPONSE_QUEUE = 'response_queue'
RESPONSE_ROUTING_KEY = 'response_key'

# HDFS configuration for the model and images
MODEL_HDFS_PATH = '/data/model_collated'
MODEL_HDFS_NAME = 'resnet50_final.pt'
# (No hardcoding of local filename; it is chosen dynamically from HDFS)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def load_pretrained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a ResNet50 model and adjust for binary classification.
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    model.eval()

    # Load the model state from HDFS.
    try:
        full_model_path = os.path.join(MODEL_HDFS_PATH, MODEL_HDFS_NAME)
        subprocess.check_call(["hdfs", "dfs", "-get", "-f", full_model_path])
        state_dict = torch.load(MODEL_HDFS_NAME, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully from HDFS.")
    except Exception as e:
        logger.error("Error loading model from HDFS: %s", e)
        return None
    
    return model

def download_image_from_hdfs(hdfs_path, local_path):
    logger.info("Downloading image from HDFS: '%s' to local file: '%s'", hdfs_path, local_path)
    subprocess.run(['hdfs', 'dfs', '-get', hdfs_path, local_path], check=True)

def preprocess_image(image_path):
    """
    Preprocess the input image in the same way as during training.
    For InceptionV3, use a 299x299 crop; for other architectures (e.g., ResNet50), use 224x224.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    return image

def publish_response(classification):
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
        )
        channel = connection.channel()
        # Declare the exchange without durable flag (or set durable=False to match existing exchange)
        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct')
        channel.queue_declare(queue=RESPONSE_QUEUE, durable=True)
        channel.queue_bind(exchange=EXCHANGE_NAME, queue=RESPONSE_QUEUE, routing_key=RESPONSE_ROUTING_KEY)
        channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=RESPONSE_ROUTING_KEY,
            body=str(classification),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        logger.info("Published classification result: '%s'", classification)
        time.sleep(0.2)
        connection.close()
    except Exception as e:
        logger.error("Error publishing response: %s", e)

def process_message(body, model):
    logger.info("Processing message: %s", body)
    try:
        message_data = json.loads(body.decode('utf-8'))
    except Exception as e:
        logger.error("Failed to parse message: %s", e)
        return

    filename = message_data.get('filename')
    hdfs_image_path = message_data.get('hdfs_path')
    if not filename or not hdfs_image_path:
        logger.error("Invalid message data: %s", message_data)
        return

    local_image_path = filename
    try:
        download_image_from_hdfs(hdfs_image_path, local_image_path)
        # Use the updated preprocessing for consistency with training
        input_tensor = preprocess_image(local_image_path)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
        pred_label =  int(predicted.cpu().numpy()[0])
        publish_response(pred_label)
    except Exception as e:
        logger.error("Error during image processing/classification: %s", e)
    finally:
        if os.path.exists(local_image_path):
            try:
                os.remove(local_image_path)
            except Exception as e:
                logger.error("Error cleaning up image file: %s", e)

def on_message(ch, method, properties, body, model):
    logger.info("Received message: %s", body)
    process_message(body, model)
    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    model = load_pretrained_model()
    if model is None:
        logger.error("Failed to load the model. Exiting.")
        return
    
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
    )
    channel = connection.channel()
    # Declare the exchange without a durable flag.
    channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct')
    channel.queue_declare(queue=REQUEST_QUEUE, durable=True)
    channel.queue_bind(exchange=EXCHANGE_NAME, queue=REQUEST_QUEUE, routing_key=REQUEST_ROUTING_KEY)

    logger.info("Waiting for messages in queue '%s'. To exit press CTRL+C", REQUEST_QUEUE)

    def callback(ch, method, properties, body):
        on_message(ch, method, properties, body, model)

    channel.basic_consume(queue=REQUEST_QUEUE, on_message_callback=callback, auto_ack=False)
    channel.start_consuming()

if __name__ == '__main__':
    main()
