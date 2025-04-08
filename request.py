import os
import subprocess
import pickle
import json
from PIL import Image
import numpy as np
import pika
import logging

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RabbitMQ configuration
RABBITMQ_HOST = 'management'
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
MODEL_HDFS_PATH = '/data/best_model'
LOCAL_MODEL_FILENAME = 'best_model'
HDFS_IMAGE_BASE_PATH = '/data/images/'

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def load_pretrained_model():
    """
    Downloads the pretrained model from HDFS (if not present locally) and loads it.
    Assumes the model is stored as a pickle file.
    """
    if not os.path.exists(LOCAL_MODEL_FILENAME):
        logger.info("Pretrained model not found locally. Downloading from HDFS...")
        subprocess.run(['hdfs', 'dfs', '-get', MODEL_HDFS_PATH, LOCAL_MODEL_FILENAME], check=True)
    with open(LOCAL_MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    logger.info("Pretrained model loaded successfully.")
    return model

def download_image_from_hdfs(hdfs_path, local_path):
    """
    Downloads the image from HDFS to a local file.
    """
    logger.info("Downloading image from HDFS: '%s' to local file: '%s'", hdfs_path, local_path)
    subprocess.run(['hdfs', 'dfs', '-get', hdfs_path, local_path], check=True)

def preprocess_image(image_path):
    """
    Preprocesses the image for inference.
    
    This example:
      - Opens the image and converts it to RGB.
      - Resizes the image to 224x224 pixels.
      - Flattens the image into a 1D array and reshapes it to (1, -1).
    
    Adjust these steps as needed for your model.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    features = image_array.flatten().reshape(1, -1)
    return features

def publish_response(classification):
    """
    Publishes the classification result to the response queue using the
    direct exchange setup.
    """
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
        )
        channel = connection.channel()
        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct', durable=True)
        channel.queue_declare(queue=RESPONSE_QUEUE, durable=True)
        channel.queue_bind(exchange=EXCHANGE_NAME, queue=RESPONSE_QUEUE, routing_key=RESPONSE_ROUTING_KEY)
        channel.basic_publish(
            exchange=EXCHANGE_NAME, 
            routing_key=RESPONSE_ROUTING_KEY, 
            body=classification
        )
        logger.info("Published classification result: '%s'", classification)
        connection.close()
    except Exception as e:
        logger.error("Error publishing response: %s", e)

def process_message(body, model):
    """
    Parses the incoming message, downloads and preprocesses the image from HDFS,
    classifies the image using the pretrained model, and publishes the result.
    """
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

    local_image_path = filename  # Save the image locally using the filename provided.
    try:
        download_image_from_hdfs(hdfs_image_path, local_image_path)
        features = preprocess_image(local_image_path)
        prediction = model.predict(features)
        # Assume the model returns an array (e.g., ["ai"] or ["human"])
        classification = prediction[0]
        publish_response(classification)
    except Exception as e:
        logger.error("Error during image processing/classification: %s", e)
    finally:
        if os.path.exists(local_image_path):
            try:
                os.remove(local_image_path)
            except Exception as e:
                logger.error("Error cleaning up image file: %s", e)

def on_message(ch, method, properties, body, model):
    """
    Callback function that handles the consumed RabbitMQ message.
    """
    logger.info("Received message: %s", body)
    process_message(body, model)
    # Acknowledge the message if you are not using auto_ack
    ch.basic_ack(delivery_tag=method.delivery_tag)

# -----------------------------------------------------------------------------
# Main Function to Setup RabbitMQ Consumer
# -----------------------------------------------------------------------------
def main():
    model = load_pretrained_model()

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
    )
    channel = connection.channel()

    # Declare the exchange and bind the request queue with the routing key.
    channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct', durable=True)
    channel.queue_declare(queue=REQUEST_QUEUE, durable=True)
    channel.queue_bind(exchange=EXCHANGE_NAME, queue=REQUEST_QUEUE, routing_key=REQUEST_ROUTING_KEY)

    logger.info("Waiting for messages in queue '%s'. To exit press CTRL+C", REQUEST_QUEUE)

    # Define a callback that wraps the on_message function with model passed in.
    def callback(ch, method, properties, body):
        on_message(ch, method, properties, body, model)

    channel.basic_consume(queue=REQUEST_QUEUE, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

if __name__ == '__main__':
    main()
