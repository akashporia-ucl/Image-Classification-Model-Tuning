import pika
import sys
import time

# Wait for 5 seconds (if needed to ensure services are up)
time.sleep(5)

# Retrieve the message from the command line arguments.
if len(sys.argv) < 2:
    print("Usage: python send_message.py <message>")
    sys.exit(1)

# Join all provided arguments into a single message.
message = " ".join(sys.argv[1:])

# Set up the credentials and connection parameters.
credentials = pika.PlainCredentials('myuser', 'mypassword')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='management', credentials=credentials))
channel = connection.channel()

# Declare a direct exchange.
channel.exchange_declare(exchange='direct_logs', exchange_type='direct')

# Declare the queue with durability.
channel.queue_declare(queue='model_queue', durable=True)

# Bind the queue to the exchange with a specific routing key.
channel.queue_bind(exchange='direct_logs', queue='model_queue', routing_key='model_key')

# Publish the message to the direct exchange using the specified routing key.
channel.basic_publish(exchange='direct_logs', 
                      routing_key='model_key', 
                      body=message)
print(" [x] Sent %r" % message)

connection.close()

#python send_message.py "Model tuning completed"
