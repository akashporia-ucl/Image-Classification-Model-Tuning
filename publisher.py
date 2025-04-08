import pika
import time

time.sleep(5)

credentials = pika.PlainCredentials('myuser', 'mypassword')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='management', credentials=credentials))
channel = connection.channel()

# Declare a direct exchange
channel.exchange_declare(exchange='direct_logs', exchange_type='direct')

# Declare the queue
channel.queue_declare(queue='model_queue')

# Bind the queue to the exchange with a specific routing key
channel.queue_bind(exchange='direct_logs', queue='model_queue', routing_key='model_key')

message = "Model tuning completed"
# Publish the message to the direct exchange with a routing key
channel.basic_publish(exchange='direct_logs', 
                      routing_key='model_key', 
                      body=message)
print(" [x] Sent %r" % message)

connection.close()
