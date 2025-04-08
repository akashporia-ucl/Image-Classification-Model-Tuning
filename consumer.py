import pika

def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)

# Use new credentials instead of relying on the default guest account
credentials = pika.PlainCredentials('myuser', 'mypassword')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='management', credentials=credentials))
channel = connection.channel()

# Declare a direct exchange
channel.exchange_declare(exchange='direct_logs', exchange_type='direct')

# Declare the queue
channel.queue_declare(queue='request_queue')

# Bind the queue to the direct exchange with a specific routing key
channel.queue_bind(exchange='direct_logs', queue='request_queue', routing_key='request_key')

print(' [*] Waiting for messages. To exit press CTRL+C')

# Start consuming messages from the queue
channel.basic_consume(queue='request_queue', on_message_callback=callback, auto_ack=True)

channel.start_consuming()
