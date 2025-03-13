import pika

def on_message_received(ch, method, properties, body):
    print(f" [x] Received {body}")

connection_parameters = pika.ConnectionParameters('localhost', credentials=pika.PlainCredentials('user','password'))

connection = pika.BlockingConnection(connection_parameters)

channel = connection.channel()

channel.queue_declare(queue='govdocs_queue')

channel.basic_consume(queue='govdocs_queue', on_message_callback=on_message_received, auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')

channel.start_consuming()
