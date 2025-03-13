import pika

connection_parameters = pika.ConnectionParameters('localhost', credentials=pika.PlainCredentials('user','password'))


connection = pika.BlockingConnection(connection_parameters)

channel = connection.channel()

channel.queue_declare(queue='govdocs_queue')

message = 'Hello, GovDocs!'

channel.basic_publish(exchange='', routing_key='govdocs_queue', body=message)

print(f" [x] Sent '{message}'")

connection.close()