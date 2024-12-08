from balatro_connection import BalatroConnection
import time

copies = 1
for x in range(12349, 12349 + copies):
    conn = BalatroConnection(bot_port=x)
    conn.start_balatro_instance()
    print(f"Started Balatro instance on port {x}")

print("All instances started")
# while True:
#     pass
