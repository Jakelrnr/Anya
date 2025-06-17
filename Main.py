

import asyncio
import threading
from server.upload_server import create_upload_app
from server.websocket_server import run_ai_server

def run_flask():
    app = create_upload_app() #locally hosted server 4 anya
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    asyncio.run(run_ai_server())
