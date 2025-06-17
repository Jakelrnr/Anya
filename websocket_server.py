import asyncio 
import websockets
import json
import logging
import traceback
from typing import Set
import time
import os
#92

from ai_core.ai_brain import AIPersonality
from model.model_manager import ModelManager
from ai_core.speech_to_text import transcribe_audio
from ai_core.text_to_speech import text_to_speech

# Setup detailed logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSocketServer:
    def __init__(self, ai_personality: AIPersonality, host="localhost", port=8765):
        self.ai = ai_personality
        self.host = host
        self.port = port
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()

    async def register_client(self, websocket):
        self.connected_clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

    async def unregister_client(self, websocket):
        self.connected_clients.discard(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")

    async def send_message(self, websocket, message_type: str, content: str, extra_data=None):
        try:
            response_data = {
                "type": message_type,
                "content": content,
                "timestamp": time.time()
            }
            if extra_data:
                response_data.update(extra_data)
            await websocket.send(json.dumps(response_data))
        except websockets.ConnectionClosed:
            logger.warning("Connection closed while sending message")
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"Error in send_message: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def broadcast_message(self, message_type: str, content: str):
        if self.connected_clients:
            tasks = [
                self.send_message(client, message_type, content)
                for client in self.connected_clients.copy()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_user_message(self, websocket, message_data):
        try:
            user_message = message_data.get("content", "").strip()
            if not user_message:
                await self.send_message(websocket, "error", "Message cannot be empty")
                return

            await self.send_message(websocket, "typing", "AI is typing...")

            ai_response = await self.ai.generate_response(user_message)
            audio_path = text_to_speech(ai_response)

            await self.send_message(
                websocket,
                "ai_response_audio",
                ai_response,
                extra_data={
                    "audio_path": audio_path,
                    "conversation_summary": self.ai.get_conversation_summary(),
                    "response_time": time.time()
                }
            )

        except Exception as e:
            logger.error(f"Error in handle_user_message: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            await self.send_message(websocket, "error", f"An error occurred: {str(e)}")

    async def handle_client_connection(self, websocket):
        try:
            await self.register_client(websocket)
            await self.send_message(
                websocket,
                "welcome",
                f"Connected to {self.ai.personality_config['name']}!",
                extra_data={"ai_personality": self.ai.personality_config["traits"]}
            )

            async for raw_message in websocket:
                try:
                    parsed_data = json.loads(raw_message)
                    msg_type = parsed_data.get("type", "user_message")

                    if msg_type == 'user_message':
                        await self.handle_user_message(websocket, parsed_data)
                    elif msg_type == 'ping':
                        await self.send_message(websocket, "pong", "alive")
                    elif msg_type == "get_conversation_summary":
                        summary = self.ai.get_conversation_summary()
                        await self.send_message(websocket, "conversation_summary", "", extra_data=summary)
                    elif msg_type == "audio_input":
                        await self.handle_audio_input(websocket, parsed_data)
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    await self.send_message(websocket, "error", "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    await self.send_message(websocket, "error", f"Internal error: {str(e)}")

        except websockets.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in handle_client_connection: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
        finally:
            await self.unregister_client(websocket)

    async def handle_audio_input(self, websocket, message_data):
        temp_filename = None
        try:
            audio_data = message_data.get("audio_data")
            audio_format = message_data.get("format", "wav")

            if not audio_data:
                await self.send_message(websocket, "error", "Audio data is empty")
                return

            import base64, uuid
            audio_bytes = base64.b64decode(audio_data)
            temp_filename = f"temp_audio_{uuid.uuid4().hex}.{audio_format}"

            with open(temp_filename, "wb") as f:
                f.write(audio_bytes)

            await self.send_message(websocket, "typing", "AI is processing audio input...")
            user_text = transcribe_audio(temp_filename)

            if not user_text.strip():
                await self.send_message(websocket, "error", "Could not transcribe audio")
                return

            ai_response = await self.ai.generate_response(user_text)
            audio_path = text_to_speech(ai_response)

            await self.send_message(
                websocket,
                "ai_response_audio",
                ai_response,
                extra_data={"audio_path": audio_path}
            )

        except Exception as e:
            logger.error(f"Error processing audio input: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            await self.send_message(websocket, "error", f"Audio processing error: {str(e)}")
        finally:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                    logger.debug(f"Temporary audio file {temp_filename} removed")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_filename}: {e}")

    async def start_server(self):
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        server = await websockets.serve(
            self.handle_client_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
            max_size=10000000
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        return server

async def run_ai_server():
    logger.info("Initializing AI Model...")
    try:
        manager = ModelManager()
        model_path = "models/Qwen2.5-14B-Instruct-IQ2_M.gguf"
        if not os.path.exists(model_path):
            model_path = manager.download_model()

        # ✅ FIX: only one object is returned
        model = manager.load_model(model_path)

        ai_personality = AIPersonality(model,tokenizer=manager.tokenizer)
        server = WebSocketServer(ai_personality)
        websocket_server = await server.start_server()

        try:
            await websocket_server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Shutting down WebSocket server...")
            websocket_server.close()
            await websocket_server.wait_closed()

    except Exception as e:
        logger.error(f"Error in run_ai_server: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(run_ai_server())
