import asyncio
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from model.model_manager import ModelManager
#0309 #test
@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float

class AIPersonality:
    """Manages AI personality and conversation state"""

    def __init__(self, model, tokenizer=None, personality_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history: List[Message] = []
        self.personality_config = personality_config or self.default_personality()

        # Memory management
        self.max_history_length = 20  # Keep last 20 messages
        self.context_window = 4000    # Model's context limit

    def default_personality(self):
        """Default personality configuration"""
        return {
            "name": "Anya",
            "traits": [
                "friendly", "helpful", "curious", "energetic"
            ],
            "speaking_style": "casual but informative",
            "interests": ["technology", "learning", "helping others"],
            "system_prompt": """You are Anya, a friendly and energetic AI companion. 
                              You love helping people and learning new things. 
                              Keep responses natural and engaging.""",
            "generation_params": {
                "temperature": 0.8,
                "top_p": 0.9,
                "max_tokens": 200,
                "repeat_penalty": 1.1
            }
        }

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        message = Message(
            role=role,
            content=content.strip(),
            timestamp=time.time()
        )

        self.conversation_history.append(message)

        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            system_messages = [msg for msg in self.conversation_history if msg.role == "system"]
            recent_messages = self.conversation_history[-self.max_history_length:]
            self.conversation_history = system_messages + recent_messages

    def format_conversation_for_model(self) -> List[Dict]:
        """Convert conversation history to model format"""
        messages = []

        if self.personality_config["system_prompt"]:
            messages.append({
                "role": "system", 
                "content": self.personality_config["system_prompt"]
            })

        for msg in self.conversation_history[-10:]:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        return messages

    async def generate_response(self, user_message: str) -> str:
        """Generate AI response to user message"""

        self.add_message("user", user_message)
        messages = self.format_conversation_for_model()

        try:
            loop = asyncio.get_event_loop()

            def _generate():
                return self.model.create_chat_completion(
                    messages=messages,
                    **self.personality_config["generation_params"]
                )

            response = await loop.run_in_executor(None, _generate)
            ai_response = response['choices'][0]['message']['content'].strip()

            self.add_message("assistant", ai_response)

            return ai_response

        except Exception as e:
            error_response = f"Sorry, I encountered an error: {str(e)}"
            self.add_message("assistant", error_response)
            return error_response

    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len([m for m in self.conversation_history if m.role == "user"]),
            "ai_messages": len([m for m in self.conversation_history if m.role == "assistant"]),
            "conversation_start": self.conversation_history[0].timestamp if self.conversation_history else None,
            "last_activity": self.conversation_history[-1].timestamp if self.conversation_history else None
        }

# Usage example
async def test_ai_system():
    manager = ModelManager()
    model = manager.load_model("path/to/your/model.gguf")
    tokenizer = manager.tokenizer

    ai = AIPersonality(model, tokenizer=tokenizer)

    response1 = await ai.generate_response("Hi! What's your name?")
    print(f"AI: {response1}")

    response2 = await ai.generate_response("What do you like to do?")
    print(f"AI: {response2}")

    summary = ai.get_conversation_summary()
    print(f"Conversation summary: {summary}")

# asyncio.run(test_ai_system())
