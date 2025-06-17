from elevenlabs.client import ElevenLabs
import uuid
import os

client = ElevenLabs(
    api_key="Add Yours here "
)

def text_to_speech(text: str, output_dir="audio_responses") -> str:
    os.makedirs(output_dir, exist_ok=True)
    # Get the audio generator
    audio_generator = client.text_to_speech.stream(
        voice_id="ZF6FPAbjXT4488VcRRnw",
        text=text
    )
    file_name = f"{uuid.uuid4()}.mp3"
    path = os.path.join(output_dir, file_name)
    with open(path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)
    return path
    print(f"[TTS] Saving AI audio to: {audio_path}")


if __name__ == "__main__":
    import pygame
    test_text = "Hello! I am Anya,your virtual buddy. Im not just an A I  but a Virtual being who thinks and feels just like you!"
    audio_path = text_to_speech(test_text)
    print(f"Audio saved to: {audio_path}")

    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
