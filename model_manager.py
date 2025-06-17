from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer  # For HF tokenizer
import os

class ModelManager:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.model_path = None
            cls._instance.model_name = None
        return cls._instance

    def download_model(self, model_name="bartowski/Qwen2.5-14B-Instruct-GGUF"):
        """Download and cache model locally"""
        self.model_name = model_name

        if self.model_path and os.path.exists(self.model_path):
            print(f"Using cached model at: {self.model_path}")
            return self.model_path

        print("Downloading model (this may take 10-20 minutes first time)...")

        model_path = hf_hub_download(
            repo_id=model_name,
            filename="Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            cache_dir="./models"
        )

        self.model_path = model_path
        print(f"Model downloaded to: {model_path}")
        return model_path

    def load_model(self, model_path=None, load_tokenizer=False, hf_tokenizer_name=None):
        """
        Load model into memory for inference.
        - load_tokenizer: whether to load a HF tokenizer
        - hf_tokenizer_name: HF tokenizer repo name (if different from model)
        """
        if self.model:
            print("Model already loaded. Skipping reload.")
            return self.model

        if model_path:
            self.model_path = model_path

        if not self.model_path:
            raise ValueError("No model path specified")

        print("Loading model into memory...")

        # Load llama_cpp model
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_batch=512,
            n_threads=8,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
            chat_format="chatml"
        )

        print("Model loaded successfully!")

        # Optionally load HF tokenizer
        if load_tokenizer:
            tokenizer_name = hf_tokenizer_name or self.model_name
            print(f"Loading tokenizer '{tokenizer_name}' from Hugging Face Hub...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print("Tokenizer loaded successfully!")

        return self.model

    def test_model(self):
        """Test the loaded model with a simple query"""
        if not self.model:
            raise ValueError("Model not loaded")

        test_prompt = "Hello! Can you introduce yourself?"

        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=100,
            temperature=0.7
        )

        return response['choices'][0]['message']['content']


# Example usage
if __name__ == "__main__":
    manager = ModelManager()
    path = manager.download_model()
    # Set load_tokenizer=True only if you want a tokenizer (e.g., for HF models)
    model = manager.load_model(path, load_tokenizer=False)
    test_response = manager.test_model()
    print(f"Test response: {test_response}")
