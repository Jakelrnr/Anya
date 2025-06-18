from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
#2212
class ModelManager:
    def __init__(self, base_model_name="     def download_model(self, model_name="bartowski/Qwen2.5-14B-Instruct-GGUF", lora_path="./openllama_lora"):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"🔄 Loading tokenizer from {self.lora_path if os.path.exists(self.lora_path) else self.base_model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lora_path if os.path.exists(self.lora_path) else self.base_model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"🔄 Loading base model from {self.base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"🔄 Loading LoRA weights from {self.lora_path}...")
        self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        self.model.eval()
        print("✅ Model with LoRA weights loaded successfully.")

    def test_model(self, instruction="Who created you?"):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("⚠️ Model is not loaded. Call load_model() first.")

        # Prompt format for OpenLLaMA (no special chat tags)
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

# Example usage
if __name__ == "__main__":
    manager = ModelManager()
    manager.load_model()
    output = manager.test_model("Tell me about your creator.")
    print(f"🧠 Model says:\n{output}")
