from pathlib import Path
from ctransformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv

class ModelManager:
    def __init__(self, models_dir: str = "../models"):
        load_dotenv()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.llm = None
        self.embedding_model = None
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
    def load_llm(self, model_type: str = "gpt4all-j"):
        """Load the language model optimized for CPU inference"""
        if model_type == "gpt4all-j":
            model_path = self.models_dir / "gpt4all-j-v1.3-groovy.bin"
            if not model_path.exists():
                # Model will be downloaded automatically
                pass
            
            self.llm = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                model_type="gpt4all-j",
                max_new_tokens=256,
                context_length=2048,
                threads=os.cpu_count(),
            )
        return self.llm
    
    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        self.embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cpu',
            token=self.hf_token
        )
        return self.embedding_model
    
    def generate_text(self, prompt: str, max_length: int = 256) -> str:
        """Generate text using the loaded LLM"""
        if self.llm is None:
            self.load_llm()
        
        return self.llm(prompt, max_new_tokens=max_length)
    
    def get_embedding(self, text: str) -> list:
        """Get embeddings for the input text"""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        return self.embedding_model.encode(text).tolist()
