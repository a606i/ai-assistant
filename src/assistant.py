from model_manager import ModelManager
from typing import List, Dict
import json
from pathlib import Path
import time

class Assistant:
    def __init__(self):
        self.model_manager = ModelManager()
        self.conversation_history: List[Dict] = []
        self.system_prompt = """You are a helpful AI assistant. You aim to provide accurate, 
        helpful, and concise responses while being friendly and professional."""
        
    def initialize(self):
        """Initialize models and prepare the assistant"""
        self.model_manager.load_llm()
        self.model_manager.load_embedding_model()
    
    def process_message(self, message: str) -> str:
        """Process a user message and generate a response"""
        # Add message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": time.time()
        })
        
        # Construct prompt with history
        context = self._build_context()
        
        # Generate response
        response = self.model_manager.generate_text(context)
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time()
        })
        
        return response
    
    def _build_context(self) -> str:
        """Build context from conversation history"""
        context = self.system_prompt + "\n\n"
        
        # Add last few messages for context
        for msg in self.conversation_history[-4:]:
            role = msg["role"].capitalize()
            content = msg["content"]
            context += f"{role}: {content}\n"
        
        context += "Assistant:"
        return context
    
    def save_conversation(self, filepath: str = "conversation.json"):
        """Save the conversation history to a file"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def load_conversation(self, filepath: str = "conversation.json"):
        """Load a conversation history from a file"""
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                self.conversation_history = json.load(f)
                
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
