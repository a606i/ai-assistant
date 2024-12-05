from model_manager import ModelManager
from rich.console import Console

console = Console()

def test_huggingface_integration():
    console.print("[bold green]Testing Hugging Face Integration...[/bold green]")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Test embedding model
    console.print("[yellow]Testing embedding model...[/yellow]")
    model_manager.load_embedding_model()
    test_text = "Hello, this is a test of the embedding model."
    embeddings = model_manager.get_embedding(test_text)
    console.print(f"Successfully generated embeddings of size: {len(embeddings)}")
    
    console.print("[bold green]SUCCESS: Hugging Face integration test completed![/bold green]")

if __name__ == "__main__":
    test_huggingface_integration()
