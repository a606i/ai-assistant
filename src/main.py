from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from assistant import Assistant
import uvicorn
from rich.console import Console
from rich.prompt import Prompt
import typer

app = FastAPI()
assistant = Assistant()
console = Console()
cli = typer.Typer()

class Message(BaseModel):
    content: str

@app.post("/chat")
async def chat(message: Message):
    try:
        response = assistant.process_message(message.content)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_api():
    """Start the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

@cli.command()
def chat_cli():
    """Start interactive CLI chat"""
    console.print("[bold green]Initializing AI Assistant...[/bold green]")
    assistant.initialize()
    console.print("[bold green]Ready! Type 'exit' to quit.[/bold green]")
    
    while True:
        user_input = Prompt.ask("[bold blue]You[/bold blue]")
        
        if user_input.lower() == 'exit':
            break
            
        response = assistant.process_message(user_input)
        console.print(f"[bold green]Assistant:[/bold green] {response}")

@cli.command()
def start_server():
    """Start the API server"""
    console.print("[bold green]Starting AI Assistant API...[/bold green]")
    assistant.initialize()
    start_api()

if __name__ == "__main__":
    cli()
