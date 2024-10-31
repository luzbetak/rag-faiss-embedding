#!/usr/bin/env python3
import os
import asyncio
import aiohttp
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress

class APISearch:
    def __init__(self):
        self.api_url = "http://localhost:8000/search"
        self.console = Console()
    
    def print_results(self, results):
        if not results:
            self.console.print(Panel("No documents found.", 
                                   title="Search Results", 
                                   style="yellow"))
            return

        # Create results table
        table = Table(title="Search Results", 
                     show_header=True, 
                     header_style="bold magenta")
        table.add_column("Doc #", style="dim", width=6)
        table.add_column("Title", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Content Preview", style="white")

        for i, doc in enumerate(results, 1):
            # Truncate content for preview
            content = doc.get('content', 'N/A')
            preview = content[:200] + "..." if len(content) > 200 else content
            
            table.add_row(
                str(i),
                doc.get('title', 'N/A'),
                f"{doc.get('score', 0):.3f}",
                preview
            )

        self.console.print(table)

        # Print detailed view option
        self.console.print("\nFor detailed view of a document, enter its number (or press Enter to continue)")
        choice = input("> ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            self.show_detailed_view(results[int(choice)-1])

    def show_detailed_view(self, doc):
        """Show detailed view of a single document"""
        self.console.print("\n")
        self.console.print(Panel(
            Text.from_markup(f"""[bold cyan]Title:[/] {doc.get('title', 'N/A')}
[bold cyan]URL:[/] {doc.get('url', 'N/A')}
[bold cyan]Score:[/] {doc.get('score', 0):.3f}

[bold cyan]Content:[/]
{doc.get('content', 'N/A')}"""),
            title="Document Details",
            expand=False
        ))
        input("\nPress Enter to continue...")
    
    async def search_loop(self):
        # Print welcome message
        self.console.print(Panel(
            "[bold]Welcome to RAG API Search[/]\n"
            "Enter your search queries below, or type 'exit' to quit\n"
            "API URL: http://localhost:8000",
            style="bold blue"
        ))
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Get search query
                    query = self.console.input("\n[bold yellow]Enter search query:[/] ")
                    
                    if query.lower() == 'exit':
                        self.console.print("\n[bold green]Goodbye![/]")
                        break
                    
                    # Show searching indicator
                    with self.console.status("[bold green]Searching...") as status:
                        try:
                            async with session.post(
                                self.api_url,
                                json={"text": query, "top_k": 3}
                            ) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    
                                    # Print results
                                    self.print_results(data["similar_documents"])
                                    
                                    # Print generated response
                                    if data.get("generated_response"):
                                        self.console.print(Panel(
                                            data["generated_response"],
                                            title="Generated Response",
                                            style="green"
                                        ))
                                else:
                                    error_text = await response.text()
                                    self.console.print(Panel(
                                        f"API request failed with status {response.status}\n"
                                        f"Details: {error_text}",
                                        title="Error",
                                        style="bold red"
                                    ))
                        except aiohttp.ClientError as e:
                            self.console.print(Panel(
                                f"Error connecting to API server: {str(e)}\n"
                                "Make sure the API server is running at http://localhost:8000",
                                title="Connection Error",
                                style="bold red"
                            ))
                except KeyboardInterrupt:
                    self.console.print("\n[bold red]Search interrupted[/]")
                    break
                except Exception as e:
                    logger.exception("Unexpected error")
                    self.console.print(f"\n[bold red]Unexpected error:[/] {str(e)}")

def main():
    try:
        # Initialize and run search
        searcher = APISearch()
        asyncio.run(searcher.search_loop())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\nFatal error: {str(e)}")

if __name__ == "__main__":
    main()

