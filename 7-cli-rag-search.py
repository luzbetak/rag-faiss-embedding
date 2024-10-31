#!/usr/bin/env python3
# 7-cli-rag-search.py

import os
import asyncio
from query import QueryEngine
from loguru import logger
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress
from datastore_manager import RAGDatabaseInitializer

os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'OPENBLAS_MAIN_FREE': '1',
    'OMP_NUM_THREADS': '1'
})

class CLISearch:
    def __init__(self):
        self.query_engine = QueryEngine()
        self.console = Console()
        RAGDatabaseInitializer().load_indices()

    def print_results(self, results):
        if not results:
            self.console.print(Panel("No documents found.", 
                                   title="Search Results", 
                                   style="yellow"))
            return

        table = Table(title="Search Results", 
                     show_header=True, 
                     header_style="bold magenta")
        table.add_column("Doc #", style="dim", width=6)
        table.add_column("Title", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Content Preview", style="white")

        for i, doc in enumerate(results, 1):
            content = doc.get('content', 'N/A')
            preview = content[:200] + "..." if len(content) > 200 else content
            
            table.add_row(
                str(i),
                doc.get('title', 'N/A'),
                f"{doc.get('score', 0):.3f}",
                preview
            )

        self.console.print(table)

        self.console.print("\nFor detailed view of a document, enter its number (or press Enter to continue)")
        choice = input("> ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            doc = results[int(choice)-1]
            self.show_detailed_view(doc)

    def show_detailed_view(self, doc):
        self.console.print("\n")
        self.console.print(Panel(
            Text.from_markup(f"""[bold cyan]Title:[/] {doc.get('title', 'N/A')}
[bold cyan]URL:[/] {doc.get('url', 'N/A')}
[bold cyan]Score:[/] {doc.get('score', 0):.3f}
            
[bold cyan]Content Preview:[/]
{doc.get('content', 'N/A')[:1000]}..."""),  # Limit content preview
            title="Document Details",
            expand=False
        ))
        input("\nPress Enter to continue...")

    async def search_loop(self):
        self.console.print(Panel(
            "[bold]Welcome to RAG CLI Search[/]\n"
            "Enter your search queries below, or type 'exit' to quit",
            style="bold blue"
        ))

        while True:
            try:
                query = self.console.input("\n[bold yellow]Enter search query:[/] ")

                if query.lower() == 'exit':
                    self.console.print("\n[bold green]Goodbye![/]")
                    break

                with Progress(transient=True) as progress:
                    task = progress.add_task("[green]Searching...", total=None)
                    
                    results = await self.query_engine.search(query)
                    self.print_results(results)
                    
                    if results:
                        progress.update(task, description="[green]Generating response...")
                        response = await self.query_engine.generate_response(query, results)
                        
                        if response:
                            self.console.print(Panel(
                                response,
                                title="Generated Response",
                                style="green"
                            ))

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Search interrupted[/]")
                break
            except Exception as e:
                logger.exception("Search error")
                self.console.print(f"\n[bold red]Error during search:[/] {str(e)}")

def main():
    try:
        searcher = CLISearch()
        asyncio.run(searcher.search_loop())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\nFatal error: {str(e)}")
    finally:
        if 'searcher' in locals():
            searcher.query_engine.close()

if __name__ == "__main__":
    main()
