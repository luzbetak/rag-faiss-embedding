#!/usr/bin/env python3
# 7-cli-rag-search.py

import os
import asyncio
from query import QueryEngine
from loguru import logger
from config import Config
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from datastore_manager import RAGDatabaseInitializer


# Set OpenBLAS environment variables
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'OPENBLAS_MAIN_FREE': '1',
    'OMP_NUM_THREADS': '1'
})

class CLISearch:
    def __init__(self):
        self.query_engine = QueryEngine()
        self.console = Console()
        # Load FAISS index from disk
        RAGDatabaseInitializer().load_indices()

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
            doc = results[int(choice)-1]
            self.show_detailed_view(doc)

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
            "[bold]Welcome to RAG CLI Search[/]\n"
            "Enter your search queries below, or type 'exit' to quit",
            style="bold blue"
        ))

        while True:
            try:
                # Get search query
                query = self.console.input("\n[bold yellow]Enter search query:[/] ")

                if query.lower() == 'exit':
                    self.console.print("\n[bold green]Goodbye![/]")
                    break

                # Show searching indicator
                with self.console.status("[bold green]Searching..."):
                    # Perform search
                    results = await self.query_engine.search(query)
                    
                    # Get generated response
                    response = await self.query_engine.generate_response(query, results)

                # Print results
                self.print_results(results)
                
                # Print generated response
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
        # Initialize and run search
        searcher = CLISearch()
        asyncio.run(searcher.search_loop())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\nFatal error: {str(e)}")
    finally:
        # Cleanup
        if 'searcher' in locals():
            searcher.query_engine.close()

if __name__ == "__main__":
    main()

