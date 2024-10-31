#!/usr/bin/env python3
# 2-cli-rag-search.py

import os
import asyncio
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress
from rag_datastore_manager import RAGDatabaseManager

# Configure OpenBLAS to avoid warnings
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'OPENBLAS_MAIN_FREE': '1',
    'OMP_NUM_THREADS': '1'
})

class CLISearch:
    def __init__(self):
        self.rag_manager = RAGDatabaseManager()
        self.console = Console()
        self.rag_manager.load_indices()
        logger.info("Initialized CLI Search with RAG Database Manager")

    def print_results(self, results):
        """Display search results in a formatted table"""
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
        table.add_column("Similarity", justify="right", style="green")
        table.add_column("Content Preview", style="white")

        for i, doc in enumerate(results, 1):
            content = doc.get('content', 'N/A')
            preview = content[:200] + "..." if len(content) > 200 else content
            # Convert FAISS distance to similarity score (inverse of distance)
            similarity = 1 / (1 + doc.get('distance', 0))
            
            table.add_row(
                str(i),
                doc.get('title', 'N/A'),
                f"{similarity:.3f}",
                preview
            )

        self.console.print(table)

        self.console.print("\nFor detailed view of a document, enter its number (or press Enter to continue)")
        choice = input("> ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            doc = results[int(choice)-1]
            self.show_detailed_view(doc)

    def show_detailed_view(self, doc):
        """Display detailed view of a single document"""
        similarity = 1 / (1 + doc.get('distance', 0))
        self.console.print("\n")
        self.console.print(Panel(
            Text.from_markup(f"""[bold cyan]Title:[/] {doc.get('title', 'N/A')}
[bold cyan]URL:[/] {doc.get('url', 'N/A')}
[bold cyan]Similarity Score:[/] {similarity:.3f}
            
[bold cyan]Content:[/]
{doc.get('content', 'N/A')}"""),
            title="Document Details",
            expand=False
        ))
        input("\nPress Enter to continue...")

    async def search(self, query: str) -> list:
        """Perform search using RAG Database Manager"""
        try:
            results = self.rag_manager.search_similar_documents(query)
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def search_loop(self):
        """Main search loop for CLI interface"""
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

                if not query.strip():
                    continue

                with Progress(transient=True) as progress:
                    task = progress.add_task("[green]Searching...", total=None)
                    
                    # Perform search
                    results = await self.search(query)
                    progress.update(task, completed=True)
                    
                    # Display results
                    self.print_results(results)

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Search interrupted[/]")
                break
            except Exception as e:
                logger.exception("Search error")
                self.console.print(f"\n[bold red]Error during search:[/] {str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        self.rag_manager.cleanup()
        logger.info("Cleaned up CLI Search resources")

def main():
    """Main entry point"""
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
            searcher.cleanup()

if __name__ == "__main__":
    main()
