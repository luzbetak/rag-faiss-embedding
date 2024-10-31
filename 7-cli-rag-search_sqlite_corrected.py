#!/usr/bin/env python3
# 7-cli-rag-search.py - Modified for SQLite

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
        # Instantiate QueryEngine with SQLite database
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
        table = Table(title="Search Results")
        table.add_column("URL", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_column("Content", style="white")

        for result in results:
            table.add_row(result['url'], result['title'], result['content'])

        self.console.print(table)

    async def search(self, query_text: str):
        results = await self.query_engine.query(query_text)
        self.print_results(results)

    def run(self):
        self.console.print(Text("Welcome to CLI RAG Search", style="bold green"))
        self.console.print("Type your query below or 'exit' to quit.")
        
        while True:
            query_text = input("Query: ")
            if query_text.lower() == 'exit':
                break
            asyncio.run(self.search(query_text))

if __name__ == "__main__":
    cli_search = CLISearch()
    cli_search.run()
