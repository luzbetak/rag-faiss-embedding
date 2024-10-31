#!/usr/bin/env python3

import re
import sys
from typing import Dict, Any, Optional, List, Literal
from loguru import logger
import json
from pathlib import Path
import spacy
from transformers import pipeline
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add("logs/document_validation.log", rotation="500 MB")

console = Console()

class DocumentValidator:
    def __init__(self, default_input='data/search-index.json', 
                 default_output='data/validated-index.json',
                 summarization_method: Literal['spacy', 'transformers', 'textrank', 'basic'] = 'basic'):
        """Initialize the document validator with summarization options"""
        self.required_fields = ['url', 'title', 'content']
        self.default_input = default_input
        self.default_output = default_output
        self.summarization_method = summarization_method
        
        # Initialize summarization components based on method
        if summarization_method == 'spacy':
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("spaCy model not found. Falling back to basic summarization")
                self.summarization_method = 'basic'
        elif summarization_method == 'transformers':
            try:
                self.summarizer = pipeline("summarization", 
                                         model="facebook/bart-large-cnn")
            except Exception as e:
                logger.warning(f"Transformers initialization failed: {e}. Falling back to basic summarization")
                self.summarization_method = 'basic'
        elif summarization_method == 'textrank':
            try:
                import networkx as nx
                self.nlp = spacy.load('en_core_web_sm')
                self.nx = nx
            except Exception as e:
                logger.warning(f"TextRank initialization failed: {e}. Falling back to basic summarization")
                self.summarization_method = 'basic'
        
        logger.info(f"Initialized DocumentValidator with {self.summarization_method} summarization")

    @staticmethod
    def clean_url(url: str) -> str:
        """Clean and validate URL"""
        if not url:
            return ''
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        return url

    @staticmethod
    def clean_title(title: str) -> str:
        """Clean and normalize title"""
        if not title:
            return ''
        title = ' '.join(title.split())
        return title.strip()

    @staticmethod
    def clean_content(content: str) -> str:
        """Clean and normalize content"""
        if not content:
            return ''
        # Remove special characters but keep periods and commas
        content = re.sub(r'[^\w\s.,]', ' ', content)
        content = ' '.join(content.split())
        return content.strip().lower()

    def validate_document(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and clean individual document fields"""
        try:
            if not doc:
                logger.warning("Empty document received")
                return None

            missing_fields = [field for field in self.required_fields if field not in doc]
            if missing_fields:
                logger.warning(f"Document missing required fields: {missing_fields}")
                return None

            cleaned_url = self.clean_url(doc['url'])
            cleaned_title = self.clean_title(doc['title'])
            cleaned_content = self.clean_content(doc['content'])

            if not cleaned_url or not re.match(r'^https?://', cleaned_url):
                logger.warning(f"Invalid URL in document: {doc.get('title', 'Unknown')}")
                return None

            if len(cleaned_content.split()) < 10:  # Minimum content length
                logger.warning(f"Content too short in document: {doc.get('title', 'Unknown')}")
                return None

            # Generate summary
            content_summary = self.summarize_text(cleaned_content)

            validated_doc = {
                "url": cleaned_url,
                "title": cleaned_title or "Untitled",
                "content": cleaned_content,
                "metadata": {
                    "word_count": len(cleaned_content.split()),
                    "original_length": len(doc.get('content', '')),
                    "cleaned_length": len(cleaned_content),
                    "summary": content_summary,
                    "summary_length": len(content_summary.split())
                }
            }

            return validated_doc

        except Exception as e:
            logger.error(f"Error validating document: {str(e)}")
            return None

    def summarize_spacy(self, text: str, num_sentences: int = 3) -> str:
        """Summarize text using spaCy's sentence segmentation"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return ' '.join(sentences[:num_sentences])

    def summarize_transformers(self, text: str, max_length: int = 130) -> str:
        """Summarize text using transformers pipeline"""
        try:
            summary = self.summarizer(text, 
                                    max_length=max_length, 
                                    min_length=30, 
                                    do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Transformer summarization error: {e}")
            return self.summarize_basic(text)

    def summarize_textrank(self, text: str, num_sentences: int = 3) -> str:
        """Summarize text using TextRank algorithm"""
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            # Create similarity matrix
            similarity_matrix = []
            for sent1 in doc.sents:
                similarities = []
                for sent2 in doc.sents:
                    if sent1.vector_norm and sent2.vector_norm:
                        similarity = sent1.similarity(sent2)
                    else:
                        similarity = 0
                    similarities.append(similarity)
                similarity_matrix.append(similarities)

            # Create graph and apply PageRank
            nx_graph = self.nx.from_numpy_array(similarity_matrix)
            scores = self.nx.pagerank(nx_graph)
            
            # Rank sentences and get top ones
            ranked_sentences = [(score, sent) 
                              for sent, score in zip(sentences, scores.values())]
            ranked_sentences.sort(reverse=True)
            
            return ' '.join(sent for _, sent in ranked_sentences[:num_sentences])
        except Exception as e:
            logger.error(f"TextRank summarization error: {e}")
            return self.summarize_basic(text)

    def summarize_basic(self, text: str, num_sentences: int = 3) -> str:
        """Basic summarization by taking first few sentences"""
        sentences = text.split('.')
        return '. '.join(sent.strip() for sent in sentences[:num_sentences] if sent.strip()) + '.'

    def summarize_text(self, text: str) -> str:
        """Summarize text using the selected method"""
        try:
            if self.summarization_method == 'spacy':
                return self.summarize_spacy(text)
            elif self.summarization_method == 'transformers':
                return self.summarize_transformers(text)
            elif self.summarization_method == 'textrank':
                return self.summarize_textrank(text)
            else:
                return self.summarize_basic(text)
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return self.summarize_basic(text)

    def batch_validate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a batch of documents with progress tracking"""
        valid_docs = []
        total_docs = len(documents)
        skipped_docs = 0

        with console.status("[bold green]Validating documents...") as status:
            for index, doc in enumerate(documents, 1):
                validated_doc = self.validate_document(doc)
                if validated_doc:
                    valid_docs.append(validated_doc)
                else:
                    skipped_docs += 1

                if index % 50 == 0:
                    status.update(f"[bold green]Processed {index}/{total_docs} documents")
                    logger.info(f"Processed {index}/{total_docs} documents")

        logger.info(f"Validation complete: {len(valid_docs)} valid, {skipped_docs} skipped")
        return valid_docs

    def display_summary(self, documents: List[Dict[str, Any]]) -> None:
        """Display an enhanced summary of the validated documents"""
        if not documents:
            console.print("[red]No valid documents to display[/red]")
            return

        # Document statistics
        table = Table(title="Document Validation Summary")
        table.add_column("Metric", justify="right", style="cyan")
        table.add_column("Value", justify="left", style="green")

        total_docs = len(documents)
        avg_word_count = sum(doc['metadata']['word_count'] for doc in documents) / total_docs
        avg_reduction = sum((doc['metadata']['original_length'] - doc['metadata']['cleaned_length']) / 
                          doc['metadata']['original_length'] * 100 for doc in documents) / total_docs
        avg_summary_length = sum(doc['metadata']['summary_length'] for doc in documents) / total_docs

        stats = [
            ("Total Documents", str(total_docs)),
            ("Unique URLs", str(len(set(doc['url'] for doc in documents)))),
            ("Average Word Count", f"{avg_word_count:.1f}"),
            ("Average Content Reduction", f"{avg_reduction:.1f}%"),
            ("Shortest Document", str(min(doc['metadata']['word_count'] for doc in documents))),
            ("Longest Document", str(max(doc['metadata']['word_count'] for doc in documents))),
            ("Average Summary Length", f"{avg_summary_length:.1f} words"),
            ("Summarization Method", self.summarization_method)
        ]

        for metric, value in stats:
            table.add_row(metric, value)

        console.print("\n")
        console.print(Panel("[bold blue]Document Validation Results[/bold blue]"))
        console.print(table)

        # Sample document preview
        console.print("\n[bold]Sample Document Preview:[/bold]")
        doc = documents[0]
        console.print(Panel(
            f"[cyan]Title:[/cyan] {doc['title']}\n"
            f"[cyan]URL:[/cyan] {doc['url']}\n"
            f"[cyan]Content Preview:[/cyan] {' '.join(doc['content'].split()[:20])}...\n"
            f"[cyan]Summary:[/cyan] {doc['metadata']['summary']}\n"
            f"[cyan]Word Count:[/cyan] {doc['metadata']['word_count']}"
        ))

    def validate_file(self, input_file: Optional[str] = None, output_file: Optional[str] = None, display: bool = True) -> None:
        """Validate documents from a JSON file with default paths"""
        try:
            input_path = Path(input_file or self.default_input)
            output_path = Path(output_file or self.default_output)

            if not input_path.exists():
                logger.error(f"Input file not found: {input_path}")
                return

            logger.info(f"Reading documents from {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)

            valid_documents = self.batch_validate_documents(documents)

            if not valid_documents:
                logger.error("No valid documents found after validation")
                return

            if display:
                self.display_summary(valid_documents)

            # Save validated documents
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(valid_documents, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(valid_documents)} validated documents to {output_path}")

            return valid_documents

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {input_path}")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")

def main():
    """Main entry point with summarization options"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate and summarize documents for RAG system")
    parser.add_argument("--input", "-i", help="Input JSON file (default: data/search-index.json)")
    parser.add_argument("--output", "-o", help="Output file (default: data/validated-index.json)")
    parser.add_argument("--summarize", "-s", choices=['spacy', 'transformers', 'textrank', 'basic'],
                        default='basic', help="Summarization method")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-display", action="store_true", help="Disable summary display")

    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")

    try:
        validator = DocumentValidator(summarization_method=args.summarize)
        validator.validate_file(args.input, args.output, display=not args.no_display)
    except Exception as e:
        logger.error(f"Error during document validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
