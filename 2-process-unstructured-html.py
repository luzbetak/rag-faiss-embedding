#!/usr/bin/env python3

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import argparse
from pymongo import MongoClient
from datetime import datetime
from json import JSONEncoder

from bs4 import BeautifulSoup
import spacy
from spacy.language import Language

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DateTimeEncoder(JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class IndexEntry:
    """Data class for storing content data"""
    _id_counter = 1  # Class variable to keep track of incremental IDs

    def __init__(self, url: str, title: str, content: str):
        self.id = IndexEntry._id_counter  # Assign the current counter value as the ID
        IndexEntry._id_counter += 1       # Increment the counter for the next entry
        self.url = url
        self.title = title
        self.content = content
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        """Convert entry to dictionary with ISO format dates"""
        return {
            'id': self.id,  # Include the unique ID in the output
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# Updated `process_html_file` function
def process_html_file(self, file_path: Path) -> Optional[IndexEntry]:
    """Process a single HTML file and return a content entry"""
    try:
        logger.debug(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()

            # Extract and clean text
            body_text = soup.get_text(separator=' ', strip=True)
            clean_body_text = self.clean_text(body_text)

            if not clean_body_text:
                logger.warning(f"Skipping {file_path}: No meaningful content")
                return None

            content = self.summarize_text(clean_body_text)
            relative_path = str(file_path.relative_to(Path.cwd()))
            url_path = f"https://kevinluzbetak.com/{relative_path}"

            # Return a new IndexEntry with an incremented ID
            return IndexEntry(
                url=url_path.strip(),
                title=file_path.name,
                content=content
            )

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


class TextSummarizer:
    def __init__(self, output_dir: str = "data", mongodb_uri: str = "mongodb://localhost:27017/"):
        """Initialize the text summarizer with spaCy NLP and MongoDB connection"""
        self.output_dir = Path(output_dir).resolve()
        self.nlp = self._initialize_spacy()
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client.rag_database
        self.collection = self.db.documents
        logger.info(f"Initialized TextSummarizer with output directory: {self.output_dir}")
        logger.info(f"Connected to MongoDB database: {self.db.name}")

    def _initialize_spacy(self) -> Language:
        """Initialize spaCy with error handling"""
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model")
            return nlp
        except OSError:
            logger.warning("SpaCy model 'en_core_web_sm' not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            return spacy.load("en_core_web_sm")

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\b(menu|html|title|include)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'-+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def summarize_text(self, text: str) -> str:
        """Summarize the text using spaCy"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return ' '.join(sentences[:3])

    def _write_to_mongodb(self, entries: List[IndexEntry]) -> None:
        """Write entries to MongoDB"""
        try:
            # Convert entries to dictionaries
            documents = [entry.to_dict() for entry in entries if entry and entry.url and entry.title and entry.content]
            
            if not documents:
                logger.error("No valid entries to write to MongoDB!")
                return

            # Clear existing documents
            self.collection.delete_many({})
            
            # Insert new documents
            result = self.collection.insert_many(documents)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} documents into MongoDB")
            
            # Create text index for search
            self.collection.create_index([("content", "text")])
            logger.info("Created text index on content field")

        except Exception as e:
            logger.error(f"Error writing to MongoDB: {e}", exc_info=True)
            raise

    def _write_index_file(self, entries: List[IndexEntry]) -> None:
        """Write the summaries to a JSON file"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.output_dir / "search-index.json"
            
            valid_entries = [
                entry.to_dict() for entry in entries
                if entry and entry.url and entry.title and entry.content
            ]

            if not valid_entries:
                logger.error("No valid entries to write!")
                return

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(valid_entries, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
            logger.info(f"Successfully created {output_file} with {len(valid_entries)} entries")

        except Exception as e:
            logger.error(f"Error writing index file: {e}", exc_info=True)
            raise

    def generate_index(self) -> None:
        """Generate the content index from HTML files"""
        logger.info("Starting content index generation...")

        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")

        html_files = [
            p for p in current_dir.rglob("*.html")
            if p.name != "index.html" and not str(p).startswith(str(self.output_dir))
        ]

        if not html_files:
            logger.warning("No HTML files found to process")
            return

        logger.info(f"Found {len(html_files)} HTML files to process")
        logger.debug(f"Files to process: {[str(f) for f in html_files]}")

        with ThreadPoolExecutor() as executor:
            entries = list(filter(None, executor.map(
                self.process_html_file,
                html_files
            )))

        if not entries:
            logger.error("No valid entries generated from HTML files")
            return

        logger.info(f"Generated {len(entries)} valid entries")

        # Write to both MongoDB and JSON file
        self._write_to_mongodb(entries)
        self._write_index_file(entries)

        logger.info("Index generation completed")

    def __del__(self):
        """Cleanup MongoDB connection"""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate content index from HTML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for the content index"
    )
    parser.add_argument(
        "--mongodb-uri",
        default="mongodb://localhost:27017/",
        help="MongoDB connection URI"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        summarizer = TextSummarizer(args.output_dir, args.mongodb_uri)
        summarizer.generate_index()
    except Exception as e:
        logger.error(f"Failed to generate content index: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()

