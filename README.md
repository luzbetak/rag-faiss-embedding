# RAG-FAISS Embedding

A project that combines Retrieval-Augmented Generation (RAG) with FAISS for efficient document retrieval and embedding-based search. It uses sentence-transformers for generating embeddings, FAISS for similarity search, and various text generation models for robust response generation.

## Features

- **Embeddings Generation**: Utilizes sentence-transformers for high-quality vector embeddings.
- **FAISS Integration**: Efficient and scalable similarity search for quick retrieval of relevant documents.
- **RAG Workflow**: Incorporates retrieval-augmented generation for improved response quality.
- **Multi-Model Support**: Supports different text generation models for flexible use cases.

## Requirements

- Python 3.x
- Required Python libraries listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/luzbetak/rag-faiss-embedding.git
   cd rag-faiss-embedding
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to generate embeddings and perform a RAG-based search:
```bash
python main.py --input "sample_query"
```

Refer to the `--help` flag for more options:
```bash
python main.py --help
```

## Project Structure

- **main.py**: Main script for embedding generation and retrieval.
- **models/**: Directory for pre-trained and custom models.
- **data/**: Sample data for testing and validation.
- **utils/**: Helper scripts for preprocessing and managing FAISS index.

## Contributing

Contributions are welcome! Fork the repo and submit a pull request for enhancements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
