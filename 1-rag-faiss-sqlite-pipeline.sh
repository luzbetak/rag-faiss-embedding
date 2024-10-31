#!/bin/bash 

python process_unstructured_html.py --output-dir data --max-content-length 512 --max-sentences 2

sleep 1

python rag_datastore_manager.py

sleep 1




