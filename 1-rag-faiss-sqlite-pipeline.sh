#!/bin/bash 

python process-unstructured-html.py --output-dir data --max-content-length 512 --max-sentences 2

sleep 1

python rag-datastore-manager.py

sleep 1




