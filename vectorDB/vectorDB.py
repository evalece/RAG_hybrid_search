from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from typing import List
from tqdm import tqdm
import joblib
import weaviate
import re
from weaviate.util import generate_uuid5
from pprint import pprint
import os
from dotenv import load_dotenv

load_dotenv()

from utils import (
    suppress_subprocess_output,
    kill_processes_on_ports
)


# Kill processes on ports before importing flask_app
# WARNING: Running this cell twice may kill the active kernel
# kill_processes_on_ports([5001, 8080, 8097, 50050, 50051])
import flask_app

with suppress_subprocess_output():
    client = weaviate.connect_to_embedded(
        persistence_data_path="./.collections", # This is the path where the collections will be saved and persisted. It has a default value, but you can add a custom path
        environment_variables={
            "ENABLE_API_BASED_MODULES": "true", # Enable API based modules 
            "ENABLE_MODULES": 'text2vec-transformers, reranker-transformers', # We will be using a transformer model
            "TRANSFORMERS_INFERENCE_API":"http://127.0.0.1:5001/", # The endpoint the weaviate API will be using to vectorize
            "RERANKER_INFERENCE_API":"http://127.0.0.1:5001/" # The endpoint the weaviate API will be using to rerank, using flask_API 
        }
    )