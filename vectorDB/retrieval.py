"""
Codebase reference: https://learn.deeplearning.ai/courses/retrieval-augmented-generation/lesson/ssvq4/introduction-to-the-weaviate-api 
This code retrieves info using hybrid search, BM25 and semantic search in additional of metadata filtering from what has beem previuosly stored in vectorDB 

"""

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
client = None 
import flask_app

from utils import (
    suppress_subprocess_output,
    print_object_properties

)

try:

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

    
    collection = client.collections.get("example_collection")
    ######## Metadata filtering 
    # Here we are fetching 2 objects with a filter by property, filtering by 'user_ratings, only objects with value greater or equal to 3.5'
    #result = collection.query.fetch_objects(limit = 2, filters = Filter.by_property('user_ratings').greater_or_equal(3.5))
 
    #print("result; filter by user rating > 3.5")
    #for obj in result.objects:
        #print_object_properties(obj.properties)


    ######## Semantic Search 
    q= 'I want suggestions to travel during Winter. I want cheap places.'
    result = collection.query.near_text(query = q, limit = 2)

    # Let's iterate over the result objects and return their properties
    print(f'Sematic Search:{q}')
    for obj in result.objects:
        print_object_properties(obj.properties)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    if client is not None:
        client.close()
        print("Weaviate client closed.")