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
    kill_processes_on_ports,
    print_object_properties
)


# Kill processes on ports before importing flask_app
# WARNING: Running this cell twice may kill the active kernel
# kill_processes_on_ports([5001, 8080, 8097, 50050, 50051])

import flask_app
client = None
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

    # To-Do: checking textual data ready
    data = joblib.load("data.joblib") # a set of places to visit, with some properties describing each location. The properties here are place, state, description, best_season_to_visit, attractions, budget, user_ratings, last_updated
    # print_object_properties(data[0])


    # 2.3 input + embedding for vectorDB settings for joblib's visit place data 
    vector_config_ = [Configure.NamedVectors.text2vec_transformers(
                    name="vector", # This is the name you will need to access the vectors of the objects in your collection
                    source_properties=['place', 'state', 'description', 'best_season_to_visit', 'attractions', 'budget'], # which properties should be used to generate a vector, they will be appended to each other when vectorizing
                    vectorize_collection_name = False, # This tells the client to not vectorize the collection name. 
                                                    # If True, it will be appended at the beginning of the text to be vectorized
                    inference_url="http://127.0.0.1:5001", # Since we are using an API based vectorizer, you need to pass the URL used to make the calls 
                                                        # This was setup in our Flask application
                )]

    if not client.collections.exists('example_collection'): # Creates only if the collection does not exist
        try:
            collection = client.collections.create(
                name='example_collection',


                vector_config=vector_config_, # The config we defined before,
            
                properties=[  # Define properties
                Property(name="place",vectorize_property_name=True,data_type= DataType.TEXT),
                Property(name="state",vectorize_property_name=True, data_type=DataType.TEXT),
                Property(name="description",vectorize_property_name=True, data_type=DataType.TEXT),
                Property(name="best_season_to_visit",vectorize_property_name=True, data_type=DataType.TEXT),
                Property(name="attractions",vectorize_property_name=True, data_type=DataType.TEXT),
                Property(name="budget",vectorize_property_name=True, data_type=DataType.TEXT),
                Property(name="user_ratings", data_type=DataType.NUMBER),
                Property(name="last_updated", data_type=DataType.DATE),
                        
            ]
            )
        except Exception as e:
            print(e)
    else:
        collection = client.collections.get("example_collection")

    print(collection) # check if client  created visited places collecitons



    ### 2.4 Adding elements into a Collection 

    # Set up a batch process with specified fixed size and concurrency
    with collection.batch.fixed_size(batch_size=1, concurrent_requests=1) as batch:
        # Iterate over a subset of the dataset
        for document in tqdm(data): # tqdm is a library to show progress bars
                # Generate a UUID based on the article_content text for unique identification
                uuid = generate_uuid5(document)

                # Add the object to the batch with properties and UUID. 
                # properties expects a dictionary with the keys being the properties.
                batch.add_object( 
                    properties=document,
                    uuid=uuid,
                )

    # 2. Insert to vectorDB completion, check  collection size:
    print("Collection size inserted = ", len(collection))

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    if client is not None:
        client.close()
        print("Weaviate client closed.")