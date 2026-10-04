"""
Codebase reference: https://learn.deeplearning.ai/courses/retrieval-augmented-generation/lesson/ssvq4/introduction-to-the-weaviate-api 
This code retrieves info using hybrid search, BM25 and semantic search in additional of metadata filtering from what has beem previuosly stored in vectorDB 

"""

from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
import weaviate
from dotenv import load_dotenv
from weaviate.classes.query import Rerank, MetadataQuery


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
    
    ##### BM25, Semantic, Hybrid Search ####
    ### params
    q= 'I want suggestions to travel during Winter. I want cheap places.'
    prop_="attractions" # reranking attribute/ criteria
    ######## Semantic Search 
    """ 
    
    result = collection.query.near_text(query = q, limit = 2)

    # Let's iterate over the result objects and return their properties
    print(f'Sematic Search:{q}')
    for obj in result.objects:
        print_object_properties(obj.properties)
    """

    """ 
    ######## BM25/ Keyword Search 
    result = collection.query.bm25(query = q, 
                                    filters = Filter.by_property('budget').contains_any(['Low','Moderate']),
                                    limit = 2)
    print(f'BM25 Search:{q}')
    for obj in result.objects:
        print_object_properties(obj.properties)
    """

    """
    ######## Hybrid Search; Alpha = % of BM25 
    result = collection.query.hybrid(query = q, 
                                        filters = Filter.by_property('budget').contains_any(['Low','Moderate']),
                                        alpha = 0.3,
                                        limit = 2)
    print(f'Hybrid Search:{q}')
    for obj in result.objects:
        print_object_properties(obj.properties)

    """

    ##### Reranking 
    
    response = collection.query.near_text( #  near text is vector search 
        query= q,  
        limit=5,
        rerank=Rerank(
            prop=prop_,                  # The property to rerank on
            query=q  # If not provided, the original query will be used
        ),
        return_metadata=MetadataQuery( 
        score=True
    )
    )
    print(f'Reranking on:{prop_} using query: {q}')
    for obj in response.objects:
        print_object_properties(obj.properties)
        print("Rerank score:", obj.metadata.score)
        print("----------------------")

    

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    if client is not None:
        client.close()
        print("Weaviate client closed.")