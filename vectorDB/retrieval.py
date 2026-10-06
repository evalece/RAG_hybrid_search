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
#import flask_app

from utils import (
    suppress_subprocess_output,
    print_object_properties

)
"""  
#### debugging 
try:

    client = weaviate.connect_to_local(   #check lsof -nP -iTCP -sTCP:LISTEN if any, variations due to standalone implemntation
    host="127.0.0.1",  # Use a string to specify the host
    port=8079,
    grpc_port=50050, # gRPC to Weaviate
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
    
######## Hybrid Search; Alpha = % of BM25  + Reranking 
def hybrid_S(client, q, prop_, alpha): # client= connection, q= query, prop_= properties, alpha % of BM25
    collection = client.collections.get("example_collection")
    print("############ Hybrid Search ####### ")
    result = collection.query.hybrid(
        query = q, 
        filters = Filter.by_property('budget').contains_any(['Low','Moderate']),
        alpha = 0.3,
        limit = 2,
        rerank=Rerank(
            prop=prop_,                  # The property to rerank on
            query=q  # If not provided, the original query will be used
            ),
        return_metadata=MetadataQuery( 
            score=True
        )
    )
    return result.objects
        #### debug 
"""
        print(f'Hybrid Search:{q}')
        for obj in result.objects:
            print_object_properties(obj.properties)
            print("metadata.rerank_score:", obj.metadata.rerank_score)
            print("----------------------")
"""


    

    ######## near_text + Reranking 
    """ 
    print("############ near text Search ####### ")
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

        print("metadata.rerank_score:", obj.metadata.rerank_score)
        print("----------------------")

    """
    
""" 

#debug

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    if client is not None:
        client.close()
        print("Weaviate client closed.")

""" 