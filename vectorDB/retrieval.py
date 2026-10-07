"""
Codebase reference: https://learn.deeplearning.ai/courses/retrieval-augmented-generation/lesson/ssvq4/introduction-to-the-weaviate-api 
This code retrieves info using hybrid search, BM25 and semantic search in additional of metadata filtering from what has beem previuosly stored in vectorDB 

"""
from weaviate.classes.query import Filter
from dotenv import load_dotenv
from weaviate.classes.query import Rerank, MetadataQuery

load_dotenv()

#### debugging 
 
######## Metadata filtering 
# Here we are fetching 2 objects with a filter by property, filtering by 'user_ratings, only objects with value greater or equal to 3.5'
#result = collection.query.fetch_objects(limit = 2, filters = Filter.by_property('user_ratings').greater_or_equal(3.5))

#print("result; filter by user rating > 3.5")
#for obj in result.objects:
    #print_object_properties(obj.properties)
    
######## Hybrid Search; Alpha = % of semantic  + Reranking ########
def hybrid_S(client, q, prop_, alpha_,limit_): # client= connection, q= query, prop_= properties, alpha % of semantic
    collection = client.collections.get("example_collection")
    print("######## Hybrid Search ####### ")
    result = collection.query.hybrid(
        query = q, 
        filters = Filter.by_property('budget').contains_any(['Low','Moderate']),
        alpha = alpha_,
        limit = limit_,
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