import weaviate
from weaviate.classes.query import Filter

client = weaviate.connect_to_local(   #check lsof -nP -iTCP -sTCP:LISTEN if any, variations due to standalone implemntation
host="127.0.0.1",  # Use a string to specify the host
port=8079,
grpc_port=50050, # gRPC to Weaviate
)
collection = client.collections.get("chunking_example")

chunk_s=["fixed_size_25"] # a truncated set 


## example of query, filtered by chunk strategy
search_string = "history of git"  # Or "available git remote commands" 

for chunking_strategy in chunk_s:
    # where_filter = Filter.by_property('chunking_strategy').equal(chunking_strategy)
    response = collection.query.near_text(search_string, filters =  Filter.by_property('chunking_strategy').equal(chunking_strategy), limit = 2)
    print(f"RETRIEVED OBJECTS FOR CHUNKING STRATEGY {chunking_strategy.upper()}:\n")
    for i, obj in enumerate(response.objects):
        print(f"===== Object {i} =====")
        print(f"{obj.properties['chunk']}")
        print()