import weaviate
from pprint import pprint
from weaviate.classes.config import Configure
import inspect

client = weaviate.connect_to_local(
    port=8079,
    grpc_port=50050
)

try:
    print(weaviate.__version__)
    collection = client.collections.get("Example_collection")

    config = collection.config.get()

    pprint(config)


    print("NamedVectors:")
    print(inspect.signature(Configure.NamedVectors.text2vec_transformers))

    print("\nVectors:")
    print(inspect.signature(Configure.Vectors.text2vec_transformers))

finally:
    client.close()