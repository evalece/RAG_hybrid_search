import time
import weaviate
import flask_app

client = None

try:
    client = weaviate.connect_to_embedded(
        persistence_data_path="./.collections",
        environment_variables={
            "ENABLE_API_BASED_MODULES": "true",
            "ENABLE_MODULES": "text2vec-transformers,reranker-transformers",
            "TRANSFORMERS_INFERENCE_API": "http://127.0.0.1:5001/",
            "RERANKER_INFERENCE_API": "http://127.0.0.1:5001/"
        }
    )

    print("Weaviate running")
    print("Flask inference server running on :5001")
    print("Press Ctrl+C to stop.")

    # Keep this Python process alive
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping services...")

finally:
    if client is not None:
        client.close()
        print("Weaviate client closed.")