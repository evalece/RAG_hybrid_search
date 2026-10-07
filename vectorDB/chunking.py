""" 
Reference codebase: https://learn.deeplearning.ai/courses/retrieval-augmented-generation/lesson/rrngmm/chunking

"""

### To-do: merge chunking into generation pipeline


from typing import List
import requests
import re
import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.util import generate_uuid5
import tqdm
from weaviate.classes.query import Filter

client = weaviate.connect_to_local(   #check lsof -nP -iTCP -sTCP:LISTEN if any, variations due to standalone implemntation
host="127.0.0.1",  # Use a string to specify the host
port=8079,
grpc_port=50050, # gRPC to Weaviate
)

#### Strategy
# Goal:
# chunk online text data such as books
# 
# def get_chunks_xxxx : returns array of chunked data
# def get_book_text_objects : returns metadata applying to all chunks of the same book 
# def build_chunk_objs : returns an arrayo of dict, each containing get_book_text_objects() and get_chunks_xxxx[i], id=i 
# 


## fixed size chunking
def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float) -> List[str]:
    """
    Splits a given text into chunks of a fixed size with a specified overlap fraction between consecutive chunks.

    Parameters:
    - text (str): The input text to be split into chunks.
    - chunk_size (int): The number of words each chunk should contain.
    - overlap_fraction (float): The fraction of the chunk size that should overlap with the adjacent chunk.
      For example, an overlap_fraction of 0.2 means 20% of the chunk size will be used as overlap.

    Returns:
    - List[str]: A list of chunks (each a string) where each chunk might overlap with its adjacent chunk.
    """

    # Split the text into individual words
    text_words = text.split()
    
    # Calculate the number of words to overlap between consecutive chunks
    overlap_int = int(chunk_size * overlap_fraction)
    
    # Initialize a list to store the resulting chunks
    chunks = []
    
    # Iterate over text in steps of chunk_size to create chunks
    for i in range(0, len(text_words), chunk_size):
        # Determine the start and end indices for the current chunk,
        # taking into account the overlap with the previous chunk
        chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]
        
        # Join the selected words to form a chunk string
        chunk = " ".join(chunk_words)
        
        # Append the chunk to the list of chunks
        chunks.append(chunk)
    
    # Return the list of chunks
    return chunks


## Variable size chunking




def mixed_chunking(source_text, min_len):
    """
    Splits the given source_text into chunks using a mix of fixed-size and variable-size chunking.
    It first splits the text by Asciidoc markers and then processes the chunks to ensure they are 
    of appropriate size. Smaller chunks are merged with the next chunk, and larger chunks can be 
    further split at the middle or specific markers within the chunk.

    Args:
    - source_text (str): The text to be chunked.

    Returns:
    - list: A list of text chunks.
    """

    # Split the text by Asciidoc marker
    chunks = source_text.split("\n==")

    # Chunking logic
    new_chunks = []
    chunk_buffer = ""
    min_length = min_len # suggests 25

    for chunk in chunks:
        new_buffer = chunk_buffer + chunk  # Create new buffer
        new_buffer_words = new_buffer.split(" ")  # Split into words
        if len(new_buffer_words) < min_length:  # Check whether buffer length is too small
            chunk_buffer = new_buffer  # Carry over to the next chunk
        else:
            new_chunks.append(new_buffer)  # Add to chunks
            chunk_buffer = ""

    if len(chunk_buffer) > 0:
        new_chunks.append(chunk_buffer)  # Add last chunk, if necessary

    return new_chunks





## create book object by extract basic info except chunk into dict
### hard-coded an assumed sanitized data source, parsing each json object and download path into list of dict
def get_book_text_objects():
    # Source location
    text_objs = list()
    api_base_url = 'https://api.github.com/repos/progit/progit2/contents/book'  # Book base URL
    chapter_urls = ['/01-introduction/sections', '/02-git-basics/sections']  # List of section URLs

    # Loop through book chapters
    for chapter_url in chapter_urls:
        response = requests.get(api_base_url + chapter_url)  # Get the JSON data for the section files in the chapter

        # Loop through inner files (sections)
        for file_info in response.json():
            if file_info['type'] == 'file':  # Only process files (not directories)
                file_response = requests.get(file_info['download_url'])

                # Build objects including metadata
                chapter_title = file_info['download_url'].split('/')[-3]
                filename = file_info['download_url'].split('/')[-1]
                text_obj = {
                    "body": file_response.text,
                    "chapter_title": chapter_title,
                    "filename": filename
                }
                text_objs.append(text_obj)
    return text_objs


## create chunk object
def build_chunk_objs(book_text_obj, chunks, stra): 
    """
    Constructs a list of chunk objects from a given book text object 
    and its associated chunks.

    Args:
        book_text_obj (dict): A dictionary containing metadata for the book text, 
                              including 'chapter_title' and 'filename'.
        chunks (list): A list of chunks that represent parts of the book text.

    Returns:
        list: A list of dictionaries, each representing a chunk object 
              with 'chapter_title', 'filename', 'chunk', and 'chunk_index'.
    """
    chunk_objs = list()  # Initialize an empty list to store chunk objects
    
    # Iterate over the chunks with an index
    for i, c in enumerate(chunks):
        # Create a dictionary for each chunk with its associated data
        chunk_obj = {
            "chapter_title": book_text_obj["chapter_title"],  # Chapter title from the book text object
            "filename": book_text_obj["filename"],            # Filename from the book text object
            "chunking_strategy": stra,                            
            "chunk": c,                                       # The actual chunk of text
            "chunk_index": i                                  # The index of the chunk in the list
        }
        # Append the constructed chunk object to the list
        chunk_objs.append(chunk_obj)

    # Return the list of chunk objects
    return chunk_objs




#### Connect to Vector DB to Insert chunked data, assume vectorDB ready

client = weaviate.connect_to_local(   #check lsof -nP -iTCP -sTCP:LISTEN if any, variations due to standalone implemntation
host="127.0.0.1",  # Use a string to specify the host
port=8079,
grpc_port=50050, # gRPC to Weaviate
)

vector_config_= Configure.Vectors.text2vec_transformers(
        name="vector", # This is the name we will need to access the vectors of the objects in our collection
        #source_properties=['chunk'], # which properties should be used to generate a vector, they will be appended to each other when vectorizing
        vectorize_collection_name = False, # This tells the client to not vectorize the collection name. 
                                            # If True, it will be appended at the beginning of the text to be vectorized
        inference_url="http://127.0.0.1:5001", # Since we are using an API based vectorizer, we need to pass the URL used to make the calls 
                                                # This was setup in our Flask application
    )

if not client.collections.exists("chunking_example"):
    collection= client.collections.create(
        name= "chunking_example",
        vector_config=vector_config_, # The config we defined before,
        reranker_config=Configure.Reranker.transformers(), # The reranker config
            properties=[  # Define properties
            Property(name="chunk",data_type= DataType.TEXT),
            Property(name="chapter_title", data_type=DataType.TEXT),
            Property(name="filename",data_type=DataType.TEXT),
            Property(name="chunking_strategy",data_type=DataType.TEXT, tokenization = Tokenization.FIELD), # tokenization = Tokenization.FIELD means that the entire word will be treated as a token,
            Property(name="chunk_index",data_type=DataType.INT),
        ]       
        )
else:
    collection = client.collections.get("chunking_example")


######### adding chunks to the collection
# Get multiple sets of chunks - according to chunking strategy
book_text_objs= get_book_text_objects()

chunk_obj_sets = dict()
for book_text_obj in book_text_objs:
    text = book_text_obj["body"]  # Get the object's text body

    # Loop through chunking strategies:
    for strategy_name, chunks in [
        ["fixed_size_25", get_chunks_fixed_size_with_overlap(text, 25, 0.2)] 
       #["para_chunks_min_25", mixed_chunking(text)]
    ]:
        chunk_objs = build_chunk_objs(book_text_obj, chunks, strategy_name)

        if strategy_name not in chunk_obj_sets.keys():
            chunk_obj_sets[strategy_name] = list()

        chunk_obj_sets[strategy_name] += chunk_objs

if len(collection) == 0:
    with collection.batch.fixed_size(batch_size=1, concurrent_requests=20) as batch:
        for chunking_strategy, chunk_objects in tqdm.tqdm(chunk_obj_sets.items()):
            for chunk_obj in chunk_objects:
                chunk_obj["chunking_strategy"] = chunking_strategy
                batch.add_object(
                    properties=chunk_obj,
                    uuid=generate_uuid5(chunk_obj)
                )
                
print(f"Total count: {collection.aggregate.over_all().total_count}")
for chunking_strategy in chunk_obj_sets.keys():
    where_filter = Filter.by_property('chunking_strategy').equal(chunking_strategy) # Filter by chunking strategy
    count = collection.aggregate.over_all(filters = where_filter).total_count # Aggregate with filtering
    print(f"Object count for {chunking_strategy}: {count}")