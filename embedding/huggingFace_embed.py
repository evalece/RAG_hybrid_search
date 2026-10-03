"""
Reference: https://learn.deeplearning.ai/courses/retrieval-augmented-generation/lesson/33bze/vector-embeddings-in-rag 

Codebase is provided by the course with modifications made. 

Summary: Embedding using open source lib on Hugging Face
"""

from embedding_test import (
    cosine_similarity,
    euclidean_distance,
    retrieve_relevant
    

)

from utils import (
    display_widget
)



# 2.2 Loading from the Hugging Face

from sentence_transformers import SentenceTransformer

# Load the pre-trained sentence transformer model
model_name =  "BAAI/bge-base-en-v1.5" 
model = SentenceTransformer(model_name)

# To get a string embedded, just pass it to the model.
res = model.encode("RAG is awesome")
print(res.shape)
# An array of strings can be passed, and the output will be an array of vectors, each with 768 dimensions.
model.encode(['apple', 'car'])

# Print the first 100 elements of the embedding.
print(res[:100])


# 2.3 Embedding using Hugging Face; word and sentence emdeddings
    # calculating cosine similarity and euclidian distance
print("2.3 Embedding using Hugging Face; word and sentence emdeddings calculating cosine similarity and euclidian distance")
words = ['apple', 'car', 'fruit', 'automobile', 'love', 'sentiment']
vectorized_words = model.encode(words)

word = 'apple'
print(f"{word}:")
for i, w in enumerate(words):
    # Get the vectorized word for the word defined above
    vectorized_word = vectorized_words[words.index(word)]
    print(f"\t{w}:\t\tCosine Similarity: {cosine_similarity(vectorized_word, vectorized_words[i])[0]:.4f}")
print("\n\n\n")
for i, w in enumerate(words):
    # Get the vectorized word for the word defined above
    vectorized_word = vectorized_words[words.index(word)]
    print(f"\t{w}:\t\tEuclidean Distance: {euclidean_distance(vectorized_word, vectorized_words[i])[0]:.4f}")


# 2.4 
documents = [
    "Mt. Fuji is a breathtaking place to explore during autumn.",
    "Santorini offers stunning views to admire during spring.",
    "Banff National Park is a picturesque destination to visit in the summer.",
    "The Great Wall of China is a spectacular site to experience during winter.",
    "The fjords of Norway are a magical place to cruise through in the spring.",
    "Prague is an enchanting city to wander through in winter.",
    "Kyoto's cherry blossoms create a beautiful scene to witness during spring.",
    "Marrakech offers vibrant markets and culture to enjoy in the fall.",
    "The Maldives are a paradisiacal getaway to savor during summer.",
    "The Christmas markets in Vienna are a festive delight to explore in winter."
]

query = "Suggest to me great places to visit in Asia."
score = retrieve_relevant(query, documents, metric='cosine_similarity')

print("score", score)

display_widget(model)