# RAG Hybrid Search

A local Retrieval-Augmented Generation (RAG) prototype exploring the retrieval pipeline behind RAG systems: **embeddings, vector search, BM25, hybrid retrieval, metadata filtering, and cross-encoder reranking**.

Rather than treating retrieval as a black box, this project separates the major retrieval stages so their behavior, scores, and outputs can be inspected and evaluated independently.

## Architecture

```text
Documents
    │
    ▼
Embedding Model
BAAI/bge-base-en-v1.5
    │
    ▼
Weaviate Vector Database
    │
    ├── Semantic Search
    ├── BM25 Search
    ├── Hybrid Search
    └── Metadata Filtering
    │
    ▼
Candidate Documents
    │
    ▼
Cross-Encoder Reranker
BAAI/bge-reranker-base
    │
    ▼
Reranked Results
```

Weaviate runs locally in embedded mode and communicates with a custom Flask inference service for vectorization and reranking.

## Current Features

- Hugging Face sentence embeddings
- Cosine similarity and Euclidean distance experiments
- Embedded local Weaviate vector database
- Semantic search using vector similarity
- BM25 keyword search
- Hybrid BM25 + semantic retrieval
- Metadata filtering
- Deterministic UUID generation for document deduplication
- Custom Flask vectorization endpoint
- Custom Flask reranking endpoint
- Cross-encoder reranking using `BAAI/bge-reranker-base`
- Inspection of raw reranker logits and transformed rerank scores

## Repository Structure

```text
RAG_hybrid_search/
│
├── embedding/
│   ├── embedding_test.py
│   ├── huggingFace_embed.py
│   └── utils.py
│
├── llm_API/
│   ├── README.md
│   ├── utils.py
│   └── utils_test.py
│
├── vectorDB/
│   ├── create_joblib.py
│   ├── flask_app.py
│   ├── retrieval.py
│   ├── retriveal_sample_output.txt
│   ├── utils.py
│   └── vectorDB.py
│
└── img/
```

## Retrieval Pipeline

The current test dataset contains travel destinations with properties such as:

```text
place
state
description
best_season_to_visit
attractions
budget
user_ratings
last_updated
```

Textual properties are embedded and stored in Weaviate.

The retrieval layer can then perform:

### Semantic Search

Retrieves documents based on vector similarity between the query and stored documents.

### BM25 Search

Performs traditional lexical/keyword retrieval.

### Hybrid Search

Combines semantic and BM25 retrieval.

For example:

```python
result = collection.query.hybrid(
    query=q,
    filters=Filter.by_property("budget").contains_any(
        ["Low", "Moderate"]
    ),
    alpha=0.3,
    limit=2,
    rerank=Rerank(
        prop="attractions",
        query=q
    ),
    return_metadata=MetadataQuery(score=True)
)
```

`alpha` controls the balance between lexical and semantic retrieval.

Retrieved candidates can then be passed to the cross-encoder reranker.

## Reranking

The local Flask service uses:

```text
BAAI/bge-reranker-base
```

with Hugging Face:

```python
AutoTokenizer
AutoModelForSequenceClassification
```

Query-document pairs are scored by the model:

```text
(query, document)
        │
        ▼
Cross Encoder
        │
        ▼
Raw Logit
        │
        ▼
Sigmoid
        │
        ▼
Rerank Score
```

Example raw output:

```text
RAW RERANKER SCORES:
[-9.81592369 -10.17363644]

Sigmoid scores:
[5.45726201e-05 3.81618360e-05]
```

The transformed values preserve the ordering of the raw logits and are returned to Weaviate as reranking scores.

These scores are currently treated as **ranking signals rather than calibrated relevance probabilities**.

The final value can be inspected through:

```python
obj.metadata.rerank_score
```

## Example

Query:

```text
I want suggestions to travel during Winter.
I want cheap places.
```

Example hybrid retrieval result:

```text
place: Quebec City
state: Quebec
attractions:
    Old Quebec,
    Quebec Winter Carnival,
    Terrasse Dufferin

best_season_to_visit: Winter
user_ratings: 4.7
budget: Low

description:
A historic city especially popular in winter for snowy
streets, festivals, outdoor activities, and
European-style architecture.

metadata.rerank_score:
5.457262007015287e-05
```

The pipeline can combine semantic meaning, keyword matching, structured metadata filters, and reranking instead of relying on a single retrieval mechanism.

## Embedding Experiments

The `embedding/` directory contains experiments used to understand the embedding layer independently.

The current embedding model is:

```text
BAAI/bge-base-en-v1.5
```

Experiments include:

- sentence and word embeddings
- cosine similarity
- Euclidean distance
- query-document similarity
- simple semantic retrieval

For example, semantically related words can be compared directly in embedding space:

```text
apple → fruit
car   → automobile
```

This layer is used as the foundation for later vector retrieval in Weaviate.

## Local Inference API

`vectorDB/flask_app.py` provides the inference endpoints used by Weaviate:

```text
/.well-known/ready
/meta
/vectors
/rerank
```

The `/vectors` endpoint generates embeddings.

The `/rerank` endpoint accepts query-document pairs and returns reranking scores.

This keeps model inference separate from the vector database itself.

## Document Ingestion

Travel records are loaded into a Weaviate collection.

The collection currently vectorizes:

```text
place
state
description
best_season_to_visit
attractions
budget
```

while retaining additional structured metadata such as ratings and timestamps.

Objects use deterministic UUIDs generated from their content:

```python
uuid = generate_uuid5(document)
```

This prevents the same document from receiving a new random identifier every time the dataset is ingested.

## Running Locally

The project is currently an experimental prototype, so setup is still code-driven rather than packaged into a single command.

At a high level:

```text
1. Create and activate a Python virtual environment

2. Install the project dependencies

3. Generate/load the travel dataset

4. Start vectorDB/vectorDB.py
   ├── starts embedded Weaviate
   ├── initializes the collection
   └── starts the local inference service

5. Run vectorDB/retrieval.py
   └── execute semantic, BM25, or hybrid retrieval
```

Embedded Weaviate data is persisted locally under:

```text
./.collections
```

## Project Status

### Implemented

- [x] Hugging Face embeddings
- [x] Similarity metric experiments
- [x] Embedded Weaviate vector database
- [x] Document ingestion
- [x] Semantic search
- [x] BM25 search
- [x] Hybrid retrieval
- [x] Metadata filtering
- [x] Cross-encoder reranking
- [x] Local vectorization API
- [x] Local reranking API

### Next Steps

- [ ] Retrieval evaluation and metrics
- [ ] Reranker evaluation and score analysis
- [ ] RAG generation using retrieved context
- [ ] Retrieval/generation observability
- [ ] Improved process and thread shutdown coordination
- [ ] Reproducible dependency/setup configuration
- [ ] Refactor package structure as the prototype grows

## Background

This project began while studying the retrieval components of RAG systems through DeepLearning.AI's Retrieval-Augmented Generation material.

The reference implementations have been modified and extended to experiment with:

- local model inference
- Hugging Face embeddings
- Weaviate configuration
- hybrid retrieval
- metadata filtering
- cross-encoder reranking
- retrieval score inspection

The broader goal is to understand **why a RAG system retrieves a document**, rather than only passing data through a high-level RAG framework.


## Reference

- **Codebase reference**
    [1] DeepLearning AI. *Retrieval Augmented Generation (RAG)*. 
    https://learn.deeplearning.ai/courses/retrieval-augmented-generation/



## Use of ChatGPT
1. Debugging and first version of Readme.md