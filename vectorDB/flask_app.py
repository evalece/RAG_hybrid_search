'''
Code provided by https://learn.deeplearning.ai/courses/retrieval-augmented-generation/lesson/ssvq4/introduction-to-the-weaviate-api 

The follow code creates localhost endpoint for retrieval reranker 

'''
## To-do: optimizing thread and process coordination on shut down


from flask import Flask, request, jsonify
import threading
import json
# from FlagEmbedding import FlagReranker # requires torch>=2.5, hits x86_64 limitations 
import numpy as np
import torch
import threading
import logging
import os
from utils import generate_embedding
# Initialize models globally to load them once
#reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=False, cache_dir=os.environ["MODEL_M3"], use_fp16=False)


from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "BAAI/bge-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

reranker = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

reranker.eval() # sub for FlagEmbedding

app = Flask(__name__)

@app.route('/.well-known/ready', methods=['GET'])
def readiness_check():
    return "Ready", 200

@app.route('/meta', methods=['GET'])
def readiness_check_2():
    return jsonify({'status': 'Ready'}), 200

@app.route('/rerank', methods=['POST'])
def rerank():
    try:
        # Try normal JSON parsing first.
        # silent=True prevents Flask from returning 415
        # when Weaviate doesn't set application/json.
        text = request.get_json(silent=True)

        # Fall back to parsing raw request body.
        if text is None:
            text_str = request.get_data(as_text=True)
            text = json.loads(text_str)

        if (
            not isinstance(text, dict)
            or 'query' not in text
            or 'documents' not in text
        ):
            return jsonify({
                'error':
                "Expected dictionary containing 'query' and 'documents'."
            }), 400

        query = text['query']
        documents = text['documents']

        if not documents:
            return jsonify({'scores': []})

        compares = [(query, doc) for doc in documents]

        inputs = tokenizer(
            compares,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = reranker(**inputs)

        scores = outputs.logits.view(-1)
        scores_list =  np.array(scores.cpu().tolist())

        reranked_results = []
        print("RAW RERANKER SCORES:", scores_list) # To-do: Sigmoid variable x, to be converted to percentile + normalization for model confidence. 

        # Sigmoid transformation, 
        # Equivalent mathematically to the CDF of a standard logistic distribution.
        
        probabilities = 1 / (1 + np.exp(-scores_list)) 
        print("probabilities",probabilities)
        for i, doc_text in enumerate(documents):
            reranked_results.append({
                "document": doc_text,
                "score": float(probabilities[i]) 
            })

        return jsonify({
            'scores': reranked_results
        })

    except Exception as e:
        print(f"Unhandled error in /rerank: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/vectors', methods=['POST']) 
def vectorize():
    try:
        try:
            data = request.json.get('text')
        except Exception as e:
            try:
                data = request.data.decode("utf-8")
            except Exception as e:
                print(e)
        text = json.loads(data)
        if isinstance(text, str):
            text = [text]
        else:
            text =text['text']
            
        embeddings = generate_embedding(text)

        return jsonify({'vector': embeddings})


    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
app.logger.disabled = True
# Get the Flask app's logger
log = logging.getLogger('werkzeug')
# Set logging level (ERROR or CRITICAL suppresses routing logs)
log.setLevel(logging.ERROR)
def run_app():
    app.run(host='0.0.0.0', port=5001, debug = False)

flask_thread = threading.Thread(target=run_app)
flask_thread.start()
