from flask import Flask, request, jsonify
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np


encode_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
documents = []

app = Flask(__name__)


def split_text(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


@app.route('/add_document', methods=['POST'])
def add_document():
    text = request.json.get('document')

    chunks = split_text(text, 100)
    vectors = encode_model.encode(chunks)
    index.add(np.array(vectors).astype("float32"))
    documents.extend(chunks)

    return jsonify({"message": "Document indexed."})


@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query')
    top_k = request.args.get('top_k', default=5, type=int)

    vector = encode_model.encode([query_text])
    distances, indices = index.search(np.array(vector).astype("float32"), top_k)

    return jsonify({"results": [documents[i] for i in indices[0]]})


if __name__ == '__main__':
    app.run(debug=True)

