from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

COLLECTION = "Sherlock"
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(path="./qdrant_db")

print("Sherlock RAG is ready! For exit, write 'exit'\n")
while True:
    question = input("Answer: ")
    if question.lower() in ["exit","quit"]:
        break
    vec = embed_model.encode(question).tolist()
    results = qdrant.query_points(collection_name=COLLECTION, query=vec, limit=5).points
    context = "\n\n".join([r.payload['text'] for r in results])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer (english, 3-4 sentences, based on context only):"
    resp = ollama.generate(model='llama3', prompt=prompt, options={'temperature': 0.2})
    print(f"\nAnswer: {resp['response']}\n")

qdrant.close()
