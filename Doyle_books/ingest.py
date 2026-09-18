from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os

COLLECTION = "Sherlock"
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Bütün txt faylları axtar (harada olur olsun)
txt_files = []
for root, dirs, files in os.walk("."):
    if "qdrant_db" in root:
        continue
    for f in files:
        if f.endswith(".txt"):
            txt_files.append(os.path.join(root, f))

print(f"txt files found: {txt_files}")
if not txt_files:
    print("NO TEXTS FOUND! Where are the books??")
    exit()

qdrant = QdrantClient(path="./qdrant_db")
try:
    qdrant.get_collection(COLLECTION)
    print("There are collections")
except:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    print("New collection created")

idx = 0
for path in txt_files:
    print(f"Oxunur: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        for i in range(0, len(content), 800):
            chunk = content[i:i+800].strip()
            if len(chunk) < 100:
                continue
            vec = embed_model.encode(chunk).tolist()
            qdrant.upsert(
                collection_name=COLLECTION,
                points=[models.PointStruct(id=idx, vector=vec, payload={"text": chunk, "source": path})]
            )
            idx += 1
    print(f" -> {idx}")

qdrant.close()
print(f"FINISH! Total {idx} pieces loaded")