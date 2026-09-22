import json
from sherlock_rag import ask, client, embed_model

with open("eval.json", encoding="utf-8") as f:
    evals = json.load(f)

print(f"\n{len(evals)} the question is checked...\n")
for item in evals:
    q = item["question"]
    q_vec = list(embed_model.embed([q]))[0].tolist()
    res = client.query_points(collection_name="holmes", query=q_vec, limit=3).points

    print(f"Q: {q}")
    for r in res:
        print(f" -> {r.payload['source']} | score={r.score:.3f}")
    print("-"*60)