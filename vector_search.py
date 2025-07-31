from qdrant_client import QdrantClient, models
from huggingface_hub import InferenceClient
import req


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def encode(text):
    return model.encode(text).tolist()


from qdrant_client import QdrantClient, models

import req

import req

# Query points (POST /collections/:collection_name/points/query)
response = req.post(
  "https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io/collections/bajaj/points/query",
  headers={
    "api-key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tVu-4QA6NR3Mv5AnRV-b5Rlz1QGz7tYqi9s2PRffLrA"
  },
  json={
    "query": encode("This Policy is a contract of insurance"),
    "with_payload": True,
    "limit": 4
  },
)
# Parse JSON
data = response.json()

# Access the actual points list
points = data.get("result", {}).get("points", [])

# Save and print the payloads
with open("query_results.txt", "w", encoding="utf-8") as f:
    for idx, point in enumerate(points, 1):
        payload = point.get("payload", {})
        for filename, document in payload.items():
            f.write(f"--- Document {idx} ({filename}) ---\n{document}\n\n")
            print(f"--- Document {idx} ({filename}) ---\n{document}\n")

print("✅ Results saved to query_results.txt")