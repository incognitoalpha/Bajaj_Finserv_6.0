from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
import requests
from pdfminer.high_level import extract_text
import torch
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

app = FastAPI()

# Load model once
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")

client = QdrantClient(
    url="https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io:6333",
   api_key=os.getenv("api_key"),
)

UPLOAD_DIR = "."
os.makedirs(UPLOAD_DIR, exist_ok=True)
global_id = 0


class RunRequest(BaseModel):
    documents: str
    questions: List[str]


def download_file_from_url(url, save_dir="."):
    filename = url.split("?")[0].split("/")[-1]
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file")
    local_path = os.path.join(save_dir, filename)
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


def extract_text_from_pdf(file_path):
    try:
        return extract_text(file_path)
    except Exception as e:
        print(f"[PDF ERROR] {file_path}: {e}")
        return ""


def langchain_chunk(text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=0,
        model_name="gpt-3.5-turbo",
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def encode(texts):
    return model.encode(texts, convert_to_tensor=True)


@app.post("/hackrx/run")
async def hackrx_run(
    payload: RunRequest,
    authorization: Optional[str] = Header(None)
):
    global global_id

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc_url = payload.documents
    questions = payload.questions

    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing document URL or questions")

    try:
        file_path = download_file_from_url(doc_url, save_dir=UPLOAD_DIR)
        text = extract_text_from_pdf(file_path)
        chunks = langchain_chunk(text)
        chunk_vectors = encode(chunks)

        points = []
        for i, chunk in enumerate(chunks):
            points.append(
                models.PointStruct(
                    id=global_id,
                    payload={"filename": os.path.basename(file_path), "chunk": chunk},
                    vector=chunk_vectors[i].tolist()
                )
            )
            global_id += 1

        client.upsert(collection_name="bajaj", points=points)

        # Question Answering
        question_embeddings = encode(questions)
        results = []
        for q_idx, q_embed in enumerate(question_embeddings):
            search = client.search(
                collection_name="bajaj",
                query_vector=q_embed.tolist(),
                limit=3
            )
            top_chunks = [hit.payload["chunk"] for hit in search if "chunk" in hit.payload]
            result_text = "\n\n".join(top_chunks)
            results.append({
                "question": questions[q_idx],
                "answer": result_text.strip()
            })

        # Save results to query_results.txt
        with open("query_results.txt", "w", encoding="utf-8") as f:
            for item in results:
                f.write(f"Question: {item['question']}\nAnswer: {item['answer']}\n\n")

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
