import os
import faiss
import numpy as np

from dotenv import load_dotenv

import google.generativeai as genai

from sentence_transformers import (
    SentenceTransformer
)

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def chunk_text(
    text,
    chunk_size=500
):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunks.append(
            text[i:i + chunk_size]
        )

    return chunks


def create_vector_store(
    chunks
):

    embeddings = embedding_model.encode(
        chunks
    )

    embeddings = np.array(
        embeddings
    ).astype(
        "float32"
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return (
        index,
        chunks
    )


def search_chunks(
    query,
    index,
    chunks,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype(
        "float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(
            chunks[idx]
        )

    return results


def generate_answer(
    question,
    retrieved_chunks
):

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, say:

"I could not find the answer in the uploaded documents."

Context:

{context}

Question:

{question}
"""

    response = gemini_model.generate_content(
        prompt
    )

    return response.text