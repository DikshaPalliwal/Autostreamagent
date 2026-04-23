"""
rag_pipeline.py
---------------
Handles the Retrieval-Augmented Generation (RAG) pipeline for AutoStream.

How it works:
1. The knowledge base (Markdown file) is split into meaningful text chunks.
2. Each chunk is embedded using sentence-transformers (all-MiniLM-L6-v2).
3. Chunks + embeddings are stored in an in-memory ChromaDB collection.
4. At query time, the user's question is embedded and the top-K most
   semantically similar chunks are returned as context for the LLM.
"""

import os
import chromadb
from chromadb.utils import embedding_functions


# ---------------------------------------------------------------------------
# Knowledge chunks – manually curated for precision
# In a production system, you'd auto-chunk the Markdown file.
# ---------------------------------------------------------------------------
KNOWLEDGE_CHUNKS = [
    {
        "id": "about",
        "text": (
            "AutoStream is a SaaS platform providing automated video editing tools "
            "for content creators including YouTubers, Instagrammers, and TikTok creators."
        ),
    },
    {
        "id": "basic_plan",
        "text": (
            "Basic Plan costs $29 per month. It includes 10 videos per month, "
            "supports up to 720p resolution, does not include AI captions, "
            "and comes with standard business-hours support only."
        ),
    },
    {
        "id": "pro_plan",
        "text": (
            "Pro Plan costs $79 per month. It includes unlimited videos per month, "
            "supports up to 4K resolution, includes AI captions, "
            "and comes with 24/7 priority customer support."
        ),
    },
    {
        "id": "plan_comparison",
        "text": (
            "Comparing plans: Basic is $29/month with 10 videos at 720p and no AI captions. "
            "Pro is $79/month with unlimited videos at 4K with AI captions. "
            "The key upgrade benefits of Pro are: 4K resolution, AI captions, and 24/7 support."
        ),
    },
    {
        "id": "refund_policy",
        "text": (
            "Refund Policy: AutoStream provides refunds only within the first 7 days of purchase. "
            "No refunds are given after 7 days. Contact support@autostream.io for eligible refunds."
        ),
    },
    {
        "id": "support_policy",
        "text": (
            "Support Policy: 24/7 live customer support is available exclusively on the Pro plan. "
            "Basic plan users get standard support during business hours (9AM–6PM EST, Mon–Fri)."
        ),
    },
    {
        "id": "cancellation",
        "text": (
            "Cancellation Policy: Users can cancel at any time. "
            "Access continues until the end of the current billing period after cancellation."
        ),
    },
    {
        "id": "upgrade",
        "text": (
            "Users can upgrade from Basic to Pro at any time. "
            "The price difference is prorated. There is no free trial available. "
            "The Basic plan at $29/month is a low-commitment entry point."
        ),
    },
    {
        "id": "platforms_payment",
        "text": (
            "AutoStream supports videos from YouTube, Instagram, TikTok, Facebook, and standard "
            "MP4/MOV formats. Accepts major credit cards, PayPal, and annual billing with 15% discount."
        ),
    },
]


def initialize_rag() -> chromadb.Collection:
    """
    Initialize ChromaDB in-memory client, create a collection,
    and populate it with knowledge chunks if it's empty.

    Returns:
        chromadb.Collection: The populated vector store collection.
    """
    client = chromadb.Client()  # In-memory client (no persistence needed for this demo)

    # Use sentence-transformers for local, free embeddings
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="autostream_knowledge_base",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    # Only add documents if the collection is fresh/empty
    if collection.count() == 0:
        collection.add(
            documents=[chunk["text"] for chunk in KNOWLEDGE_CHUNKS],
            ids=[chunk["id"] for chunk in KNOWLEDGE_CHUNKS],
        )
        print(f"[RAG] Initialized knowledge base with {len(KNOWLEDGE_CHUNKS)} chunks.")

    return collection


def retrieve_context(query: str, collection: chromadb.Collection, n_results: int = 3) -> str:
    """
    Retrieve the top-N most relevant knowledge chunks for a given query.

    Args:
        query:      The user's question or message.
        collection: The ChromaDB collection to search.
        n_results:  Number of chunks to retrieve (default: 3).

    Returns:
        A single string with the retrieved chunks joined by newlines.
    """
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )
    # results["documents"] is a list-of-lists (one list per query)
    docs = results["documents"][0]
    return "\n- ".join([""] + docs)  # Format as a bullet list
