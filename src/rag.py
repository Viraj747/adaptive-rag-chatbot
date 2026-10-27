from __future__ import annotations

import uuid
from typing import TypedDict, Literal

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from qdrant_client import QdrantClient, models
from langgraph.graph import StateGraph, START, END

from src.config import settings


class RAGState(TypedDict, total=False):
    query: str
    session_id: str
    route: str
    context: str
    answer: str
    sources: list[str]


def _check_api_key() -> None:
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add your Gemini API key to .env."
        )


def _embeddings() -> GoogleGenerativeAIEmbeddings:
    _check_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
    )


def _qdrant() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return QdrantClient(path=settings.qdrant_local_path)


def _ensure_collection(client: QdrantClient, vector_size: int) -> None:
    existing = {
        collection.name
        for collection in client.get_collections().collections
    }
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _embeddings().embed_documents(texts)



def split_text(
    text: str, chunk_size: int = 1000, overlap: int = 200
) -> list[str]:
    import re

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split at sentence boundaries where possible.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)

            # Keep the previous chunk's ending for context.
            current = (
                current[-overlap:].strip() + " " + sentence
                if current and overlap
                else sentence
            )

            # Handle a single sentence longer than the chunk size.
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[max(1, chunk_size - overlap):]

    if current:
        chunks.append(current)

    return chunks



def index_document(
    text: str, filename: str, description: str = ""
) -> int:
    chunks = split_text(text)
    if not chunks:
        raise ValueError("No readable text found in the uploaded document.")

    vectors = embed_texts(chunks)
    client = _qdrant()
    try:
        _ensure_collection(client, len(vectors[0]))

        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename),
                        )
                    ]
                )
            ),
            wait=True,
        )

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "description": description,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
            wait=True,
        )
        return len(points)
    finally:
        client.close()



def retrieve(query: str, limit: int = 6) -> tuple[str, list[str]]:
    vector = _embeddings().embed_query(query)
    client = _qdrant()

    try:
        existing = {
            collection.name
            for collection in client.get_collections().collections
        }

        if settings.qdrant_collection not in existing:
            return "", []

        result = client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        
        passages = []
        sources = []
        seen_texts = set()

        for point in result.points:
            payload = point.payload or {}
            text = str(payload.get("text", "")).strip()
            filename = str(
                payload.get("filename", "uploaded document")
            )

            normalized = " ".join(text.split()).casefold()

            if text and normalized not in seen_texts:
                passages.append(text)
                seen_texts.add(normalized)

            if filename not in sources:
                sources.append(filename)

        return "\n\n".join(passages), sources


    finally:
        client.close()



def has_indexed_documents() -> bool:
    client = _qdrant()
    try:
        existing = {
            collection.name
            for collection in client.get_collections().collections
        }
        if settings.qdrant_collection not in existing:
            return False

        info = client.get_collection(settings.qdrant_collection)
        return info.points_count > 0
    finally:
        client.close()



def classify_query(query: str) -> Literal["index", "general", "search"]:
    """Route current questions to search and other queries by intent."""
    q = query.lower().strip()

    current_markers = (
        "latest", "today", "current", "right now",
        "recent news", "this week", "this year",
    )
    if any(marker in q for marker in current_markers):
        return "search"

    document_markers = (
        "according to my document",
        "in the uploaded document",
        "in my document",
        "uploaded file",
        "from the document",
        "in the pdf",
        "in the text file",
    )
    if any(marker in q for marker in document_markers):
        return "index"

    return "general"






def _generate(
    query: str,
    context: str = "",
    source_type: str = "general knowledge",
) -> str:
    _check_api_key()

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.google_api_key,
    )

    if source_type == "web search results":
        instructions = (
            "You are an assistant answering questions using web search results. "
            "Use only facts supported by the supplied search context for claims "
            "about current events. Do not invent headlines, events, dates, names, "
            "quotes, statistics, or details. Distinguish reported facts from "
            "uncertainty. If the results are insufficient, say so clearly. "
            "Summarize the available results instead of pretending to have "
            "verified details they do not contain. Do not invent citations. "
            "The application will provide source URLs separately."
        )
    else:
        instructions = (
            "Answer naturally in clear, human-readable language. Use Markdown "
            "headings and bullet points when useful. If context is supplied, "
            "prioritize it and say when it is insufficient. Do not invent "
            "citations. For general knowledge questions without external context, "
            "answer using your learned knowledge."
        )

    prompt = (
        f"{instructions}\n\n"
        f"Question: {query}\n\n"
        f"Information source: {source_type}\n\n"
        f"Context:\n{context or '(No external context supplied)'}"
    )

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, str):
        return content.strip() or "I couldn't generate a response."

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])

        readable_text = "\n".join(text_parts).strip()
        return readable_text or "I couldn't generate a readable response."

    return str(content)

    



def _web_search(query: str) -> tuple[str, list[str]]:
    if not settings.tavily_api_key:
        return "", []

    from tavily import TavilyClient

    result = TavilyClient(
        api_key=settings.tavily_api_key
    ).search(
        query=query,
        topic="news",
        days=7,
        max_results=5,
        search_depth="advanced",
        include_answer=False,
    )

    rows = result.get("results", [])

    context_parts = []
    sources = []

    for row in rows:
        title = row.get("title", "").strip()
        url = row.get("url", "").strip()
        content = row.get("content", "").strip()

        if not url or not content:
            continue

        context_parts.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )
        sources.append(url)

    return "\n\n".join(context_parts), sources


def analyze_node(state: RAGState) -> RAGState:
    return {"route": classify_query(state["query"])}


def retrieve_node(state: RAGState) -> RAGState:
    context, sources = retrieve(state["query"])
    return {"context": context, "sources": sources}


def general_node(state: RAGState) -> RAGState:
    return {"answer": _generate(state["query"]), "sources": []}



def search_node(state: RAGState) -> RAGState:
    context, sources = _web_search(state["query"])

    # Temporary debugging: inspect Tavily's actual results.
    print("\n--- TAVILY SEARCH CONTEXT ---\n")
    print(context)
    print("\n--- TAVILY SOURCE URLS ---\n")
    print(sources)

    if not context:
        return {
            "answer": (
                "This question may need current information, but web search "
                "is not configured. Add TAVILY_API_KEY to .env and restart."
            ),
            "sources": [],
        }

    return {
        "answer": _generate(state["query"], context, "web search results"),
        "sources": sources,
    }



def generate_node(state: RAGState) -> RAGState:
    if not state.get("context"):
        return {
            "answer": (
                "I couldn't find relevant passages in the indexed documents. "
                "Try rephrasing your question or uploading the relevant document."
            ),
            "sources": [],
        }
    return {
        "answer": _generate(
            state["query"],
            state["context"],
            "retrieved document passages",
        ),
        "sources": state.get("sources", []),
    }


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("general", general_node)
    graph.add_node("web_search", search_node)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        lambda state: state["route"],
        {
            "index": "retrieve",
            "general": "general",
            "search": "web_search",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("general", END)
    graph.add_edge("web_search", END)
    return graph.compile()


def answer_query(query: str, session_id: str = "default") -> dict:
    result = build_graph().invoke(
        {"query": query, "session_id": session_id}
    )
    return {
        "type": "ai",
        "content": result.get("answer", ""),
        "route": result.get("route", "unknown"),
        "sources": result.get("sources", []),
    }