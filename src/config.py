import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_chat_model: str = os.getenv(
        "GEMINI_CHAT_MODEL", "gemini-2.5-flash"
    )
    gemini_embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
    )

    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv(
        "QDRANT_COLLECTION", "adaptive_rag_docs"
    )
    qdrant_local_path: str = os.getenv(
        "QDRANT_LOCAL_PATH", ".qdrant"
    )

    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")


settings = Settings()