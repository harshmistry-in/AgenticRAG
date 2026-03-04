from langchain_cohere import CohereRerank
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.core.logger import logger
from src.core.settings import settings

from typing import List

from uuid_utils import uuid4


class RAGServices:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.client = QdrantClient(host="localhost", port=6333)
        self.embeddings = None
        self.reranker = CohereRerank(
            model="rerank-english-v3.0", top_n=3, cohere_api_key=settings.COHERE_API_KEY
        )
        logger.info("RAGServices initialized successfully")

    def load_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        logger.info(
            "Initialized HuggingFaceEmbeddings with model 'sentence-transformers/all-mpnet-base-v2'"
        )

    def load_file(self, file_path: str) -> List[Document]:
        logger.info(f"Loading file: {file_path}")
        loader = PyMuPDFLoader(file_path=file_path)
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Splitting documents: {len(documents)}")
        return self.splitter.split_documents(documents)

    def embed_documents(
        self, documents: List[Document], collection_name: str = "default"
    ):
        # Using lazy loading for embeddings to avoid unnecessary initialization if not needed
        if self.embeddings is None:
            self.load_embeddings()

        logger.info(f"Embedding documents: {len(documents)}")
        if not self.client.collection_exists(collection_name=collection_name):
            logger.info(
                f"Collection '{collection_name}' does not exist. Creating collection."
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        logger.info(f"Collection '{collection_name}' is ready for embedding")
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        import hashlib

        uuids = []
        for doc in documents:
            # Create a deterministic hash based on content and source info to avoid duplicates
            content = doc.page_content
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", 0)

            hash_input = f"{source}-{page}-{content}".encode("utf-8")
            doc_id = hashlib.sha256(hash_input).hexdigest()
            # Convert the first 32 hex characters to a valid UUID format
            formatted_uuid = f"{doc_id[:8]}-{doc_id[8:12]}-{doc_id[12:16]}-{doc_id[16:20]}-{doc_id[20:32]}"
            uuids.append(formatted_uuid)

        logger.info(
            f"Adding/updating documents to vector store: {len(documents)} logic with stable UUIDs"
        )
        vector_store.add_documents(documents, ids=uuids)

        return uuids

    def search(self, query: str, collection_name: str = "default", top_k: int = 10):
        # Using lazy loading for embeddings to avoid unnecessary initialization if not needed
        if self.embeddings is None:
            self.load_embeddings()

        logger.info(
            f"Searching for query: '{query}' in collection: '{collection_name}'"
        )
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        results = vector_store.similarity_search(query=query, k=top_k)
        reranked_docs = self.reranker.compress_documents(query=query, documents=results)
        logger.info(
            f"Search completed. Found {len(results)} results, reranked to {len(reranked_docs)} results."
        )
        results = [dict(res) for res in reranked_docs]
        logger.info(f"Reranked results: {results}")
        return results
