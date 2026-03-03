from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.core.logger import logger

from typing import List

from uuid_utils import uuid4

class RAGServices:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.client = QdrantClient(host="localhost", port=6333)
        logger.info("RAGServices initialized successfully")
    
    def load_file(self, file_path: str) -> List[Document]:
        loader = PyMuPDFLoader(file_path=file_path)
        logger.info(f"File loaded successfully: {file_path}")
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Splitting documents: {len(documents)}")
        return self.splitter.split_documents(documents)
    
    def embed_documents(self, documents: List[Document], collection_name: str = "default"):
        logger.info(f"Embedding documents: {len(documents)}")
        if not self.client.collection_exists(collection_name=collection_name):
            logger.info(f"Collection '{collection_name}' does not exist. Creating collection.")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )

        logger.info(f"Collection '{collection_name}' is ready for embedding")
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
        
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        logger.info(f"Adding documents to vector store: {len(documents)}")
        vector_store.add_documents(documents, ids=uuids)