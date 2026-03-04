from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from services.ai_services import AIServices
from src.api.v1.files import router as file_router
from src.api.v1.query import router as query_router

app = FastAPI(
    title="Agentic RAG API's",
    version="0.0.0",
    description="API's for Agentic RAG system",
    summary="Agentic Retrieval-Augmented Generation API for intelligent document processing and query answering",
)


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return JSONResponse(
        content={
            "message": "Welcome to the Agentic RAG API's",
            "data": None,
            "error": None,
        }
    )


app.include_router(file_router, prefix="/api/v1/files", tags=["files"])
app.include_router(query_router, prefix="/api/v1/query", tags=["query"])
