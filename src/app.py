from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from src.api.v1.files import router as file_router

app = FastAPI(
    title="Agentic RAG API's",
    version="0.0.0"
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