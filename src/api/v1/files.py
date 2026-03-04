from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from src.services.file_services import FileServices
from src.services.rag_services import RAGServices
from src.core.logger import logger

from functools import lru_cache

router = APIRouter()


@lru_cache()
def get_file_services() -> FileServices:
    return FileServices()


@lru_cache()
def get_rag_services() -> RAGServices:
    return RAGServices()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = "default",
    file_services: FileServices = Depends(get_file_services),
    rag_services: RAGServices = Depends(get_rag_services),
):
    try:
        file_path = await run_in_threadpool(file_services.save_file, file)

        if file_path:
            logger.info(f"File uploaded successfully: {file_path}")

            documents = await run_in_threadpool(rag_services.load_file, file_path)
            logger.info(f"Documents loaded successfully: {len(documents)}")

            split_docs = await run_in_threadpool(
                rag_services.split_documents, documents
            )
            logger.info(f"Documents split successfully: {len(split_docs)}")

            await run_in_threadpool(
                rag_services.embed_documents,
                split_docs,
                collection_name if collection_name != "default" else file.filename,
            )
            logger.info(f"Documents embedded successfully: {len(split_docs)}")

            return JSONResponse(
                content={
                    "message": "File uploaded successfully",
                    "data": [{"file_path": file_path}],
                    "error": None,
                },
                status_code=status.HTTP_201_CREATED,
            )

        logger.error("Failed to upload file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file",
        )

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
