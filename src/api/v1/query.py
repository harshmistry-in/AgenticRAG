from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from src.core.logger import logger
from src.services.rag_services import RAGServices

router = APIRouter()


def get_rag_services() -> RAGServices:
    # This could be more sophisticated (e.g., getting from app state), but
    # a simple dependency allows for easier mocking / caching later if needed.
    return RAGServices()


@router.post("/", status_code=status.HTTP_200_OK)
async def query_endpoint(
    query: str,
    collection_name: str = "default",
    top_k: int = 5,
    rag_services: RAGServices = Depends(get_rag_services),
):
    try:
        results = await run_in_threadpool(
            rag_services.search,
            query=query,
            collection_name=collection_name,
            top_k=top_k,
        )
        logger.info(f"Query successful: '{query}' - Results: {len(results)}")
        return JSONResponse(
            content={
                "message": "Documents found",
                "data": [{"results": results}],
                "error": None,
            },
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/ask-ai", status_code=status.HTTP_200_OK)
async def ask_ai_endpoint(
    query: str,
    collection_name: str = "default",
    top_k: int = 10,
    ai_services: AIServices = Depends(get_ai_services),
    rag_services: RAGServices = Depends(get_rag_services),
):
    try:
        results = await run_in_threadpool(
            rag_services.search,
            query=query,
            collection_name=collection_name,
            top_k=top_k,
        )

        if results:
            ai_response = await run_in_threadpool(
                ai_services.generate_answer,
                query=query,
                context=results,
            )
            logger.info(f"AI response generated successfully for query: '{query}'")
            return JSONResponse(
                content={
                    "message": "AI response generated successfully",
                    "data": [{"answer": ai_response}],
                    "error": None,
                },
                status_code=status.HTTP_200_OK,
            )

    except Exception as e:
        logger.error(f"Error executing AI query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
