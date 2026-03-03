from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from src.services.file_services import FileServices
from src.services.rag_services import RAGServices
from src.core.logger import logger


router = APIRouter()
file_services = FileServices()
rag_services = RAGServices()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = file_services.save_file(file)
        

        if file_path:
            logger.info(f"File uploaded successfully: {file_path}")
            documents = rag_services.load_file(file_path)
            logger.info(f"Documents loaded successfully: {len(documents)}")
            split_docs = rag_services.split_documents(documents)
            logger.info(f"Documents split successfully: {len(split_docs)}")
            rag_services.embed_documents(split_docs)
            logger.info(f"Documents embedded successfully: {len(split_docs)}")
            return JSONResponse(
                content={
                    "message": "File uploaded successfully",
                    "data": [{"file_path": file_path}],
                    "error": None,
                },
                status_code=status.HTTP_201_CREATED
            )
            

            


        logger.error("Failed to upload file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file",
        )
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
