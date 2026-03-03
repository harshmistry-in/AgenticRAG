from fastapi import UploadFile
import os

from src.core.settings import settings
from src.core.logger import logger

class FileServices:
    def __init__(self):
        pass
    
    def save_file(self, file: UploadFile):
        if not os.path.exists(settings.UPLOAD_DIR):
            logger.info(f"Upload directory does not exist. Creating directory: {settings.UPLOAD_DIR}")
            os.makedirs(settings.UPLOAD_DIR)
        
        logger.info(f"Saving file: {file.filename} to {settings.UPLOAD_DIR}")
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        logger.info(f"File saved successfully: {file_path}")
            
        return file_path