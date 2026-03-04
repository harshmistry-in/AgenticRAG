from fastapi import UploadFile
import os

from src.core.settings import settings
from src.core.logger import logger

from werkzeug.utils import secure_filename

class FileServices:
    def __init__(self):
        logger.info("FileServices initialized successfully")

    def save_file(self, file: UploadFile):
        if not file.filename.lower().endswith(".pdf"):
            logger.error(f"Invalid file type uploaded: {file.filename}")
            raise ValueError("Only PDF files are supported.")

        if not os.path.exists(settings.UPLOAD_DIR):
            logger.info(
                f"Upload directory does not exist. Creating directory: {settings.UPLOAD_DIR}"
            )
            os.makedirs(settings.UPLOAD_DIR)

        safe_filename = secure_filename(file.filename)
        logger.info(f"Saving file: {safe_filename} to {settings.UPLOAD_DIR}")
        file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as f:
            f.write(file.file.read())
        logger.info(f"File saved successfully: {file_path}")

        return file_path
