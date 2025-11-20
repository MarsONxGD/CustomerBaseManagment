import logging
import os

import pandas as pd
import PyPDF2
import pytesseract
from docx import Document
from PIL import Image

logger = logging.getLogger(__name__)


class AttachmentProcessor:
    def __init__(self):
        self.supported_image_extensions = [".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    def extract_text_from_file(self, file_path):
        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext in self.supported_image_extensions:
                return self._extract_text_from_image(file_path)
            elif file_ext == ".pdf":
                return self._extract_text_from_pdf(file_path)
            elif file_ext in [".doc", ".docx"]:
                return self._extract_text_from_word(file_path)
            elif file_ext in [".xls", ".xlsx"]:
                return self._extract_text_from_excel(file_path)
            else:
                logger.warning(f"Неподдерживаемый формат файла: {file_ext}")
                return ""

        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из {file_path}: {e}")
            return ""

    @staticmethod
    def _extract_text_from_image(image_path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang="eng")
            logger.info(f"Извлечен текст из изображения: {len(text)} символов")
            # logger.info(f"Извлечен текст из изображения: {text}")
            return text
        except Exception as e:
            logger.error(f"Ошибка OCR для {image_path}: {e}")
            return ""

    @staticmethod
    def _extract_text_from_pdf(pdf_path):
        try:
            text = ""
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            logger.info(f"Извлечен текст из PDF: {len(text)} символов")
            return text
        except Exception as e:
            logger.error(f"Ошибка чтения PDF {pdf_path}: {e}")
            return ""

    @staticmethod
    def _extract_text_from_word(doc_path):
        try:
            doc = Document(doc_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.info(f"Извлечен текст из Word: {len(text)} символов")
            return text
        except Exception as e:
            logger.error(f"Ошибка чтения Word {doc_path}: {e}")
            return ""

    @staticmethod
    def _extract_text_from_excel(excel_path):
        try:
            text = ""
            excel_file = pd.ExcelFile(excel_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                text += f"Лист: {sheet_name}\n"
                text += df.to_string() + "\n\n"
            logger.info(f"Извлечен текст из Excel: {len(text)} символов")
            return text
        except Exception as e:
            logger.error(f"Ошибка чтения Excel {excel_path}: {e}")
            return ""
