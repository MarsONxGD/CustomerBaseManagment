import email
import imaplib
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.email_config import credentials
from src.tools.ANSIColorFormatter import ANSIColorFormatter


def setup_logging():
    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(exist_ok=True)

    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(
        ANSIColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    log_file = log_dir / "dataset_collector.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging_handler,
        ],
    )

    if __name__ == "__main__":
        output = logging.getLogger("dataset_collector")
    else:
        output = logging.getLogger(__name__)

    return output


logger = setup_logging()


class DatasetCollector:
    def __init__(self):
        self.config = credentials()
        self.mail = None
        self.dataset_path = PROJECT_ROOT / "datasets" / "data.csv"
        self.datasets_dir = PROJECT_ROOT / "datasets"

        self.datasets_dir.mkdir(exist_ok=True)

    @staticmethod
    def extract_email_text(msg):
        text_parts = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if part.get_content_maintype() == "multipart":
                    continue

                filename = part.get_filename()
                if filename:
                    continue

                if (
                    content_type == "text/plain"
                    and "attachment" not in content_disposition.lower()
                ):
                    try:
                        body = part.get_payload(decode=True)
                        if body:
                            text_parts.append(body.decode("utf-8", errors="ignore"))
                    except Exception as e:
                        logger.warning(f"Ошибка декодирования plain текста: {e}")
        else:
            content_type = msg.get_content_type()
            try:
                body = msg.get_payload(decode=True)
                if body and content_type == "text/plain":
                    text_parts.append(body.decode("utf-8", errors="ignore"))
            except Exception as e:
                logger.warning(f"Ошибка декодирования тела письма: {e}")

        return "\n".join(text_parts)

    @staticmethod
    def cleanup_text(text):
        text = text.lower()
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^а-яА-ЯёЁ\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if stem:
            tokens = word_tokenize(text, language="russian")
            filtered_tokens = [
                stemmer.stem(token)
                for token in tokens
                if token not in stop_words and len(token) > 2
            ]
            return " ".join(filtered_tokens)
        return text

    def connect(self):
        try:
            self.mail = imaplib.IMAP4_SSL(
                self.config["imap_server"], self.config["imap_port"]
            )
            self.mail.login(self.config["email"], self.config["password"])
            logger.info("Успешное подключение к почтовому серверу")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            return False

    def disconnect(self):
        if self.mail:
            try:
                self.mail.logout()
                logger.info("Отключение от почтового сервера")
            except:
                pass

    def get_emails_from_folder(self, folder, label):
        emails_data = []

        try:
            self.mail.select(folder)
            status, messages = self.mail.uid("SEARCH", None, "ALL")

            if status != "OK":
                logger.warning(f"Не удалось получить письма из папки {folder}")
                return emails_data

            email_uids = messages[0].split()
            logger.info(f"Найдено {len(email_uids)} писем в папке {folder}")

            for email_uid in email_uids[:100]:
                try:
                    status, msg_data = self.mail.uid("FETCH", email_uid, "(RFC822)")
                    if status != "OK" or not msg_data:
                        continue

                    for item in msg_data:
                        if isinstance(item, tuple) and len(item) == 2:
                            if isinstance(item[1], bytes):
                                try:
                                    msg = email.message_from_bytes(item[1])

                                    email_text = self.extract_email_text(msg)

                                    if email_text.strip():
                                        emails_data.append(
                                            {
                                                "text": self.cleanup_text(email_text),
                                                "label": label,
                                            }
                                        )
                                        logger.info(
                                            f"Добавлено письмо UID {email_uid.decode()} из {folder} с label={label}"
                                        )

                                except Exception as e:
                                    logger.warning(
                                        f"Ошибка обработки письма UID {email_uid.decode()}: {e}"
                                    )

                except Exception as e:
                    logger.warning(
                        f"Ошибка получения письма UID {email_uid.decode()}: {e}"
                    )

        except Exception as e:
            logger.error(f"Ошибка работы с папкой {folder}: {e}")

        return emails_data

    def backup_existing_dataset(self):
        if not self.dataset_path.exists():
            return

        existing_backups = []
        pattern = re.compile(r"data_old_(\d+)")

        for file in self.datasets_dir.iterdir():
            match = pattern.match(file.name)
            if match:
                existing_backups.append(int(match.group(1)))

        next_number = max(existing_backups) + 1 if existing_backups else 1

        backup_name = f"data_old_{next_number}.csv"
        backup_path = self.datasets_dir / backup_name

        self.dataset_path.rename(backup_path)
        logger.info(f"Создана резервная копия: {backup_name}")

    def collect_dataset(self):
        if not self.connect():
            return False

        try:
            true_emails = self.get_emails_from_folder("DATASET/TRUE", 1)
            false_emails = self.get_emails_from_folder("DATASET/FALSE", 0)

            logger.info(
                f"Собрано писем: TRUE={len(true_emails)}, FALSE={len(false_emails)}"
            )

            if not true_emails and not false_emails:
                logger.info("Новых писем для добавления в датасет не найдено")
                return True

            existing_data = []
            if self.dataset_path.exists():
                try:
                    existing_df = pd.read_csv(self.dataset_path)
                    existing_data = existing_df.to_dict("records")
                    logger.info(
                        f"Загружен существующий датасет: {len(existing_data)} записей"
                    )
                except Exception as e:
                    logger.error(f"Ошибка загрузки существующего датасета: {e}")
                    existing_data = []

            all_emails = true_emails + false_emails
            combined_data = existing_data + all_emails

            df = pd.DataFrame(combined_data)

            df = df.drop_duplicates(subset=["text"], keep="first")
            logger.info(f"После удаления дубликатов: {len(df)} записей")

            if self.dataset_path.exists():
                self.backup_existing_dataset()

            df.to_csv(self.dataset_path, index=False, encoding="utf-8-sig")
            logger.info(f"Датасет сохранен: {self.dataset_path} ({len(df)} записей)")

            return True

        except Exception as e:
            logger.error(f"Ошибка сбора датасета: {e}")
            return False
        finally:
            self.disconnect()


def main():
    logger.info("Запуск сбора датасета из писем")

    collector = DatasetCollector()
    success = collector.collect_dataset()

    if success:
        logger.info("Сбор датасета завершен успешно")
    else:
        logger.error("Сбор датасета завершен с ошибками")


if __name__ == "__main__":
    stem = False
    if stem:
        stemmer = SnowballStemmer("russian")
        stop_words = set(stopwords.words("russian"))
    main()
