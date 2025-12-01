import email
import imaplib
import json
import logging
import os
import sys
import time
import uuid
from email.header import decode_header
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.email_config import credentials
from src.email_classifier.predict import predict_single_text
from src.tools.ANSIColorFormatter import ANSIColorFormatter
from src.tools.article_matcher import ArticleMatcher
from src.tools.attachment_processor import AttachmentProcessor


def setup_logging():
    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        ANSIColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    console_handler.setLevel(logging.INFO)

    log_file = log_dir / "softwarecbm.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s : %(name)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            console_handler,
            file_handler,
        ],
    )

    if __name__ == "__main__":
        output = logging.getLogger("email_handler")
    else:
        output = logging.getLogger(__name__)

    return output


logger = setup_logging()


class EmailHandler:
    def __init__(self):
        self.config = credentials()
        self.mail = None
        self.processed_dir = "PROCESSED"
        self.spam_dir = "PROCESSED/SPAM"
        self.correct_dir = "PROCESSED/CORRECT"
        self.incorrect_dir = "PROCESSED/INCORRECT"

        self.data_dir = PROJECT_ROOT / "temp" / "email"
        self.attachments_dir = PROJECT_ROOT / "temp" / "email" / "attachments"
        self.results_dir = PROJECT_ROOT / "temp" / "results"

        self.attachment_processor = AttachmentProcessor()
        self.article_matcher = ArticleMatcher()

        self.max_emails_per_run = 50
        self.delay_between_emails = 0
        self.delay_reconnect = 5
        self.socket_timeout = 120
        self.max_retries = 2

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def decode_mime_words(text):
        if text is None:
            return ""
        decoded_parts = decode_header(text)
        decoded_text = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                if encoding:
                    decoded_text += part.decode(encoding)
                else:
                    decoded_text += part.decode("utf-8", errors="ignore")
            else:
                decoded_text += part
        return decoded_text

    @staticmethod
    def classify_email_basic(email_data):
        try:
            text_parts = []
            if "body" in email_data:
                if "plain" in email_data["body"]:
                    text_parts.append(email_data["body"]["plain"])

            email_text = "\n".join(text_parts)
            email_text += f"\n{email_data.get('subject', '')}"

            prediction = predict_single_text(email_text)

            logger.info(
                f"Классификация письма {email_data['id']}: {prediction['class_name']} (уверенность: {prediction['confidence']:.2%})"
            )

            return (
                prediction["class_name"] == "Заявка",
                prediction["confidence"],
                email_text,
            )

        except Exception as e:
            logger.error(
                f"Ошибка при классификации письма {email_data.get('id', 'unknown')}: {e}"
            )
            return False, 0.0, ""

    def create_processed_folders(self):
        try:
            folders = [
                self.processed_dir,
                self.spam_dir,
                self.correct_dir,
                self.incorrect_dir,
            ]
            for folder in folders:
                try:
                    self.mail.create(folder)
                    logger.debug(f"Создана папка {folder}")
                except Exception as e:
                    logger.debug(
                        f"Папка {folder} уже существует или не может быть создана: {e}"
                    )
        except Exception as e:
            logger.error(f"Ошибка при создании папок: {e}")

    def extract_text_from_attachments(self, attachments):
        combined_text = ""

        for attachment in attachments:
            attachment_path = self.attachments_dir / attachment["saved_name"]
            if attachment_path.exists():
                logger.info(f"Обработка вложения: {attachment['filename']}")
                extracted_text = self.attachment_processor.extract_text_from_file(
                    str(attachment_path)
                )
                if extracted_text:
                    combined_text += extracted_text + "\n"

        return combined_text

    def delete_email_files(self, email_id, attachments):
        try:
            json_path = self.data_dir / f"{email_id}.json"
            if json_path.exists():
                json_path.unlink()
                logger.info(f"Удален JSON файл: {json_path}")

            for attachment in attachments:
                attachment_path = self.attachments_dir / attachment["saved_name"]
                if attachment_path.exists():
                    attachment_path.unlink()
                    logger.info(f"Удалено вложение: {attachment_path}")

            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении файлов письма {email_id}: {e}")
            return False

    def save_application_result(self, email_data, found_articles):
        try:
            if not found_articles:
                logger.info(
                    f"Для письма {email_data['id']} не найдены артикулы - результат не сохраняется"
                )
                return False

            result = {
                "email": email_data.get("from", ""),
                "date": email_data.get("date", ""),
                "subject": email_data.get("subject", ""),
                "articles_count": len(found_articles),
                "found_articles": found_articles,
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "email_id": email_data.get("id", ""),
            }

            result_filename = f"application_{email_data['id']}.json"
            result_path = self.results_dir / result_filename

            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            csv_path = self.results_dir / "applications.csv"
            csv_header = not csv_path.exists()

            with open(csv_path, "a", encoding="utf-8") as f:
                if csv_header:
                    f.write(
                        "Email,Date,Subject,ArticlesCount,FoundArticles,ProcessedDate\n"
                    )

                articles_str = ";".join(found_articles)
                f.write(
                    f"\"{result['email']}\",\"{result['date']}\",\"{result['subject']}\",{result['articles_count']},\"{articles_str}\",\"{result['processed_date']}\"\n"
                )

            logger.info(
                f"Сохранена информация о заявке: {result['email']} (найдено артикулов: {len(found_articles)})"
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка при сохранении результата: {e}")
            return False

    def connect(self):
        try:
            self.mail = imaplib.IMAP4_SSL(
                self.config["imap_server"], self.config["imap_port"]
            )
            self.mail.socket().settimeout(self.socket_timeout)
            self.mail.login(self.config["email"], self.config["password"])
            logger.info("Успешное подключение к почтовому серверу")

            self.create_processed_folders()

            return True
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            return False

    def disconnect(self):
        if self.mail:
            try:
                try:
                    self.mail.select("INBOX")
                except Exception as e:
                    logger.debug(f"Команда select не выполнена: {e}")

                try:
                    self.mail.close()
                except Exception as e:
                    logger.debug(f"Команда close не выполнена: {e}")

                try:
                    self.mail.logout()
                    logger.info("Успешное отключение от почтового сервера")
                except Exception as e:
                    logger.debug(f"Команда logout не выполнена: {e}")

            except Exception as e:
                logger.warning(f"Ошибка при отключении: {e}")
            finally:
                self.mail = None

    def safe_connect(self):
        for attempt in range(3):
            try:
                if self.connect():
                    return True
                else:
                    logger.warning(f"Попытка подключения {attempt + 1} не удалась")
                    if attempt < 2:
                        time.sleep(5)
            except Exception as e:
                logger.error(f"Ошибка при попытке подключения {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(5)
        return False

    def save_attachment(self, part, email_id, attachment_number):
        filename = part.get_filename()
        if filename:
            filename = self.decode_mime_words(filename)
            if not filename:
                filename = f"attachment_{uuid.uuid4().hex}"

            filename = "".join(
                c for c in filename if c.isalnum() or c in (" ", "-", "_", ".")
            ).rstrip()
            filepath = (
                self.attachments_dir / f"{email_id}_{attachment_number}_{filename}"
            )

            try:
                payload = part.get_payload(decode=True)
                if payload:
                    with open(filepath, "wb") as f:
                        f.write(payload)
                    logger.info(
                        f"Сохранено вложение {attachment_number}: {filename} ({len(payload)} байт)"
                    )
                    return {
                        "filename": filename,
                        "saved_name": f"{email_id}_{attachment_number}_{filename}",
                        "size": len(payload),
                        "content_type": part.get_content_type(),
                    }
                else:
                    logger.warning(f"Пустое вложение {attachment_number}: {filename}")
                    return None
            except Exception as e:
                logger.error(
                    f"Ошибка сохранения вложения {attachment_number} {filename}: {e}"
                )
                return None
        return None

    def parse_email(self, msg, email_id):
        email_data = {
            "id": email_id,
            "subject": self.decode_mime_words(msg.get("Subject")),
            "from": self.decode_mime_words(msg.get("From")),
            "date": msg.get("Date"),
            "body": {},
            "attachments": [],
        }

        attachment_counter = 0

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if part.get_content_maintype() == "multipart":
                    continue

                filename = part.get_filename()
                if filename:
                    attachment_counter += 1
                    logger.info(f"Найдено вложение {attachment_counter}: {filename}")
                    attachment_info = self.save_attachment(
                        part, email_id, attachment_counter
                    )
                    if attachment_info:
                        email_data["attachments"].append(attachment_info)
                    continue

                if (content_type == "text/plain") and (
                    "attachment" not in content_disposition.lower()
                ):
                    try:
                        body = part.get_payload(decode=True)
                        if body and "plain" not in email_data["body"]:
                            email_data["body"]["plain"] = body.decode(
                                "utf-8", errors="ignore"
                            )
                            logger.debug("Сохранен plain текст")
                    except Exception as e:
                        logger.warning(f"Ошибка декодирования plain текста: {e}")

        else:
            content_type = msg.get_content_type()
            try:
                body = msg.get_payload(decode=True)
                if body:
                    if content_type == "text/plain":
                        email_data["body"]["plain"] = body.decode(
                            "utf-8", errors="ignore"
                        )
            except Exception as e:
                logger.warning(f"Ошибка декодирования тела письма: {e}")

        json_filename = f"{email_id}.json"
        json_path = self.data_dir / json_filename

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(email_data, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Сохранен JSON файл: {json_filename} с {len(email_data['attachments'])} вложениями"
            )
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения JSON: {e}")
            return False

    def mark_as_processed(self, email_id, folder):
        try:
            result = self.mail.uid("COPY", email_id, folder)
            if result[0] == "OK":
                self.mail.uid("STORE", email_id, "+FLAGS", "\\Deleted")
                self.mail.expunge()
                logger.info(f"Письмо UID {email_id} перемещено в {folder}")
                return True
            else:
                logger.warning(
                    f"Не удалось переместить письмо UID {email_id} в {folder}: {result}"
                )
                return False
        except Exception as e:
            logger.error(
                f"Ошибка при перемещении письма UID {email_id} в {folder}: {e}"
            )
            return False

    def get_email_with_retry(self, email_id):
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    logger.warning(
                        f"Попытка {attempt + 1} для письма {email_id}, переподключение..."
                    )
                    self.disconnect()
                    if self.delay_reconnect > 0:
                        logger.warning(
                            f"Пауза {self.delay_reconnect} перед переподключением..."
                        )
                        time.sleep(self.delay_reconnect)
                    if not self.safe_connect():
                        continue
                    self.mail.select("INBOX")

                if attempt < self.max_retries - 1:
                    status, msg_data = self.mail.uid("FETCH", email_id, "(BODY.PEEK[])")
                else:
                    status, msg_data = self.mail.uid("FETCH", email_id, "(RFC822)")

                if status != "OK":
                    logger.warning(
                        f"Попытка {attempt + 1} для {email_id}: статус {status}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                    continue

                if not msg_data or msg_data == [None]:
                    logger.warning(
                        f"Попытка {attempt + 1} для {email_id}: пустой ответ"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                    continue

                for item in msg_data:
                    if isinstance(item, tuple) and len(item) == 2:
                        if isinstance(item[1], bytes) and len(item[1]) > 100:
                            try:
                                msg = email.message_from_bytes(item[1])
                                logger.info(
                                    f"Письмо UID {email_id} успешно получено (попытка {attempt + 1})"
                                )
                                return msg
                            except Exception as e:
                                logger.warning(
                                    f"Ошибка парсинга письма UID {email_id} (попытка {attempt + 1}): {e}"
                                )

                    elif isinstance(item, bytes) and len(item) > 100:
                        try:
                            msg = email.message_from_bytes(item)
                            logger.info(
                                f"Письмо UID {email_id} успешно получено (попытка {attempt + 1})"
                            )
                            return msg
                        except Exception as e:
                            logger.warning(
                                f"Ошибка парсинга письма UID {email_id} (попытка {attempt + 1}): {e}"
                            )

                if attempt < self.max_retries - 1:
                    time.sleep(1)

            except Exception as e:
                logger.warning(
                    f"Ошибка получения письма UID {email_id} (попытка {attempt + 1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2)

        logger.error(
            f"Не удалось получить письмо UID {email_id} после {self.max_retries} попыток"
        )
        return None

    def process_single_email(self, email_id):
        email_id_str = email_id.decode()
        unique_id = f"{uuid.uuid4().hex}_{email_id_str}"

        try:
            msg = self.get_email_with_retry(email_id)
            if msg is None:
                logger.error(f"Не удалось получить письмо {email_id_str}")
                return False

            if not self.parse_email(msg, unique_id):
                logger.error(f"Не удалось сохранить данные письма {email_id_str}")
                return False

            json_path = os.path.join(self.data_dir, f"{unique_id}.json")
            with open(json_path, "r", encoding="utf-8") as f:
                email_data = json.load(f)

            is_application, confidence, basic_text = self.classify_email_basic(
                email_data
            )

            # is_application = True
            if not is_application:
                if self.mark_as_processed(email_id, self.spam_dir):
                    self.delete_email_files(
                        unique_id, email_data.get("attachments", [])
                    )
                    logger.info(
                        f"Письмо {unique_id} классифицировано как НЕ заявка и перемещено в SPAM"
                    )
                else:
                    logger.error(f"Не удалось переместить письмо {unique_id} в SPAM")
            else:
                attachments_text = ""
                if email_data.get("attachments"):
                    attachments_text = self.extract_text_from_attachments(
                        email_data["attachments"]
                    )

                found_articles = self.article_matcher.find_articles_in_email(
                    email_data, attachments_text
                )

                if found_articles:
                    self.save_application_result(email_data, found_articles)
                    target_folder = self.correct_dir
                    logger.info(
                        f"Письмо {unique_id} классифицировано как ЗАЯВКА с артикулами (уверенность: {confidence:.2%}, найдено артикулов: {len(found_articles)})"
                    )
                else:
                    target_folder = self.incorrect_dir
                    logger.info(
                        f"Письмо {unique_id} классифицировано как ЗАЯВКА без артикулов (уверенность: {confidence:.2%})"
                    )

                if self.mark_as_processed(email_id, target_folder):
                    logger.info(f"Письмо {unique_id} перемещено в {target_folder}")
                else:
                    logger.error(
                        f"Не удалось переместить письмо {unique_id} в {target_folder}"
                    )

            return True

        except Exception as e:
            logger.error(f"Критическая ошибка обработки письма {email_id_str}: {e}")
            return False

    def get_unread_count(self):
        try:
            if not self.mail:
                if not self.safe_connect():
                    return 0

            self.mail.select("INBOX")
            status, messages = self.mail.uid("SEARCH", None, "UNSEEN")

            if status == "OK":
                return len(messages[0].split())
            else:
                return 0
        except Exception as e:
            logger.error(f"Ошибка получения количества непрочитанных писем: {e}")
            return 0

    def process_emails(self, force_mode=False):
        if not self.safe_connect():
            logger.error("Не удалось подключиться к серверу")
            return

        try:
            self.mail.select("INBOX")
            status, messages = self.mail.uid("SEARCH", None, "UNSEEN")

            if status != "OK":
                logger.error("Ошибка поиска писем")
                return

            email_ids = messages[0].split()

            if not force_mode and len(email_ids) > self.max_emails_per_run:
                logger.info(
                    f"Найдено {len(email_ids)} писем, ограничиваем обработку до {self.max_emails_per_run}"
                )
                email_ids = email_ids[: self.max_emails_per_run]
            else:
                logger.info(f"Найдено {len(email_ids)} новых писем")

            processed_count = 0
            error_count = 0
            spam_count = 0
            correct_count = 0
            incorrect_count = 0

            for i, email_id in enumerate(email_ids):
                email_id_str = email_id.decode()
                logger.info(
                    f"Обработка письма {i + 1}/{len(email_ids)}: UID {email_id_str}"
                )

                if self.process_single_email(email_id):
                    processed_count += 1
                else:
                    error_count += 1

                if (self.delay_between_emails > 0) and (i < len(email_ids) - 1):
                    logger.info(
                        f"Пауза {self.delay_between_emails} секунд перед следующим письмом..."
                    )
                    time.sleep(self.delay_between_emails)

            try:
                self.mail.select(self.correct_dir)
                status, correct_messages = self.mail.uid("SEARCH", None, "ALL")
                correct_count = (
                    len(correct_messages[0].split()) if status == "OK" else 0
                )

                self.mail.select(self.incorrect_dir)
                status, incorrect_messages = self.mail.uid("SEARCH", None, "ALL")
                incorrect_count = (
                    len(incorrect_messages[0].split()) if status == "OK" else 0
                )

                self.mail.select(self.spam_dir)
                status, spam_messages = self.mail.uid("SEARCH", None, "ALL")
                spam_count = len(spam_messages[0].split()) if status == "OK" else 0

            except Exception as e:
                logger.warning(f"Не удалось получить статистику по папкам: {e}")

            logger.info(
                f"Обработка завершена: {processed_count} успешно, {error_count} с ошибками"
            )
            logger.info(
                f"Статистика: CORRECT={correct_count}, INCORRECT={incorrect_count}, SPAM={spam_count}"
            )

            remaining = self.get_unread_count()

            print(
                f"РЕЗУЛЬТАТ: Обработано {processed_count} писем, ошибок: {error_count}"
            )
            print(
                f"Статистика: CORRECT={correct_count}, INCORRECT={incorrect_count}, SPAM={spam_count}"
            )
            print(f"ОСТАЛОСЬ: {remaining} непрочитанных писем")

            if remaining > 0:
                logger.info(f"Осталось непрочитанных писем: {remaining}")

            if force_mode and (error_count > 0 or remaining > 0):
                logger.info("Форсированный режим: повторная попытка обработки...")
                time.sleep(5)
                self.process_emails(force_mode=True)

        except Exception as e:
            logger.error(f"Ошибка в процессе обработки: {e}")
            if force_mode:
                logger.info("Форсированный режим: повторная попытка после ошибки...")
                time.sleep(10)
                self.disconnect()
                self.process_emails(force_mode=True)
        finally:
            self.disconnect()


def main(force_mode=False):
    logger.info(
        "Запуск обработки писем" + (" в форсированном режиме" if force_mode else "")
    )

    email_handler = EmailHandler()
    email_handler.process_emails(force_mode=force_mode)

    logger.info("Обработка писем завершена")


if __name__ == "__main__":
    force_mode = len(sys.argv) > 1 and sys.argv[1] == "--force"
    main(force_mode=force_mode)
