import imaplib
import email
import json
import os
import logging
from email.header import decode_header
from datetime import datetime
import uuid
import time
import config.email_config

# Настройка логирования
os.makedirs("../log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../log/email_handler.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def decode_mime_words(text):
    """Декодирование MIME заголовков"""
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


class EmailHandler:
    def __init__(self):
        self.config = config.email_config.credentials()
        self.mail = None
        self.data_dir = "../data/email"
        self.attachments_dir = "../data/email/attachments"
        self.processed_dir = "PROCESSED"

        # Настройки обработки
        self.max_emails_per_run = 5
        self.delay_between_emails = 2
        self.max_retries = 3

        # Создаем необходимые директории
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.attachments_dir, exist_ok=True)

    def connect(self):
        """Подключение к IMAP серверу"""
        try:
            self.mail = imaplib.IMAP4_SSL(
                self.config["imap_server"], self.config["imap_port"]
            )
            self.mail.socket().settimeout(60)
            self.mail.login(self.config["email"], self.config["password"])
            logger.info("Успешное подключение к почтовому серверу")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            return False

    def disconnect(self):
        """Отключение от сервера"""
        if self.mail:
            try:
                try:
                    # Пытаемся выбрать INBOX, чтобы перевести соединение в состояние SELECTED
                    self.mail.select("INBOX")
                except:
                    pass

                # Закрываем текущий mailbox
                try:
                    self.mail.close()
                except Exception as e:
                    logger.debug(f"Команда close не выполнена: {e}")

                # Выходим
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
        """Безопасное подключение с повторными попытками"""
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
        """Сохранение вложений"""
        filename = part.get_filename()
        if filename:
            filename = decode_mime_words(filename)
            if not filename:
                filename = f"attachment_{uuid.uuid4().hex}"

            # Очистка имени файла
            filename = "".join(
                c for c in filename if c.isalnum() or c in (" ", "-", "_", ".")
            ).rstrip()

            # Добавляем номер вложения к имени файла для уникальности
            filepath = os.path.join(
                self.attachments_dir, f"{email_id}_{attachment_number}_{filename}"
            )

            try:
                # Получаем содержимое вложения
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
        """Парсинг письма и сохранение в JSON"""
        email_data = {
            "id": email_id,
            "subject": decode_mime_words(msg.get("Subject")),
            "from": decode_mime_words(msg.get("From")),
            "to": decode_mime_words(msg.get("To")),
            "date": msg.get("Date"),
            "processed_date": datetime.now().isoformat(),
            "body": {},
            "attachments": [],
        }

        attachment_counter = 0

        # Обработка частей письма
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Пропускаем multipart контейнеры
                if part.get_content_maintype() == "multipart":
                    continue

                # Вложения - определяем по наличию filename
                filename = part.get_filename()
                if filename:
                    attachment_counter += 1
                    logger.info(f"Найдено вложение {attachment_counter}: {filename}")
                    attachment_info = self.save_attachment(
                        part, email_id, attachment_counter
                    )
                    if attachment_info:
                        email_data["attachments"].append(attachment_info)
                    # Продолжаем обработку для других частей
                    continue

                # Текст письма (обрабатываем только если это не вложение)
                if (
                        content_type == "text/plain"
                        and "attachment" not in content_disposition.lower()
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

                elif (
                        content_type == "text/html"
                        and "attachment" not in content_disposition.lower()
                ):
                    try:
                        body = part.get_payload(decode=True)
                        if body and "html" not in email_data["body"]:
                            email_data["body"]["html"] = body.decode(
                                "utf-8", errors="ignore"
                            )
                            logger.debug("Сохранен HTML текст")
                    except Exception as e:
                        logger.warning(f"Ошибка декодирования HTML: {e}")
        else:
            # Простое письмо (не multipart)
            content_type = msg.get_content_type()
            try:
                body = msg.get_payload(decode=True)
                if body:
                    if content_type == "text/plain":
                        email_data["body"]["plain"] = body.decode(
                            "utf-8", errors="ignore"
                        )
                    elif content_type == "text/html":
                        email_data["body"]["html"] = body.decode(
                            "utf-8", errors="ignore"
                        )
            except Exception as e:
                logger.warning(f"Ошибка декодирования тела письма: {e}")

        # Сохранение JSON
        json_filename = f"{email_id}.json"
        json_path = os.path.join(self.data_dir, json_filename)

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

    def mark_as_processed(self, email_id):
        """Перемещение письма в папку PROCESSED"""
        try:
            try:
                self.mail.create(self.processed_dir)
                logger.info(f"Создана папка {self.processed_dir}")
            except:
                pass

            result = self.mail.copy(email_id, self.processed_dir)
            if result[0] == "OK":
                self.mail.store(email_id, "+FLAGS", "\\Deleted")
                self.mail.expunge()
                logger.info(f"Письмо {email_id} перемещено в PROCESSED")
                return True
            else:
                logger.warning(f"Не удалось переместить письмо {email_id}: {result}")
                return False
        except Exception as e:
            logger.error(f"Ошибка при перемещении письма {email_id}: {e}")
            return False

    def get_email_with_retry(self, email_id):
        """Получение письма с повторными попытками"""
        for attempt in range(self.max_retries):
            try:
                if attempt % 2 == 0:
                    status, msg_data = self.mail.fetch(email_id, "(RFC822)")
                else:
                    status, msg_data = self.mail.fetch(email_id, "(BODY.PEEK[])")

                if status != "OK" or not msg_data:
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
                                    f"Письмо {email_id} успешно получено (попытка {attempt + 1})"
                                )
                                return msg
                            except Exception as e:
                                logger.warning(
                                    f"Ошибка парсинга письма {email_id} (попытка {attempt + 1}): {e}"
                                )

                    elif isinstance(item, bytes) and len(item) > 100:
                        try:
                            msg = email.message_from_bytes(item)
                            logger.info(
                                f"Письмо {email_id} успешно получено (попытка {attempt + 1})"
                            )
                            return msg
                        except Exception as e:
                            logger.warning(
                                f"Ошибка парсинга письма {email_id} (попытка {attempt + 1}): {e}"
                            )

                if attempt < self.max_retries - 1:
                    time.sleep(1)

            except Exception as e:
                logger.warning(
                    f"Ошибка получения письма {email_id} (попытка {attempt + 1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2)

        logger.error(
            f"Не удалось получить письмо {email_id} после {self.max_retries} попыток"
        )
        return None

    def process_single_email(self, email_id):
        """Обработка одного письма"""
        email_id_str = email_id.decode()
        try:
            msg = self.get_email_with_retry(email_id)

            if msg is None:
                logger.error(f"Не удалось получить письмо {email_id_str}")
                return False

            unique_id = f"{uuid.uuid4().hex}_{email_id_str}"

            if self.parse_email(msg, unique_id):
                if self.mark_as_processed(email_id):
                    logger.info(f"Успешно обработано письмо: {unique_id}")
                    return True
                else:
                    logger.error(f"Не удалось переместить письмо {email_id_str}")
                    return False
            else:
                logger.error(f"Не удалось сохранить данные письма {email_id_str}")
                return False

        except Exception as e:
            logger.error(f"Критическая ошибка обработки письма {email_id_str}: {e}")
            return False

    def get_unread_count(self):
        """Получить количество непрочитанных писем"""
        try:
            if not self.mail:
                if not self.safe_connect():
                    return 0

            self.mail.select("INBOX")
            status, messages = self.mail.search(None, "UNSEEN")

            if status == "OK":
                return len(messages[0].split())
            else:
                return 0
        except Exception as e:
            logger.error(f"Ошибка получения количества непрочитанных писем: {e}")
            return 0

    def process_emails(self, force_mode=False):
        """Основной метод обработки писем"""
        if not self.safe_connect():
            logger.error("Не удалось подключиться к серверу")
            return

        try:
            self.mail.select("INBOX")
            status, messages = self.mail.search(None, "UNSEEN")

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

            for i, email_id in enumerate(email_ids):
                email_id_str = email_id.decode()
                logger.info(
                    f"Обработка письма {i + 1}/{len(email_ids)}: {email_id_str}"
                )

                if self.process_single_email(email_id):
                    processed_count += 1
                else:
                    error_count += 1

                if i < len(email_ids) - 1:
                    logger.info(
                        f"Пауза {self.delay_between_emails} секунд перед следующим письмом..."
                    )
                    time.sleep(self.delay_between_emails)

            logger.info(
                f"Обработка завершена: {processed_count} успешно, {error_count} с ошибками"
            )

            # Получаем оставшееся количество непрочитанных писем
            remaining = self.get_unread_count()

            print(
                f"РЕЗУЛЬТАТ: Обработано {processed_count} писем, ошибок: {error_count}"
            )
            print(f"ОСТАЛОСЬ: {remaining} непрочитанных писем")

            if remaining > 0:
                logger.info(f"Осталось непрочитанных писем: {remaining}")

            # Если включен форсированный режим и есть ошибки или остались письма - повторяем
            if force_mode and (error_count > 0 or remaining > 0):
                logger.info("Форсированный режим: повторная попытка обработки...")
                time.sleep(5)
                self.process_emails(force_mode=True)

        except Exception as e:
            logger.error(f"Ошибка в процессе обработки: {e}")
            # В форсированном режиме переподключаемся и пробуем снова
            if force_mode:
                logger.info("Форсированный режим: повторная попытка после ошибки...")
                time.sleep(10)
                self.disconnect()
                self.process_emails(force_mode=True)
        finally:
            self.disconnect()


def main(force_mode=False):
    """Основная функция"""
    logger.info(
        "Запуск обработки писем" + (" в форсированном режиме" if force_mode else "")
    )

    handler = EmailHandler()
    handler.process_emails(force_mode=force_mode)

    logger.info("Обработка писем завершена")


if __name__ == "__main__":
    import sys

    force_mode = len(sys.argv) > 1 and sys.argv[1] == "--force"
    main(force_mode=force_mode)
