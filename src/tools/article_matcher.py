import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ArticleMatcher:
    def __init__(self, articles_file_path=None):
        if articles_file_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.articles_file_path = project_root / "config" / "articles.json"
        else:
            self.articles_file_path = Path(articles_file_path)

        self.articles = self.load_articles()

    def load_articles(self):
        try:
            if self.articles_file_path.exists():
                with open(self.articles_file_path, "r", encoding="utf-8") as f:
                    articles_data = json.load(f)

                if isinstance(articles_data, list):
                    articles = articles_data
                elif isinstance(articles_data, dict) and "articles" in articles_data:
                    articles = articles_data["articles"]
                else:
                    articles = list(articles_data.values())

                logger.info(
                    f"Загружено {len(articles)} артикулов из {self.articles_file_path}"
                )
                return set(str(article).strip().upper() for article in articles)
            else:
                logger.warning(
                    f"Файл с артикулами не найден: {self.articles_file_path}"
                )
                return set()
        except Exception as e:
            logger.error(f"Ошибка загрузки артикулов: {e}")
            return set()

    def find_articles_in_text(self, text):
        found_articles = set()

        if not text or not self.articles:
            return list(found_articles)

        text_upper = text.upper()

        for article in self.articles:
            if article and len(article) > 2:
                pattern = r"\b" + re.escape(article) + r"\b"
                if re.search(pattern, text_upper):
                    found_articles.add(article)

        logger.info(f"Найдено артикулов в тексте: {len(found_articles)}")
        return list(found_articles)

    def find_articles_in_email(self, email_data, attachments_text=""):
        all_text = ""

        if "body" in email_data:
            if "plain" in email_data["body"]:
                all_text += email_data["body"]["plain"] + "\n"

        all_text += email_data.get("subject", "") + "\n"
        all_text += attachments_text

        return self.find_articles_in_text(all_text)
