import json
import logging
import os
import re
import sys

import torch

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from src.email_classifier.model import EmailClassifier

logger = logging.getLogger(__name__)

# ============================== !!! STEM TOGGLE !!! ==============================
stem = True
if stem:
    stemmer = SnowballStemmer("russian")
    stop_words = set(stopwords.words("russian"))


class EmailClassifierPredictor:
    def __init__(
        self,
        model_path="../models/email_classifier.pth",
        vocab_path="../models/vocabulary.json",
    ):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = None
        self.vocab = None
        self.max_length = 200
        self.load_model()

    def load_model(self):
        """Загрузка модели и словаря"""
        try:
            logger.info(f"Попытка загрузки модели из {self.model_path}")
            logger.info(f"Попытка загрузки словаря из {self.vocab_path}")

            # Проверка существования файлов
            if not os.path.exists(self.model_path):
                error_msg = f"Файл модели {self.model_path} не найден"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            if not os.path.exists(self.vocab_path):
                error_msg = f"Файл словаря {self.vocab_path} не найден"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Загрузка словаря
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            logger.info(f"Словарь загружен. Размер: {len(self.vocab)} токенов")

            # Загрузка параметров модели
            checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
            logger.info("Параметры модели загружены")

            # Создание модели
            self.model = EmailClassifier(
                vocab_size=checkpoint["vocab_size"],
                embedding_dim=checkpoint["embedding_dim"],
                hidden_dim=checkpoint["hidden_dim"],
                output_dim=checkpoint["output_dim"],
                n_layers=checkpoint["n_layers"],
                dropout=checkpoint["dropout"],
                bidirectional=checkpoint["bidirectional"],
            )

            # Загрузка весов
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            success_msg = (
                f"LSTM модель успешно загружена! "
                f"Архитектура: {checkpoint['n_layers']} LSTM слоев, "
                f"hidden_dim: {checkpoint['hidden_dim']}, "
                f"bidirectional: {checkpoint['bidirectional']}"
            )
            logger.debug(success_msg)

        except FileNotFoundError as e:
            logger.error(f"Файл не найден: {e}", exc_info=True)
            print(f"ОШИБКА: {e}")
            print("Сначала выполните обучение модели: python train.py")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}", exc_info=True)
            print(f"ОШИБКА при загрузке модели: {e}")
            sys.exit(1)

    def preprocess_text(self, text):
        """Предобработка текста"""
        try:
            text = text.lower()
            text = re.sub(r"\S+@\S+", "", text)
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[^а-яА-ЯёЁ\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if stem:
                pre_tokens = word_tokenize(text, language="russian")
                tokens = [
                    stemmer.stem(token)
                    for token in pre_tokens
                    if token not in stop_words and len(token) > 2
                ]
            else:
                tokens = text.split()

            if len(tokens) < 1:
                logger.warning(
                    f"Текст содержит недостаточно токенов после предобработки: {len(tokens)}"
                )
                return None, None

            logger.debug(f"Текст токенизирован. Получено {len(tokens)} токенов")

            # Преобразование в индексы
            indexed = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
            text_length = min(len(indexed), self.max_length)

            # Паддинг/обрезка
            if len(indexed) > self.max_length:
                indexed = indexed[: self.max_length]
                logger.debug(f"Текст обрезан до {self.max_length} токенов")
            else:
                indexed = indexed + [self.vocab["<PAD>"]] * (
                    self.max_length - len(indexed)
                )
                logger.debug(f"Текст дополнен до {self.max_length} токенов")

            # Подсчет неизвестных токенов
            unknown_count = indexed.count(self.vocab["<UNK>"])
            if unknown_count > 0:
                logger.warning(f"Обнаружено {unknown_count} неизвестных токенов")

            return (
                torch.tensor(indexed, dtype=torch.long).unsqueeze(0),
                torch.tensor(text_length, dtype=torch.long).unsqueeze(0),
            )

        except Exception as e:
            logger.error(f"Ошибка при предобработке текста: {e}", exc_info=True)
            raise

    def predict(self, text):
        """Предсказание для одного текста"""
        if self.model is None:
            error_msg = "Модель не загружена"
            logger.error(error_msg)
            return {"error": error_msg}

        try:
            logger.info(f"Начало предсказания для текста длиной {len(text)} символов")

            # Предобработка
            input_tensor, length_tensor = self.preprocess_text(text)

            if input_tensor is None:
                result = {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "predicted_class": 0,  # Не заявка
                    "class_name": "Не заявка",
                    "confidence": 1.0,  # 100% уверенность
                    "probabilities": {"не_заявка": 1.0, "заявка": 0.0},
                }
                logger.info(
                    f"Предсказание завершено (недостаточно токенов): {result['class_name']} "
                    f"(уверенность: {result['confidence']:.2%})"
                )
                return result

            # Предсказание
            with torch.no_grad():
                output = self.model(input_tensor, length_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = torch.max(probabilities).item()

            # Интерпретация результата
            class_names = {0: "Не заявка", 1: "Заявка"}

            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "predicted_class": predicted_class,
                "class_name": class_names[predicted_class],
                "confidence": round(confidence, 4),
                "probabilities": {
                    "не_заявка": round(probabilities[0][0].item(), 4),
                    "заявка": round(probabilities[0][1].item(), 4),
                },
            }

            logger.info(
                f"Предсказание завершено: {result['class_name']} "
                f"(уверенность: {result['confidence']:.2%}, "
                f"вероятности: {result['probabilities']})"
            )

            return result

        except Exception as e:
            error_msg = f"Ошибка при предсказании: {e}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}


def predict_single_text(text):
    """Функция для предсказания одного текста"""
    return EmailClassifierPredictor().predict(text)
