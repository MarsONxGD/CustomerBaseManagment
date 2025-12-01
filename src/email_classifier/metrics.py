import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.email_classifier.model import EmailClassifier

stem = True
if stem:
    stemmer = SnowballStemmer("russian")
    stop_words = set(stopwords.words("russian"))


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


logger = setup_logging()


def load_model_and_vocab(model_path, vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    model = EmailClassifier(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint["dropout"],
        bidirectional=checkpoint["bidirectional"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab


def preprocess_text(text, vocab, max_length=200):
    """Точная копия функции предобработки из predict.py"""
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
            logger.warning(f"Текст содержит недостаточно токенов после предобработки: {len(tokens)}")
            return None, None

        indexed = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        text_length = min(len(indexed), max_length)

        if len(indexed) > max_length:
            indexed = indexed[:max_length]
        else:
            indexed = indexed + [vocab["<PAD>"]] * (max_length - len(indexed))

        return (
            torch.tensor(indexed, dtype=torch.long).unsqueeze(0),
            torch.tensor(text_length, dtype=torch.long).unsqueeze(0),
        )

    except Exception as e:
        logger.error(f"Ошибка при предобработке текста: {e}")
        return None, None


def calculate_metrics():
    model_path = PROJECT_ROOT / "models" / "email_classifier.pth"
    vocab_path = PROJECT_ROOT / "models" / "vocabulary.json"
    test_data_path = PROJECT_ROOT / "datasets" / "test_data.csv"

    model, vocab = load_model_and_vocab(model_path, vocab_path)
    print("✅ Модель и словарь загружены")

    try:
        df = pd.read_csv(test_data_path)
        texts = df["text"].tolist()
        true_labels = df["label"].tolist()
        print(f"✅ Загружено {len(texts)} тестовых примеров")

        class_counts = df['label'].value_counts()
        print(f"📊 Распределение классов в тестовых данных:")
        print(f"   Класс 0 (Не заявка): {class_counts.get(0, 0)} примеров")
        print(f"   Класс 1 (Заявка): {class_counts.get(1, 0)} примеров")

    except FileNotFoundError:
        print(f"❌ ОШИБКА: Файл {test_data_path} не найден!")
        return
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return

    predicted_labels = []
    predicted_probs = []
    processed_count = 0
    failed_count = 0

    print("\n🔮 Выполняю предсказания...")

    with torch.no_grad():
        for i, (text, true_label) in enumerate(zip(texts, true_labels)):
            input_tensor, length_tensor = preprocess_text(text, vocab)

            if input_tensor is None:
                predicted_labels.append(0)
                predicted_probs.append(0.0)
                failed_count += 1
                continue

            try:
                output = model(input_tensor, length_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = torch.max(probabilities).item()

                predicted_labels.append(predicted_class)
                predicted_probs.append(probabilities[0][1].item())
                processed_count += 1

            except Exception as e:
                logger.error(f"Ошибка предсказания для примера {i}: {e}")
                predicted_labels.append(0)
                predicted_probs.append(0.0)
                failed_count += 1
                continue

            if (i + 1) % 10 == 0:
                print(f"   Обработано {i + 1}/{len(texts)} примеров")

    print(f"📈 Обработка завершена: {processed_count} успешно, {failed_count} с ошибками")

    pred_class_0 = predicted_labels.count(0)
    pred_class_1 = predicted_labels.count(1)

    print(f"\n📊 Результаты предсказаний:")
    print(f"   Предсказано класс 0 (Не заявка): {pred_class_0}")
    print(f"   Предсказано класс 1 (Заявка): {pred_class_1}")

    if pred_class_0 == 0 or pred_class_1 == 0:
        print("⚠️  ВНИМАНИЕ: Модель предсказывает только один класс!")
        print("   Это может указывать на проблемы с обучением или несбалансированностью данных")

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    print("\n" + "=" * 60)
    print("📊 ОСНОВНЫЕ МЕТРИКИ КЛАССИФИКАЦИИ")
    print("=" * 60)
    print(f"🎯 Accuracy (Точность):  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"📏 Precision (Точность): {precision:.4f}")
    print(f"📈 Recall (Полнота):     {recall:.4f}")
    print(f"⚖️  F1-Score:            {f1:.4f}")

    cm = confusion_matrix(true_labels, predicted_labels)
    print("\n" + "=" * 60)
    print("🔄 МАТРИЦА ОШИБОК")
    print("=" * 60)
    print("\t\t\tПредсказано")
    print("\t\t\tНет\t\tДа")
    print(f"Реально\tНет\t{cm[0, 0]:3d}\t\t{cm[0, 1]:3d}")
    print(f"\t\tДа\t{cm[1, 0]:3d}\t\t{cm[1, 1]:3d}")

    print("\n" + "=" * 60)
    print("📋 ДЕТАЛЬНЫЙ ОТЧЕТ")
    print("=" * 60)
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=["Не заявка", "Заявка"],
            zero_division=0
        )
    )

    print("\n" + "=" * 60)
    print("🎯 АНАЛИЗ УВЕРЕННОСТИ МОДЕЛИ")
    print("=" * 60)

    confidence_class_0 = [prob for pred, prob in zip(predicted_labels, predicted_probs) if pred == 0]
    confidence_class_1 = [prob for pred, prob in zip(predicted_labels, predicted_probs) if pred == 1]

    if confidence_class_0:
        print(f"Уверенность для класса 0 (Не заявка):")
        print(f"   Средняя: {sum(confidence_class_0) / len(confidence_class_0):.3f}")
        print(f"   Минимальная: {min(confidence_class_0):.3f}")
        print(f"   Максимальная: {max(confidence_class_0):.3f}")

    if confidence_class_1:
        print(f"Уверенность для класса 1 (Заявка):")
        print(f"   Средняя: {sum(confidence_class_1) / len(confidence_class_1):.3f}")
        print(f"   Минимальная: {min(confidence_class_1):.3f}")
        print(f"   Максимальная: {max(confidence_class_1):.3f}")

    print("\n" + "=" * 60)
    print("🔍 ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
    print("=" * 60)

    correct_count = 0
    incorrect_examples = []

    for i, (true, pred, prob, text) in enumerate(
            zip(true_labels, predicted_labels, predicted_probs, texts)
    ):
        status = "✅" if true == pred else "❌"
        if true == pred:
            correct_count += 1
        else:
            incorrect_examples.append((true, pred, prob, text))

        if i < 3:
            class_name_true = "Заявка" if true == 1 else "Не заявка"
            class_name_pred = "Заявка" if pred == 1 else "Не заявка"
            confidence = prob if pred == 1 else (1 - prob)

            print(f"{status} Пример {i + 1}:")
            print(f"   Истина: {class_name_true}")
            print(f"   Предсказано: {class_name_pred}")
            print(f"   Уверенность: {confidence:.3f}")
            print(f"   Текст: {text[:80]}...")
            print()

    if incorrect_examples:
        print(f"\n❌ ОШИБОЧНЫЕ ПРЕДСКАЗАНИЯ (первые 3):")
        for i, (true, pred, prob, text) in enumerate(incorrect_examples[:3]):
            class_name_true = "Заявка" if true == 1 else "Не заявка"
            class_name_pred = "Заявка" if pred == 1 else "Не заявка"
            confidence = prob if pred == 1 else (1 - prob)

            print(f"   Ошибка {i + 1}:")
            print(f"      Истина: {class_name_true}, Предсказано: {class_name_pred}")
            print(f"      Уверенность: {confidence:.3f}")
            print(f"      Текст: {text[:60]}...")

    print(f"\n📊 Итоговая статистика:")
    print(f"   Правильных предсказаний: {correct_count}/{len(texts)} ({correct_count / len(texts) * 100:.1f}%)")
    print(f"   Ошибок: {len(incorrect_examples)}/{len(texts)} ({len(incorrect_examples) / len(texts) * 100:.1f}%)")

    if pred_class_0 == 0 or pred_class_1 == 0:
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        print(f"   1. Проверьте баланс классов в обучающих данных")
        print(f"   2. Убедитесь, что модель обучена на разнообразных примерах")
        print(f"   3. Попробуйте настроить порог классификации")
        print(f"   4. Проверьте качество предобработки текста")

if __name__ == "__main__":
    calculate_metrics()