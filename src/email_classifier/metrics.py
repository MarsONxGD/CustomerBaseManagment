import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
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
    except FileNotFoundError:
        print(f"❌ ОШИБКА: Файл {test_data_path} не найден!")
        return

    predicted_labels = []
    predicted_probs = []

    print("\n🔮 Выполняю предсказания...")

    with torch.no_grad():
        for text in texts:
            tokens = text.lower().split()
            indexed = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

            if len(indexed) > 200:
                indexed = indexed[:200]
            else:
                indexed = indexed + [vocab["<PAD>"]] * (200 - len(indexed))

            input_tensor = torch.tensor(indexed, dtype=torch.long).unsqueeze(0)
            length_tensor = torch.tensor([min(len(tokens), 200)], dtype=torch.long)

            output = model(input_tensor, length_tensor)
            prob = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            predicted_labels.append(predicted.item())
            predicted_probs.append(prob[0][1].item())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

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
    print("         Предсказано")
    print("         Нет   Да")
    print(f"Реально Нет  {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"        Да   {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    print("\n" + "=" * 60)
    print("📋 ДЕТАЛЬНЫЙ ОТЧЕТ")
    print("=" * 60)
    print(
        classification_report(
            true_labels, predicted_labels, target_names=["Не заявка", "Заявка"]
        )
    )

    print("\n" + "=" * 60)
    print("🔍 ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
    print("=" * 60)

    correct_count = 0
    for i, (true, pred, prob, text) in enumerate(
        zip(true_labels, predicted_labels, predicted_probs, texts)
    ):
        status = "✅" if true == pred else "❌"
        if true == pred:
            correct_count += 1

        if i < 10:
            print(
                f"{status} [Истина: {true}, Предсказано: {pred}, Вероятность: {prob:.3f}]"
            )
            print(f"   Текст: {text[:80]}...")
            print()

    print(
        f"📊 Правильных предсказаний: {correct_count}/{len(texts)} ({correct_count / len(texts) * 100:.1f}%)"
    )


if __name__ == "__main__":
    calculate_metrics()
