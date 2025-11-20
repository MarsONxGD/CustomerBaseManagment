import json
import logging
import os
import re
import sys
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.email_classifier.model import EmailClassifier


def setup_logging():
    log_dir = '../../log'
    log_file = os.path.join(log_dir, 'trainer.log')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Запуск обучения модели EmailClassifier")
    logger.info("=" * 60)

    return logger


logger = setup_logging()


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Токенизация и преобразование в индексы
        tokens = self.tokenize(text)
        indexed = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        text_length = min(len(indexed), self.max_length)

        # Обрезка или паддинг до max_length
        if len(indexed) > self.max_length:
            indexed = indexed[:self.max_length]
        else:
            indexed = indexed + [self.vocab['<PAD>']] * (self.max_length - len(indexed))

        return (torch.tensor(indexed, dtype=torch.long),
                torch.tensor(text_length, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))

    @staticmethod
    def tokenize(text):
        # Простая токенизация
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens


class TextProcessor:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def build_vocab(self, texts, min_freq=2):
        logger.info(f"Построение словаря с min_freq={min_freq}")
        # Сбор всех токенов
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Создание словаря
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for token, count in counter.items():
            if count >= min_freq:
                self.vocab[token] = idx
                idx += 1

        self.vocab_size = len(self.vocab)
        logger.info(f"Словарь построен. Размер: {self.vocab_size} токенов")

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens


def collate_fn(batch):
    """Функция для создания батчей с разной длиной последовательностей"""
    texts, lengths, labels = zip(*batch)

    texts = torch.stack(texts)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)

    # Сортируем по длине для эффективности LSTM
    lengths, sort_idx = lengths.sort(descending=True)
    texts = texts[sort_idx]
    labels = labels[sort_idx]

    return texts, lengths, labels


def train_model():
    # Параметры
    BATCH_SIZE = 16
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 64
    OUTPUT_DIM = 2  # 2 класса: заявка и не заявка
    N_LAYERS = 2
    DROPOUT = 0.3
    N_EPOCHS = 15
    LEARNING_RATE = 0.0005
    MAX_LENGTH = 200

    logger.info("Параметры обучения:")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  EMBEDDING_DIM: {EMBEDDING_DIM}")
    logger.info(f"  HIDDEN_DIM: {HIDDEN_DIM}")
    logger.info(f"  OUTPUT_DIM: {OUTPUT_DIM}")
    logger.info(f"  N_LAYERS: {N_LAYERS}")
    logger.info(f"  DROPOUT: {DROPOUT}")
    logger.info(f"  N_EPOCHS: {N_EPOCHS}")
    logger.info(f"  LEARNING_RATE: {LEARNING_RATE}")
    logger.info(f"  MAX_LENGTH: {MAX_LENGTH}")

    # Загрузка данных - ТОЛЬКО ИЗ CSV ФАЙЛА
    try:
        logger.info("Загрузка данных из temp.csv")
        df = pd.read_csv('../../datasets/data.csv')
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        logger.info(f"Загружено {len(texts)} примеров из temp.csv")

        # Проверка наличия необходимых колонок
        if 'text' not in df.columns or 'label' not in df.columns:
            error_msg = "ОШИБКА: Файл temp.csv должен содержать колонки 'text' и 'label'"
            logger.error(error_msg)
            sys.exit(1)

    except FileNotFoundError:
        error_msg = "ОШИБКА: Файл temp.csv не найден!"
        logger.error(error_msg)
        print("Пожалуйста, создайте файл temp.csv с колонками 'text' и 'label'")
        sys.exit(1)
    except Exception as e:
        error_msg = f"ОШИБКА при загрузке temp.csv: {e}"
        logger.error(error_msg)
        sys.exit(1)

    # Проверка, что есть данные для обучения
    if len(texts) == 0:
        error_msg = "ОШИБКА: Файл temp.csv не содержит данных"
        logger.error(error_msg)
        sys.exit(1)

    # Предобработка текстов
    logger.info("Начало предобработки текстов")
    processor = TextProcessor()
    processor.build_vocab(texts)

    # Сохранение словаря
    try:
        with open('../../models/vocabulary.json', 'w', encoding='utf-8') as f:
            json.dump(processor.vocab, f, ensure_ascii=False)
        logger.info(f"Словарь сохранен в ../../models/vocabulary.json")
    except Exception as e:
        logger.error(f"Ошибка при сохранении словаря: {e}")

    # Разделение на train/validation
    logger.info("Разделение данных на train/validation")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info(f"Train: {len(train_texts)} примеров, Validation: {len(val_texts)} примеров")

    # Создание датасетов и даталоадеров
    train_dataset = TextDataset(train_texts, train_labels, processor.vocab, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, processor.vocab, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Инициализация модели
    logger.info("Инициализация модели EmailClassifier")
    model = EmailClassifier(
        vocab_size=processor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        bidirectional=True
    )

    logger.info("Модель инициализирована:")
    logger.info(f"- Vocab size: {processor.vocab_size}")
    logger.info(f"- Embedding dim: {EMBEDDING_DIM}")
    logger.info(f"- Hidden dim: {HIDDEN_DIM}")
    logger.info(f"- LSTM layers: {N_LAYERS}")
    logger.info(f"- Bidirectional: True")

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logger.info(f"Оптимизатор: Adam, LR: {LEARNING_RATE}")

    # Обучение
    train_losses = []
    val_losses = []
    val_accuracies = []

    logger.info("Начало обучения...")
    logger.info("-" * 60)

    for epoch in range(N_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for i, (texts_batch, lengths_batch, labels_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(texts_batch, lengths_batch)
            loss = criterion(outputs, labels_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for texts_batch, lengths_batch, labels_batch in val_loader:
                outputs = model(texts_batch, lengths_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_accuracy)

        # Логирование прогресса эпохи
        logger.info(f'Epoch {epoch + 1:2d}/{N_EPOCHS}:')
        logger.info(f'  Train Loss: {train_loss_avg:.4f}')
        logger.info(f'  Val Loss:   {val_loss_avg:.4f}')
        logger.info(f'  Val Accuracy: {val_accuracy:6.2f}%')
        logger.info('-' * 40)

    # Сохранение модели
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': processor.vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'output_dim': OUTPUT_DIM,
            'n_layers': N_LAYERS,
            'dropout': DROPOUT,
            'bidirectional': True
        }, '../../models/email_classifier.pth')
        logger.info("Модель успешно сохранена в ../../models/email_classifier.pth")
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {e}")

    logger.info("Обучение завершено!")
    logger.info(f"Финальная точность на валидации: {val_accuracies[-1]:.2f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.exception(f"Критическая ошибка во время обучения: {e}")
        sys.exit(1)
