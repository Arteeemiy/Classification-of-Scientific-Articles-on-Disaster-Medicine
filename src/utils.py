import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def preprocess_data(df, text_col, label_col, tokenizer_name, max_len):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    return CustomDataset(texts, labels, tokenizer, max_len)


def compute_metrics(true_labels, predictions):
    from sklearn.metrics import classification_report, confusion_matrix

    print("Classification Report:")
    print(classification_report(true_labels, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predictions))


def plot_metrics(
    train_losses, val_losses, train_accuracies, val_accuracies, num_epochs
):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, num_epochs + 1), train_accuracies, label="Train Accuracy", marker="o"
    )
    plt.plot(
        range(1, num_epochs + 1),
        val_accuracies,
        label="Validation Accuracy",
        marker="o",
    )
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
