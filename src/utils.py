import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import Dataset


def load_model(model_name="ruBERT-base"):
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    return model, tokenizer


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        texts = file.readlines()
    return [text.strip() for text in texts]


def prepare_data_for_inference(texts):
    labels = [0] * len(texts)
    return texts, labels
