import torch
from torch.utils.data import DataLoader
from utils import load_model, CustomDataset, load_text_from_file, prepare_data_for_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'ruBERT-base'  
model, tokenizer = load_model(model_name)
model.to(device)

file_path = input("Введите путь к файлу с текстами (.txt): ")

texts = load_text_from_file(file_path)
print(f"Загружено {len(texts)} текстов из файла.")

texts, labels = prepare_data_for_inference(texts)
dataset = CustomDataset(texts, labels, tokenizer)
data_loader = DataLoader(dataset, batch_size=2)

model.eval()  
predictions = []

with torch.no_grad():  
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        predictions.extend(preds.cpu().numpy())

for text, pred in zip(texts, predictions):
    print(f"Текст: {text}")
    print(f"Предсказание: {pred}\n")
