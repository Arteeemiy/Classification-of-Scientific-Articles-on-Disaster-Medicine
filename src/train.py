import torch
import torch.nn as nn

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct_predictions, total_samples = 0, 0, 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        logits = nn.functional.normalize(outputs.logits, dim=1)

        loss = criterion(logits, labels)
        _, preds = torch.max(logits, dim=1)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += batch_size

    return total_loss / total_samples, correct_predictions / total_samples