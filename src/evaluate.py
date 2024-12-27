import torch
import matplotlib.pyplot as plt


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss, correct_predictions, total_samples = 0, 0, 0

    batch_losses = []
    batch_accuracies = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            batch_loss = loss.item()
            batch_accuracy = (preds == labels).sum().item() / labels.size(0)

            batch_losses.append(batch_loss)
            batch_accuracies.append(batch_accuracy)

            total_loss += batch_loss * labels.size(0)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(batch_losses, label="Batch Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss per Batch")
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(batch_accuracies, label="Batch Accuracy", color="orange")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Batch")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return avg_loss, accuracy
