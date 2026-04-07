import sys
from data import get_data
from custom_model import model
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def main():
    dataset = get_data.get_data()
    multiling_model = model.MultiLingEmotion().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(multiling_model.parameters(), lr=1e-5, weight_decay=0.01)
    loss_overtime = []
    for epoch in range(10):
        multiling_model.train()
        total_loss = 0
        for lang, loader in dataset["train"].items():
            print(f"Epoch {epoch + 1}\nLanguage: {lang}")

            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    continue

                features, labels = batch
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                out = multiling_model(features)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        loss_overtime.append(total_loss / len(dataset["train"]))

        print("starting validation")
        multiling_model.eval()
        results = {}
        global_true = []
        global_pred = []
        with torch.no_grad():
            for lang, loader in dataset["test"].items():
                model_true, model_pred = [], []
                for batch in loader:
                    if batch is None:
                        continue

                    features, labels = batch
                    features, labels = features.to(device), labels.to(device)
                    out = multiling_model(features)

                    _, pred = torch.max(out, 1)

                    model_true.extend(labels.cpu().numpy())
                    model_pred.extend(pred.cpu().numpy())
                acc = accuracy_score(model_true, model_pred)
                macro_f1 = f1_score(model_true, model_pred, average='macro')

                results[lang] = {"accuracy": acc, "f1_macro": macro_f1}

                global_true.extend(model_true)
                global_pred.extend(model_pred)

    return 0

if __name__ == '__main__':
    main()
    sys.exit(0)
