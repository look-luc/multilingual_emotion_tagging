import sys
from data import get_data
from custom_model import model
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def main():
    print(f"Using device: {device}")
    dataset = get_data.get_data()
    multiling_model = model.MultiLingEmotion().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(multiling_model.parameters(), lr=1e-5, weight_decay=0.01)

    loss_overtime = []
    epochs = 10

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        multiling_model.train()
        total_loss = 0
        batch_count = 0
        trained_languages = 0

        for lang, loader in dataset["train"].items():
            if loader is None:
                print(f"Skipping {lang} (no data)")
                continue

            print(f"Training on {lang}")
            lang_batch_count = 0

            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    continue

                audio = batch["audio"].to(device)
                audio_attention_mask = batch.get("audio_attention_mask")
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                labels = batch["labels"].to(device)

                if audio_attention_mask is not None:
                    audio_attention_mask = audio_attention_mask.to(device)
                if input_ids is not None:
                    input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                optimizer.zero_grad()
                outputs = multiling_model(audio, input_ids, attention_mask, audio_attention_mask)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                if device.type == "mps":
                    torch.mps.empty_cache()

                total_loss += loss.item()
                batch_count += 1
                lang_batch_count += 1

            if lang_batch_count == 0:
                print(f"[TRAIN] {lang}: no usable batches after collation")
            else:
                trained_languages += 1
                print(f"[TRAIN] {lang}: trained on {lang_batch_count} batches")

        if batch_count == 0:
            raise RuntimeError(
                "No training batches were produced. Check dataset filtering/collation logs above."
            )

        avg_loss = total_loss / batch_count
        loss_overtime.append(avg_loss)

        print(f"Average Loss: {avg_loss:.4f} across {batch_count} batches from {trained_languages} languages")

        print("Starting validation...")
        multiling_model.eval()

        global_true = []
        global_pred = []

        with torch.no_grad():
            for lang, loader in dataset["test"].items():
                if loader is None:
                    continue

                lang_true, lang_pred = [], []

                for batch in loader:
                    if batch is None:
                        continue

                    audio = batch["audio"].to(device)
                    audio_attention_mask = batch.get("audio_attention_mask")
                    input_ids = batch.get("input_ids")
                    attention_mask = batch.get("attention_mask")
                    labels = batch["labels"].to(device)

                    if audio_attention_mask is not None:
                        audio_attention_mask = audio_attention_mask.to(device)
                    if input_ids is not None:
                        input_ids = input_ids.to(device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    outputs = multiling_model(audio, input_ids, attention_mask, audio_attention_mask)
                    _, preds = torch.max(outputs, 1)

                    if device.type == "mps":
                        torch.mps.empty_cache()

                    lang_true.extend(labels.cpu().numpy())
                    lang_pred.extend(preds.cpu().numpy())

                if len(lang_true) > 0:
                    acc = accuracy_score(lang_true, lang_pred)
                    f1 = f1_score(lang_true, lang_pred, average="macro")

                    print(f"{lang}: acc={acc:.3f}, f1={f1:.3f}")

                    global_true.extend(lang_true)
                    global_pred.extend(lang_pred)

        if len(global_true) > 0:
            global_acc = accuracy_score(global_true, global_pred)
            global_f1 = f1_score(global_true, global_pred, average="macro")

            print(f"\nGLOBAL: acc={global_acc:.3f}, f1={global_f1:.3f}")

    torch.save(multiling_model.state_dict(), "multi_ling_emotion.pth")
    print("\nModel saved as multi_ling_emotion.pth")


if __name__ == "__main__":
    main()
    sys.exit(0)
