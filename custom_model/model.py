import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.nn as nn
import os
import sys

class MultiLingEmotion(nn.Module):
    def __init__(self, target_emotions=None):
        super(MultiLingEmotion, self).__init__()

        self.target_emotions = target_emotions or ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]
        self.vocab_size = len(self.target_emotions)

        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self",
            attn_implementation="eager"
        )

        self.linear = nn.Linear(self.encoder.config.hidden_size, self.vocab_size)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = self.encoder(x)
        x = x.last_hidden_state
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    SUBESCO_DIR = os.path.join(PROJECT_ROOT, "SUBESCO")

    def fix_path(example):
        filename = os.path.basename(example["path"])
        example["path"] = os.path.join(SUBESCO_DIR, filename)
        return example

    current_dir = os.path.dirname(__file__)  # custom_model/
    data_dir = os.path.abspath(os.path.join(current_dir, "../data"))
    sys.path.append(data_dir)

    from get_data import get_data

    datasets = get_data()

    model = MultiLingEmotion()
    
    model.eval()
  
    all_outputs = {}

    for lang, loader in datasets["train"].items():
        if loader is None:
            print(f"No data loader for {lang}, skipping.")
            continue

        print(f"Running {lang} dataset...")
        all_outputs[lang] = []

        for batch_idx, batch in enumerate(loader):
            if batch is None:
                print(f"  Skipping empty batch {batch_idx} in {lang}")
                continue

            try:
                audio, labels = batch
                audio = audio
                labels = labels

                # Forward pass
                with torch.no_grad():
                    outputs = model(audio)

                all_outputs[lang].append(outputs)

            except RuntimeError as e:
                print(f"  Skipping corrupted batch {batch_idx} in {lang}: {e}")
                continue

    #combine all outputs for each lang
    for lang in all_outputs:
        if all_outputs[lang]:
            all_outputs[lang] = torch.cat(all_outputs[lang], dim=0)
            print(f"{lang} total embeddings shape: {all_outputs[lang].shape}")

    current_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_dir, "multi_ling_emotion.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
