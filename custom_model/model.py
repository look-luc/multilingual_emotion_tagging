import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class MultiLingEmotion(nn.Module):
    def __init__(self, target_emotions=None):
        super(MultiLingEmotion, self).__init__()

        self.target_emotions = target_emotions or [
            "angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"
        ]
        self.num_classes = len(self.target_emotions)

        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self",
            attn_implementation="eager"
        )

        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            self.num_classes
        )

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()

        outputs = self.encoder(x)
        hidden_states = outputs.last_hidden_state

        pooled = torch.mean(hidden_states, dim=1)

        logits = self.classifier(pooled)
        return logits
