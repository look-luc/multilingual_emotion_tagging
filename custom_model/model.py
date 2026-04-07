import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.nn as nn

class MultiLingEmotion(nn.Module):
    def __init__(self):
        super(MultiLingEmotion, self).__init__()

        self.vocab_size = 7
        self.target_emotions = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]

        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self",
            dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )

        self.linear = nn.Linear(self.encoder.config.hidden_size, self.vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        x = x.last_hidden_state
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        return x