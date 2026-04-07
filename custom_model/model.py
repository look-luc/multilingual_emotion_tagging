import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.nn as nn

class MultiLingEmotion(nn.Module):
    def __init__(self, input, device):
        super(MultiLingEmotion, self).__init__()

        self.input = input
        self.vocab_size = 7
        self.vocab = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]

        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self",
            dtype=torch.float16,
            attn_implementation="flash_attention_2"
        ).to(device)
        pass
    def forward(self, x, target):
        pass