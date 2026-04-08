import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertModel

class MultiLingEmotion(nn.Module):
    def __init__(self, target_emotions=None):
        super(MultiLingEmotion, self).__init__()

        self.target_emotions = target_emotions or [
            "angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"
        ]
        self.num_classes = len(self.target_emotions)

        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self",
            attn_implementation="eager"
        )
        self.text_encoder = BertModel.from_pretrained("google-bert/bert-base-multilingual-cased")

        self.classifier = nn.Sequential(
            nn.Linear(
                self.text_encoder.config.hidden_size + self.audio_encoder.config.hidden_size,
                512
            ),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()

        audio_out = self.audio_encoder(x).last_hidden_state
        pooled_audio = torch.mean(audio_out, dim=1)

        transcription_logits = self.text_encoder(x).logits
        pooled_text = torch.mean(transcription_logits, dim=1)

        combined = torch.cat((pooled_audio, pooled_text), dim=1)
        logits = self.classifier(combined)
        return logits
