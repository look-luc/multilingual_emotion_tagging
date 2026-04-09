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

        self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, 256)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 256)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )


    def forward(self, audio, input_ids, attention_mask):
        audio_seq = self.audio_encoder(audio).last_hidden_state
        text_seq = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
    
        audio_proj = self.audio_proj(audio_seq)
        text_proj = self.text_proj(text_seq)
    
        text_attended, _ = self.cross_attn(
            query=text_proj,
            key=audio_proj,
            value=audio_proj
        )

        text_feat = text_attended.mean(dim=1)
        audio_feat = audio_proj.mean(dim=1)

        combined = torch.cat([text_feat, audio_feat], dim=1)
        logits = self.classifier(combined)
    
        return logits
