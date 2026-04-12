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

        self.text_encoder = BertModel.from_pretrained(
            "google-bert/bert-base-multilingual-cased"
        )
        self.audio_hidden_size = self.audio_encoder.config.hidden_size
        self.text_hidden_size = self.text_encoder.config.hidden_size
        self.fusion_hidden_size = 512

        self.audio_projection = nn.Linear(self.audio_hidden_size, self.fusion_hidden_size)
        self.text_projection = nn.Linear(self.text_hidden_size, self.fusion_hidden_size)

        self.text_to_audio_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.audio_to_text_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.text_fusion_norm = nn.LayerNorm(self.fusion_hidden_size)
        self.audio_fusion_norm = nn.LayerNorm(self.fusion_hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(
                self.fusion_hidden_size * 2,
                512
            ),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )

    def masked_mean_pool(self, sequence, mask):
        if mask is None:
            return torch.mean(sequence, dim=1)

        expanded_mask = mask.unsqueeze(-1).to(sequence.dtype)
        masked_sequence = sequence * expanded_mask
        normalizer = expanded_mask.sum(dim=1).clamp(min=1.0)
        return masked_sequence.sum(dim=1) / normalizer

    def forward(self, audio, input_ids=None, attention_mask=None, audio_attention_mask=None):
        # Accept either a batch dict or explicit tensors so older call sites keep working.
        if isinstance(audio, dict):
            batch = audio
            audio = batch["audio"]
            audio_attention_mask = batch.get("audio_attention_mask")
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")

        if audio.dtype != torch.float32:
            audio = audio.float()
        audio_out = self.audio_encoder(
            audio,
            attention_mask=audio_attention_mask
        ).last_hidden_state
        projected_audio = self.audio_projection(audio_out)

        feature_attention_mask = None
        audio_key_padding_mask = None
        if audio_attention_mask is not None:
            feature_attention_mask = self.audio_encoder._get_feature_vector_attention_mask(
                audio_out.size(1),
                audio_attention_mask
            )
            audio_key_padding_mask = ~feature_attention_mask.bool()

        if input_ids is not None and attention_mask is not None:
            text_out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
            projected_text = self.text_projection(text_out)

            text_key_padding_mask = ~attention_mask.bool()

            text_attended, _ = self.text_to_audio_attention(
                query=projected_text,
                key=projected_audio,
                value=projected_audio,
                key_padding_mask=audio_key_padding_mask
            )
            text_attended = self.text_fusion_norm(projected_text + text_attended)

            audio_attended, _ = self.audio_to_text_attention(
                query=projected_audio,
                key=projected_text,
                value=projected_text,
                key_padding_mask=text_key_padding_mask
            )
            audio_attended = self.audio_fusion_norm(projected_audio + audio_attended)

            pooled_text = self.masked_mean_pool(text_attended, attention_mask)
            pooled_audio = self.masked_mean_pool(audio_attended, feature_attention_mask)
        else:
            pooled_audio = self.masked_mean_pool(projected_audio, feature_attention_mask)
            pooled_text = torch.zeros(
                pooled_audio.size(0),
                self.fusion_hidden_size,
                device=pooled_audio.device,
                dtype=pooled_audio.dtype
            )
        
        combined = torch.cat((pooled_audio, pooled_text), dim=1)
        logits = self.classifier(combined)

        return logits
