import torch
import torchaudio
import os
import kagglehub
from datasets import load_dataset, ClassLabel, Audio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Environment Setup
os.environ["datasets_audio_decoder_backend"] = "ffmpeg"

target_emotions = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]
shared_emotions = ClassLabel(names=target_emotions)

emotion_map = {
    "anger": "angry", "ang": "angry", "0": "angry",
    "happiness": "happy", "ale": "happy", "1": "happy",
    "sadness": "sad", "tri": "sad", "2": "sad",
    "neu": "neutral", "3": "neutral", "exhausted": "neutral",
    "surprised": "surprise", "sor": "surprise",
    "fearful": "fear", "mie": "fear",
    "disgusted": "disgust", "asc": "disgust"
}

def speech_collate_fn(batch):
    processed_audio, processed_labels = [], []
    for item in batch:
        try:
            audio_data = item.get("audio")
            if audio_data is None or "array" not in audio_data:
                continue

            audio_tensor = torch.from_numpy(audio_data["array"]).float()

            # Mono conversion
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.mean(dim=0)

            label_val = item.get("label")
            if label_val is not None:
                processed_audio.append(audio_tensor)
                processed_labels.append(torch.tensor(label_val))
        except Exception:
            continue

    if not processed_audio: return None
    return pad_sequence(processed_audio, batch_first=True), torch.stack(processed_labels)

def normalize(item, label):
    raw_val = str(item[label]).lower().strip()
    return {"standard_label": emotion_map.get(raw_val, raw_val)}

def processing(dataset, label):
    dataset = dataset.map(lambda x: normalize(x, label))
    dataset = dataset.filter(lambda x: x["standard_label"] != None)

    dataset = dataset.rename_column(label, "old_label")
    dataset = dataset.rename_column("standard_label", "label")
    dataset = dataset.cast_column("label", shared_emotions)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    split = dataset.train_test_split(test_size=0.2, seed=42)

    train_loader = DataLoader(split["train"], batch_size=32, shuffle=True, collate_fn=speech_collate_fn)
    test_loader = DataLoader(split["test"], batch_size=32, shuffle=False, collate_fn=speech_collate_fn)

    return train_loader, test_loader

def get_data():
    datasets = {"train": {}, "test": {}}

    jap = load_dataset("asahi417/jvnv-emotional-speech-corpus", split="test")
    datasets["train"]["japanese"], datasets["test"]["japanese"] = processing(jap, "style")

    ban = load_dataset("json",
                       data_files="https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus/resolve/main/train.jsonl",
                       split="train")
    ban = ban.rename_column("path", "audio")
    datasets["train"]["bangla"], datasets["test"]["bangla"] = processing(ban, "emotional_state")

    ch = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    ch = ch.map(lambda x: {"label_str": ch.features["label"].int2str(x["label"])})
    datasets["train"]["chinese"], datasets["test"]["chinese"] = processing(ch, "label_str")

    eng = load_dataset("En1gma02/english_emotions", split="train")
    datasets["train"]["english"], datasets["test"]["english"] = processing(eng, "style")

    span_path = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    span = load_dataset("audiofolder", data_dir=span_path, split="train")
    def map_span(ex):
        fname = os.path.basename(ex["audio"]["path"]).lower()
        lbl = next((v for k, v in emotion_map.items() if k in fname), "unknown")
        return {"extracted_label": lbl}
    span = span.map(map_span)
    datasets["train"]["spanish"], datasets["test"]["spanish"] = processing(span, "extracted_label")

    ara_path = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")
    ara = load_dataset("audiofolder", data_dir=ara_path, split="train")

    # Use folder names as labels
    def map_ara(ex):
        folder = os.path.basename(os.path.dirname(ex["audio"]["path"]))
        return {"folder_label": folder}

    ara = ara.map(map_ara)
    datasets["train"]["arabic"], datasets["test"]["arabic"] = processing(ara, "folder_label")

    return datasets