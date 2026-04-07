import torch
import os
import io
import kagglehub
import torchaudio
from datasets import load_dataset, ClassLabel, Audio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

os.environ["datasets_audio_decoder_backend"] = "ffmpeg"

target_emotions = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]
shared_emotions = ClassLabel(names=target_emotions)

emotion_map = {
    "anger": "angry", "ang": "angry", "0": "angry", "angry": "angry",
    "happiness": "happy", "ale": "happy", "1": "happy", "happy": "happy",
    "sadness": "sad", "tri": "sad", "2": "sad", "sad": "sad",
    "neu": "neutral", "3": "neutral", "exhausted": "neutral", "neutral": "neutral",
    "surprised": "surprise", "sor": "surprise", "surprise": "surprise",
    "fearful": "fear", "mie": "fear", "fear": "fear",
    "disgusted": "disgust", "asc": "disgust", "disgust": "disgust"
}


def speech_collate_fn(batch):
    processed_audio, processed_labels = [], []
    for item in batch:
        audio_data = item.get("audio")
        if audio_data is None:
            continue

        try:
            if isinstance(audio_data, dict) and audio_data.get("bytes"):
                waveform, sr = torchaudio.load(io.BytesIO(audio_data["bytes"]))
            elif isinstance(audio_data, dict) and audio_data.get("path"):
                waveform, sr = torchaudio.load(audio_data["path"])
            else:
                continue

            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

            audio_tensor = waveform.float()

            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.mean(dim=0)

            label_val = item.get("label")
            if label_val is not None:
                processed_audio.append(audio_tensor)
                processed_labels.append(torch.tensor(label_val))

        except Exception as e:
            continue

    if not processed_audio:
        return None
    else:
        return pad_sequence(processed_audio, batch_first=True), torch.stack(processed_labels)

def normalize(label):
    raw_val = str(label).lower().strip()
    return {"standard_label": emotion_map.get(raw_val, "unknown")}


def processing(dataset, label_column_name):
    dataset = dataset.map(normalize, input_columns=[label_column_name])
    dataset = dataset.filter(lambda x: x in target_emotions, input_columns=["standard_label"])

    if "label" in dataset.column_names and label_column_name != "label":
        dataset = dataset.remove_columns(["label"])

    dataset = dataset.rename_column(label_column_name, "old_label")
    dataset = dataset.rename_column("standard_label", "label")

    dataset = dataset.cast_column("label", shared_emotions)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

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
    ch = ch.map(lambda x: {"label_str": ch.features["label"].int2str(x)}, input_columns=["label"])
    datasets["train"]["chinese"], datasets["test"]["chinese"] = processing(ch, "label_str")

    eng = load_dataset("En1gma02/english_emotions", split="train")
    datasets["train"]["english"], datasets["test"]["english"] = processing(eng, "style")

    span_path = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    span = load_dataset("audiofolder", data_dir=span_path, split="train")
    span = span.cast_column("audio", Audio(decode=False))  # FIX: Prevent torchcodec crash
    def map_span(audio_dict):
        fname = os.path.basename(audio_dict["path"]).lower()
        lbl = next((v for k, v in emotion_map.items() if k in fname), "unknown")
        return {"extracted_label": lbl}
    span = span.map(map_span, input_columns=["audio"])
    datasets["train"]["spanish"], datasets["test"]["spanish"] = processing(span, "extracted_label")

    ara_path = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")
    ara = load_dataset("audiofolder", data_dir=ara_path, split="train")
    ara = ara.cast_column("audio", Audio(decode=False))  # FIX: Prevent torchcodec crash
    def map_ara(audio_dict):
        folder = os.path.basename(os.path.dirname(audio_dict["path"]))
        return {"folder_label": folder}
    ara = ara.map(map_ara, input_columns=["audio"])
    datasets["train"]["arabic"], datasets["test"]["arabic"] = processing(ara, "folder_label")

    return datasets