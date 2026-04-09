import os
import io
import torch
import torchaudio
import kagglehub
from datasets import load_dataset, ClassLabel, Audio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

os.environ["HF_DATASETS_AUDIO_DECODER_BACKEND"] = "ffmpeg"
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SUBESCO_DIR = os.path.join(PROJECT_ROOT, "SUBESCO")

target_emotions = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]
shared_emotions = ClassLabel(names=target_emotions)

emotion_map = {
    "anger": "angry", "ang": "angry", "0": "angry", "angry": "angry",
    "happiness": "happy", "ale": "happy", "1": "happy", "happy": "happy", "laughing":"happy",
    "sadness": "sad", "tri": "sad", "2": "sad", "sad": "sad",
    "neu": "neutral", "3": "neutral", "exhausted": "neutral", "neutral": "neutral", "confused": "neutral",
    "surprised": "surprise", "sor": "surprise", "surprise": "surprise",
    "fearful": "fear", "mie": "fear", "fear": "fear",
    "disgusted": "disgust", "asc": "disgust"
}


def speech_collate_fn(batch):
    audios, labels = [], []

    for item in batch:
        audio_data = item["audio"]

        try:
            path = audio_data["path"]
            waveform, sr = torchaudio.load(path)

            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)

            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            audios.append(waveform)
            labels.append(torch.tensor(item["label"], dtype=torch.long))

        except Exception:
            # skip corrupt files cleanly
            continue

    if len(audios) == 0:
        return None

    return {
        "audio": pad_sequence(audios, batch_first=True),
        "labels": torch.stack(labels)
    }


def label_to_tensor(example, label_field="emotional_state"):
    raw_label = str(example.get(label_field, "")).lower().strip()
    standard_label = emotion_map.get(raw_label)
    example["label"] = target_emotions.index(standard_label) if standard_label in target_emotions else -1
    return example


def processing(dataset, label_column_name):
    def standardize_label(example):
        raw_label = str(example.get(label_column_name, "neutral")).lower().strip()
        return {"label": emotion_map.get(raw_label, "neutral")}
    
    dataset = dataset.map(standardize_label, remove_columns=[])
    dataset = dataset.cast_column("label", shared_emotions)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    split = dataset.train_test_split(test_size=0.2, seed=42)

    train_loader = DataLoader(split["train"], batch_size=32, shuffle=True, collate_fn=speech_collate_fn) \
        if len(split["train"]) > 0 else None
    test_loader = DataLoader(split["test"], batch_size=32, shuffle=False, collate_fn=speech_collate_fn) \
        if len(split["test"]) > 0 else None

    return train_loader, test_loader

def safe_processing(dataset, label_column):
    try:
        return processing(dataset, label_column)
    except Exception as e:
        print(f"Skipping dataset due to error: {e}")
        return None, None

def get_data():
    datasets_dict = {"train": {}, "test": {}}

    # Japanese
    jap = load_dataset("asahi417/jvnv-emotional-speech-corpus", split="test")

    jap = jap.cast_column("audio", Audio(sampling_rate=16000, decode=True))

    def map_japanese_label(example):
        raw_label = example.get("style", "neutral").lower().strip()
        standard_label = emotion_map.get(raw_label, "neutral")
        example["label"] = target_emotions.index(standard_label)
        return example

    jap = jap.map(map_japanese_label)

    jap = jap.filter(lambda x: x["audio"] is not None and x["audio"]["array"] is not None)

    split = jap.train_test_split(test_size=0.2, seed=42)
    train_loader = DataLoader(
        split["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=speech_collate_fn
    )
    test_loader = DataLoader(
        split["test"],
        batch_size=32,
        shuffle=False,
        collate_fn=speech_collate_fn
    )

    datasets_dict["train"]["japanese"] = train_loader
    datasets_dict["test"]["japanese"] = test_loader


    # Bangla
    bangla = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus/resolve/main/train.jsonl",
        split="train"
    ).select_columns(["path", "emotional_state"])
    bangla = bangla.map(lambda x: {"audio": os.path.join(SUBESCO_DIR, os.path.basename(x["path"]))})
    bangla = bangla.map(label_to_tensor)
    bangla = bangla.filter(lambda x: x["label"] >= 0)
    datasets_dict["train"]["bangla"], datasets_dict["test"]["bangla"] = processing(bangla, "emotional_state")

    # Chinese
    ch = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    label_class = ch.features["label"]
    ch = ch.map(lambda x: {"label_str": label_class.int2str(x["label"])})
    datasets_dict["train"]["chinese"], datasets_dict["test"]["chinese"] = processing(ch, "label_str")

    # English
    eng = load_dataset("En1gma02/english_emotions", split="train")
    datasets_dict["train"]["english"], datasets_dict["test"]["english"] = processing(eng, "style")

    # Spanish
    span_path = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    span = load_dataset("audiofolder", data_dir=span_path, split="train")
    span = span.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    def extract_spanish_label(x):
        audio_path = x["audio"]["path"] if "audio" in x and "path" in x["audio"] else None
        if not audio_path:
            return {"label": None}
        fname = os.path.basename(audio_path).lower()
        for k, v in emotion_map.items():
            if k in fname:
                return {"label": v}
        return {"label": None}

    span = span.map(extract_spanish_label)
    span = span.filter(lambda x: x["label"] is not None)
    datasets_dict["train"]["spanish"], datasets_dict["test"]["spanish"] = safe_processing(span, "label")

    # Arabic
    ara_path = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")
    ara = load_dataset("audiofolder", data_dir=ara_path, split="train")
    ara = ara.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    def extract_arabic_label(x):
        audio_path = x["audio"]["path"] if "audio" in x and "path" in x["audio"] else None
        if not audio_path:
            return {"label": None}
        folder_name = os.path.basename(os.path.dirname(audio_path))
        return {"label": emotion_map.get(folder_name.lower().strip(), None)}

    ara = ara.map(extract_arabic_label)
    ara = ara.filter(lambda x: x["label"] is not None)
    datasets_dict["train"]["arabic"], datasets_dict["test"]["arabic"] = safe_processing(ara, "label")

    return datasets_dict
