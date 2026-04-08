import torch
import os
import io
import kagglehub
import torchaudio
from datasets import load_dataset, ClassLabel, Audio
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio.transforms as T

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

model.eval()
CACHE_FILE = "bangla_embeddings.pt"

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

    
    bangla_train = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus/resolve/main/train.jsonl",
        split="train"
    ).select_columns(["path", "emotional_state"])
    bangla_train = bangla_train.map(fix_path)
    bangla_train = bangla_train.map(load_audio_to_tensor)

    label_to_id = {label: i for i, label in enumerate(sorted(set(bangla_train["emotional_state"])))}

    def label_to_tensor(example):
        example["emotional_state"] = torch.tensor(label_to_id[example["emotional_state"]])
        return example
    
    bangla_train = bangla_train.map(label_to_tensor)
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached embeddings from {CACHE_FILE}...")
        data = torch.load(CACHE_FILE)
        bangla_train = data["dataset"]
        label_to_id = data["label_to_id"]
    else:
        print("Computing Wav2Vec2 embeddings for Bangla dataset (CPU may take a while)...")
        bangla_train = bangla_train.map(add_wav2vec2_embeddings, batched=True, batch_size=8)

        torch.save({
            "dataset": bangla_train,
            "label_to_id": label_to_id
        }, CACHE_FILE)

        print(f"Embeddings cached to {CACHE_FILE}")

    bangla_train.set_format(type="torch", columns=["embedding", "emotional_state"])


    
    ban_train_size = int(0.8 * len(bangla_train))
    ban_test_size = len(bangla_train) - ban_train_size
    ban_train, ban_test = random_split(
        bangla_train, [ban_train_size, ban_test_size], generator=torch.Generator().manual_seed(42)
    )

    ban_train = DataLoader(ban_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn)
    ban_test = DataLoader(ban_test, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)


    ch = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    ch = ch.map(lambda x: {"label_str": ch.features["label"].int2str(x)}, input_columns=["label"])
    datasets["train"]["chinese"], datasets["test"]["chinese"] = processing(ch, "label_str")

    eng = load_dataset("En1gma02/english_emotions", split="train")
    datasets["train"]["english"], datasets["test"]["english"] = processing(eng, "style")

    span_path = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    span = load_dataset("audiofolder", data_dir=span_path, split="train")
    span = span.cast_column("audio", Audio(decode=False))  
    def map_span(audio_dict):
        fname = os.path.basename(audio_dict["path"]).lower()
        lbl = next((v for k, v in emotion_map.items() if k in fname), "unknown")
        return {"extracted_label": lbl}
    span = span.map(map_span, input_columns=["audio"])
    datasets["train"]["spanish"], datasets["test"]["spanish"] = processing(span, "extracted_label")

    ara_path = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")
    ara = load_dataset("audiofolder", data_dir=ara_path, split="train")
    ara = ara.cast_column("audio", Audio(decode=False)) 
    def map_ara(audio_dict):
        folder = os.path.basename(os.path.dirname(audio_dict["path"]))
        return {"folder_label": folder}
    ara = ara.map(map_ara, input_columns=["audio"])
    datasets["train"]["arabic"], datasets["test"]["arabic"] = processing(ara, "folder_label")

    return datasets

def collate_fn(batch):
    embeddings = torch.stack([item["embedding"] for item in batch])
    labels = torch.stack([item["emotional_state"] for item in batch])

    return {
        "embedding": embeddings,
        "label": labels
    }

def fix_path(example):
    filename = os.path.basename(example["path"])   # get 'F_01_OISHI_S_10_ANGRY_1.wav'
    example["path"] = os.path.join("SUBESCO", filename)  # local folder
    return example

resampler = T.Resample(orig_freq=48000, new_freq=16000)  # adjust if needed

def load_audio_to_tensor(example):
    waveform, sr = torchaudio.load(example["path"])

    # convert to mono -> issue with audio file type
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    audio = waveform.numpy()

    if audio.ndim == 0:
        audio = audio.reshape(1)

    example["audio"] = audio.tolist()
    return example


def add_wav2vec2_embeddings(batch):
    inputs = processor(
        batch["audio"],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )#.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)

    batch["embedding"] = embeddings.cpu().numpy()
    return batch
