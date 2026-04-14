import os
import io
import torch
import torchaudio
import kagglehub
import soundfile as sf
import re
from datasets import load_dataset, ClassLabel, Audio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
asr_model_name = "facebook/wav2vec2-xls-r-1b"
asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
# asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)
# asr_model.eval()
ASR_SAMPLE_RATE = 16000
DEFAULT_BATCH_SIZE = 4 if torch.backends.mps.is_available() else 4 if torch.cuda.is_available() else 32

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

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text

def run_asr(waveform, sampling_rate=16000):
    try:
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)  # mono if stereo
        if sampling_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sampling_rate, 16000)

        input_values = asr_processor(waveform, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            logits = asr_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = asr_processor.batch_decode(predicted_ids)[0]
        return text
    except Exception:
        # fallback: empty string if ASR fails
        return ""

def load_waveform(audio_data):
    if audio_data is None:
        raise ValueError("Missing audio data")

    if "array" in audio_data and audio_data["array"] is not None:
        waveform = torch.tensor(audio_data["array"], dtype=torch.float32)
        sampling_rate = audio_data.get("sampling_rate", ASR_SAMPLE_RATE)
    elif "bytes" in audio_data and audio_data["bytes"] is not None:
        waveform, sampling_rate = sf.read(io.BytesIO(audio_data["bytes"]), dtype="float32")
        waveform = torch.tensor(waveform, dtype=torch.float32)
    elif "path" in audio_data and audio_data["path"]:
        audio_path = audio_data["path"]
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            waveform, sampling_rate = torchaudio.load(audio_path)
        except Exception:
            waveform, sampling_rate = sf.read(audio_path, dtype="float32")
            waveform = torch.tensor(waveform, dtype=torch.float32)
    else:
        raise ValueError("Audio entry has neither array, bytes, nor path")

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    if sampling_rate != ASR_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sampling_rate, ASR_SAMPLE_RATE)

    return waveform

def add_transcription(example):
    existing_text = example.get("text")
    if existing_text:
        example["text"] = normalize_text(str(existing_text))
        return example

    try:
        waveform = load_waveform(example["audio"])
        example["text"] = normalize_text(run_asr(waveform))
    except Exception:
        example["text"] = ""

    return example

def prepare_text_dataset(dataset, transcribe=False):
    dataset = dataset.cast_column("audio", Audio(sampling_rate=ASR_SAMPLE_RATE, decode=True))
    if transcribe:
        print("Precomputing ASR transcripts...")
        dataset = dataset.map(add_transcription)
    return dataset


def filter_existing_audio(dataset, dataset_name):
    if "audio" not in dataset.column_names:
        return dataset

    before = len(dataset)

    def has_audio_source(example):
        audio = example.get("audio")
        if audio is None:
            return False
        if audio.get("array") is not None or audio.get("bytes") is not None:
            return True
        audio_path = audio.get("path")
        return bool(audio_path) and os.path.exists(audio_path)

    dataset = dataset.filter(has_audio_source)
    after = len(dataset)

    if after != before:
        print(f"[DATA] {dataset_name}: kept {after}/{before} rows with accessible audio")

    return dataset

def build_audio_attention_mask(audios):
    max_length = max(audio.size(0) for audio in audios)
    attention_mask = torch.zeros(len(audios), max_length, dtype=torch.long)

    for index, audio in enumerate(audios):
        attention_mask[index, :audio.size(0)] = 1

    return attention_mask


# def speech_text_collate_fn(batch):
#     audios, labels, texts = [], [], []
#     skipped = 0
#
#     for idx, item in enumerate(batch):
#         try:
#             waveform = load_waveform(item["audio"])
#
#             if waveform is None or waveform.numel() == 0:
#                 print(f"[COLLATE][TEXT] Empty waveform at index {idx}")
#                 skipped += 1
#                 continue
#
#             label = item.get("label", None)
#             if label is None:
#                 print(f"[COLLATE][TEXT] Missing label at index {idx}")
#                 skipped += 1
#                 continue
#
#             text = str(item.get("text", ""))
#
#             audios.append(waveform)
#             labels.append(torch.tensor(label, dtype=torch.long))
#             texts.append(text)
#
#         except Exception as e:
#             print(f"[COLLATE][TEXT] Error at index {idx}: {e}")
#             skipped += 1
#
#     if skipped > 0:
#         print(f"[COLLATE][TEXT] Skipped {skipped}/{len(batch)} items")
#
#     if len(audios) == 0:
#         print("[COLLATE][TEXT] Entire batch skipped!")
#         return None
#
#     tokenized = tokenizer(
#         texts,
#         padding=True,
#         truncation=True,
#         return_tensors="pt"
#     )
#
#     audio_batch = pad_sequence(audios, batch_first=True)
#     audio_attention_mask = build_audio_attention_mask(audios)
#
#     return {
#         "audio": audio_batch,
#         "audio_attention_mask": audio_attention_mask,
#         "labels": torch.stack(labels),
#         "input_ids": tokenized["input_ids"],
#         "attention_mask": tokenized["attention_mask"]
#     }

def speech_collate_fn(batch):
    audios, labels = [], []
    skipped = 0

    for idx, item in enumerate(batch):
        try:
            waveform = load_waveform(item["audio"])

            if waveform is None or waveform.numel() == 0:
                print(f"[COLLATE][AUDIO] Empty waveform at index {idx}")
                skipped += 1
                continue

            label = item.get("label", None)
            if label is None:
                print(f"[COLLATE][AUDIO] Missing label at index {idx}")
                skipped += 1
                continue

            audios.append(waveform)
            labels.append(torch.tensor(label, dtype=torch.long))

        except Exception as e:
            print(f"[COLLATE][AUDIO] Error at index {idx}: {e}")
            skipped += 1
            
    if skipped > 0:
        print(f"[COLLATE][AUDIO] Skipped {skipped}/{len(batch)} items")

    if len(audios) == 0:
        print("[COLLATE][AUDIO] Entire batch skipped!")
        return None

    audio_batch = pad_sequence(audios, batch_first=True)
    audio_attention_mask = build_audio_attention_mask(audios)

    return {
        "audio": audio_batch,
        "audio_attention_mask": audio_attention_mask,
        "labels": torch.stack(labels)
    }

def label_to_tensor(example, label_field="emotional_state"):
    raw_label = str(example.get(label_field, "")).lower().strip()
    standard_label = emotion_map.get(raw_label)
    example["label"] = target_emotions.index(standard_label) if standard_label in target_emotions else -1
    return example


def summarize_labels(dataset, label_column_name, dataset_name="dataset"):
    counts = {}
    missing = 0

    for raw_label in dataset[label_column_name]:
        normalized = emotion_map.get(str(raw_label).lower().strip())
        if normalized is None:
            missing += 1
            continue
        counts[normalized] = counts.get(normalized, 0) + 1

    if counts:
        ordered = ", ".join(
            f"{emotion}={counts.get(emotion, 0)}" for emotion in target_emotions if counts.get(emotion, 0) > 0
        )
        print(f"[DATA] {dataset_name}: retained labels -> {ordered}")
    else:
        print(f"[DATA] {dataset_name}: no supported labels found")

    if missing > 0:
        print(f"[DATA] {dataset_name}: filtered {missing} rows with unsupported labels from '{label_column_name}'")


def processing(dataset, label_column_name, use_text=False):
    def standardize_label(example):
        raw_label = str(example.get(label_column_name, "")).lower().strip()
        return {"label": emotion_map.get(raw_label)}

    initial_count = len(dataset)
    summarize_labels(dataset, label_column_name)
    dataset = dataset.map(standardize_label, remove_columns=[])
    dataset = dataset.filter(lambda x: x["label"] is not None)

    filtered_count = len(dataset)
    print(f"[DATA] {label_column_name}: kept {filtered_count}/{initial_count} rows after label normalization")

    if filtered_count == 0:
        print(f"[DATA] {label_column_name}: no rows left after filtering")
        return None, None

    dataset = filter_existing_audio(dataset, label_column_name)
    if len(dataset) == 0:
        print(f"[DATA] {label_column_name}: no rows left after audio validation")
        return None, None

    dataset = dataset.cast_column("label", shared_emotions)
    if use_text:
        dataset = prepare_text_dataset(dataset, transcribe="text" not in dataset.column_names)
    else:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=ASR_SAMPLE_RATE, decode=True))

    split = dataset.train_test_split(test_size=0.2, seed=42)
    print(
        f"[DATA] {label_column_name}: train={len(split['train'])}, test={len(split['test'])}, "
        f"use_text={use_text}"
    )
    '''speech_text_collate_fn if use_text else'''
    collate_fn = speech_collate_fn

    train_loader = DataLoader(split["train"], batch_size=DEFAULT_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4) \
        if len(split["train"]) > 0 else None
    test_loader = DataLoader(split["test"], batch_size=DEFAULT_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4) \
        if len(split["test"]) > 0 else None

    return train_loader, test_loader

def safe_processing(dataset, label_column, use_text=False):
    try:
        return processing(dataset, label_column, use_text=use_text)
    except Exception as e:
        print(f"Skipping dataset due to error: {e}")
        return None, None

def get_data():
    datasets_dict = {"train": {}, "test": {}}

    # Japanese
    jap = load_dataset("asahi417/jvnv-emotional-speech-corpus", split="test")

    jap = jap.cast_column("audio", Audio(sampling_rate=ASR_SAMPLE_RATE, decode=True))

    def map_japanese_label(example):
        raw_label = example.get("style", "neutral").lower().strip()
        standard_label = emotion_map.get(raw_label, "neutral")
        example["label"] = target_emotions.index(standard_label)
        return example

    jap = jap.map(map_japanese_label)

    jap = jap.filter(lambda x: x["audio"] is not None and x["audio"]["array"] is not None)
    jap = prepare_text_dataset(jap, transcribe=True)

    split = jap.train_test_split(test_size=0.2, seed=42)
    train_loader = DataLoader(
        split["train"],
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        # _text
        collate_fn=speech_collate_fn
    )
    test_loader = DataLoader(
        split["test"],
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        collate_fn=speech_collate_fn #_text
    )

    datasets_dict["train"]["japanese"] = train_loader
    datasets_dict["test"]["japanese"] = test_loader


    # Bangla
    bangla = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus/resolve/main/train.jsonl",
        split="train"
    ).select_columns(["path", "emotional_state"])
    bangla = bangla.map(lambda x: {"audio": {"path": os.path.join(SUBESCO_DIR, os.path.basename(x["path"]))}})
    bangla = bangla.map(label_to_tensor)
    bangla = bangla.filter(lambda x: x["label"] >= 0)
    bangla = filter_existing_audio(bangla, "bangla")
    datasets_dict["train"]["bangla"], datasets_dict["test"]["bangla"] = processing(bangla, "emotional_state", use_text=True)

    # Chinese
    ch = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    label_class = ch.features["label"]
    ch = ch.map(lambda x: {"label_str": label_class.int2str(x["label"])})
    datasets_dict["train"]["chinese"], datasets_dict["test"]["chinese"] = processing(ch, "label_str", use_text=True)

    # English
    eng = load_dataset("En1gma02/english_emotions", split="train")
    datasets_dict["train"]["english"], datasets_dict["test"]["english"] = processing(eng, "style", use_text=True)

    # Spanish
    span_path = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    span = load_dataset("audiofolder", data_dir=span_path, split="train")
    span = span.cast_column("audio", Audio(sampling_rate=16000, decode=True))

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
    datasets_dict["train"]["spanish"], datasets_dict["test"]["spanish"] = safe_processing(span, "label", use_text=True)

    # Arabic
    ara_path = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")
    ara = load_dataset("audiofolder", data_dir=ara_path, split="train")
    ara = ara.cast_column("audio", Audio(sampling_rate=16000, decode=True))

    def extract_arabic_label(x):
        audio_path = x["audio"]["path"] if "audio" in x and "path" in x["audio"] else None
        if not audio_path:
            return {"label": None}
        folder_name = os.path.basename(os.path.dirname(audio_path))
        return {"label": emotion_map.get(folder_name.lower().strip(), None)}

    ara = ara.map(extract_arabic_label)
    ara = ara.filter(lambda x: x["label"] is not None)
    datasets_dict["train"]["arabic"], datasets_dict["test"]["arabic"] = safe_processing(ara, "label", use_text=True)

    return datasets_dict
