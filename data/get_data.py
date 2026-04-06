import torch
import torchaudio
import io
import kagglehub
from datasets import load_dataset, ClassLabel, Audio
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

target_emotions = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]
shared_emotions = ClassLabel(names=target_emotions)


def speech_collate_fn(batch):
    processed_audio, processed_labels = [], []

    possible_keys = ["style", "emotional_state", "label"]
    label_key = next((k for k in possible_keys if k in batch[0]), None)
    if label_key is None:
        label_key = next((k for k in batch[0].keys() if k != "audio"), None)

    for item in batch:
        audio_tensor = None
        try:
            if isinstance(item["audio"], dict):
                if "array" in item["audio"] and item["audio"]["array"] is not None:
                    audio_tensor = torch.as_tensor(item["audio"]["array"]).squeeze()
                elif "path" in item["audio"]:
                    waveform, sample_rate = torchaudio.load(item["audio"]["path"])
                    audio_tensor = waveform.squeeze()

            elif isinstance(item["audio"], torch.Tensor):
                audio_tensor = item["audio"].squeeze()

            if audio_tensor is None:
                continue

            processed_audio.append(audio_tensor)
            if label_key is not None and label_key in item:
                val = item[label_key]
                if isinstance(val, str):
                    processed_labels.append(shared_emotions.str2int(val.lower().strip()))
                else:
                    processed_labels.append(int(val))
            else:
                processed_labels.append(-1)

        except Exception as e:
            print(f"Skipping corrupted sample: {e}")
            continue

    if not processed_audio:
        return torch.empty(0), torch.empty(0)
    audio_padded = pad_sequence(processed_audio, batch_first=True)
    labels = torch.as_tensor(processed_labels, dtype=torch.long)
    return audio_padded, labels

def is_audio_valid(example):
    try:
        audio_data = example.get("audio")
        if not audio_data:
            return False
        if audio_data.get("bytes") is not None:
            return len(audio_data["bytes"]) > 0
        if audio_data.get("path") is not None:
            return len(str(audio_data["path"])) > 0

        return False
    except Exception:
        return False

def encode_labels(example):
    return {"style": shared_emotions.str2int(example["style"].lower().strip())}

def get_data():
    japanese_train = load_dataset("asahi417/jvnv-emotional-speech-corpus", split="test")
    japanese_train = japanese_train.cast_column("audio", Audio(decode=True))
    jap_train_size = int(0.8 * len(japanese_train))
    jap_train, jap_test = random_split(
        japanese_train, [jap_train_size, len(japanese_train) - jap_train_size],
        generator=torch.Generator().manual_seed(42)
    )
    jap_train = DataLoader(jap_train, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
    jap_test = DataLoader(jap_test, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)

    bangla_train = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus/resolve/main/train.jsonl",
        split="train"
    ).select_columns(["path", "emotional_state"]).rename_column("path", "audio")
    bangla_train = bangla_train.cast_column("audio", Audio(decode=False))
    bangla_train = bangla_train.filter(is_audio_valid, load_from_cache_file=False)
    bangla_train = bangla_train.cast_column("audio", Audio(decode=True))
    bangla_train = bangla_train.map(lambda x: {
        "emotional_state": "angry" if x["emotional_state"].lower().strip() == "anger" else x[
            "emotional_state"].lower().strip()
    })
    bangla_train = bangla_train.cast_column("emotional_state", shared_emotions)
    bangla_train = bangla_train.with_format("torch")
    ban_train_size = int(0.8 * len(bangla_train))
    ban_test_size = len(bangla_train) - ban_train_size
    ban_train, ban_test = random_split(
        bangla_train, [ban_train_size, ban_test_size], generator=torch.Generator().manual_seed(42)
    )
    ban_train = DataLoader(ban_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=speech_collate_fn)
    ban_test = DataLoader(ban_test, batch_size=64, shuffle=False, num_workers=0, collate_fn=speech_collate_fn)

    chinese_train = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    chinese_train = chinese_train.cast_column("audio", Audio(decode=True))
    ch_train_size = int(0.8 * len(chinese_train))
    ch_train, ch_test = random_split(
        chinese_train, [ch_train_size, len(chinese_train) - ch_train_size],
        generator=torch.Generator().manual_seed(42)
    )
    ch_train = DataLoader(ch_train, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
    ch_test = DataLoader(ch_test, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)

    english_dataset = load_dataset("En1gma02/english_emotions", split="train")
    english_clean = english_dataset.cast_column("audio", Audio(decode=False))
    english_filtered = english_clean.filter(lambda x: str(x["style"]).lower().strip() in target_emotions)
    english_filtered = english_filtered.map(encode_labels)
    english_final = english_filtered.cast_column("audio", Audio(decode=True))
    eng_train_size = int(0.8 * len(english_final))
    eng_train_split, eng_test_split = random_split(
        english_final, [eng_train_size, len(english_final) - eng_train_size],
        generator=torch.Generator().manual_seed(42)
    )
    eng_train = DataLoader(eng_train_split, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
    eng_test = DataLoader(eng_test_split, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)

    spanish_path = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    spanish = load_dataset("audiofolder", data_dir=spanish_path, split="train")
    spanish = spanish.cast_column("audio", Audio(decode=True))
    span_train_size = int(0.8 * len(spanish))
    span_train, span_test = random_split(
        spanish, [span_train_size, len(spanish) - span_train_size],
        generator=torch.Generator().manual_seed(42)
    )
    spanish_train = DataLoader(span_train, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
    spanish_test = DataLoader(span_test, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)

    arabic_path = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")
    arabic = load_dataset("audiofolder", data_dir=arabic_path, split="train")
    arabic = arabic.cast_column("audio", Audio(decode=True))
    arabic_size = int(0.8 * len(arabic))
    arabic_train_split, arabic_test_split = random_split(
        arabic, [arabic_size, len(arabic) - arabic_size],
        generator=torch.Generator().manual_seed(42)
    )
    arabic_train = DataLoader(arabic_train_split, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
    arabic_test = DataLoader(arabic_test_split, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)

    datasets = {
        "train": {
            "japanese": jap_train,
            "english": eng_train,
            "bangla": ban_train,
            "spanish": spanish_train,
            "arabic": arabic_train,
            "chinese": ch_train
        },
        "test": {
            "japanese": jap_test,
            "english": eng_test,
            "bangla": ban_test,
            "spanish": spanish_test,
            "arabic": arabic_test,
            "chinese": ch_test
        }
    }
    return datasets