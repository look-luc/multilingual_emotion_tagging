import torch
import torchaudio
try:
    torchaudio.set_audio_backend("sox_io")
except:
    try:
        torchaudio.set_audio_backend("soundfile")
    except:
        pass
import os
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["datasets_audio_decoder_backend"] = "torchaudio"
import io
import kagglehub
from datasets import load_dataset, ClassLabel, Audio
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

target_emotions = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]
shared_emotions = ClassLabel(names=target_emotions)


def speech_collate_fn(batch):
    processed_audio, processed_labels = [], []

    # Priority check for the standardized 'label' key
    if "label" in batch[0]:
        label_key = "label"
    else:
        # Fallback for datasets not yet mapped (like 'style' in Japanese)
        possible_keys = ["style", "emotional_state", "emotion_label"]
        label_key = next((k for k in possible_keys if k in batch[0]), None)

    for item in batch:
        try:
            audio_data = item.get("audio")
            audio_tensor = None
            if isinstance(audio_data, dict):
                if audio_data.get("array") is not None:
                    audio_tensor = torch.as_tensor(audio_data["array"], dtype=torch.float32).squeeze()
                elif "bytes" in audio_data and audio_data["bytes"]:
                    encoded_audio = io.BytesIO(audio_data["bytes"])
                    waveform, _ = torchaudio.load(encoded_audio)
                    audio_tensor = waveform.squeeze()

            if audio_tensor is None or audio_tensor.numel() == 0:
                continue
            label_idx = None
            if label_key:
                val = item.get(label_key)
                if isinstance(val, int):
                    label_idx = torch.tensor(val)
                elif isinstance(val, str):
                    try:
                        label_idx = torch.tensor(shared_emotions.str2int(val.lower().strip()))
                    except:
                        continue
            if label_idx is not None:
                processed_audio.append(audio_tensor)
                processed_labels.append(label_idx)

        except Exception:
            continue

    if not processed_audio:
        return torch.empty(0), torch.empty(0)
    features = pad_sequence(processed_audio, batch_first=True)
    labels = torch.stack(processed_labels)
    return features, labels

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
    jap_dataset = load_dataset("asahi417/jvnv-emotional-speech-corpus", split="test")
    jap_dataset = jap_dataset.map(lambda x: {"label": x["style"].lower().strip()})
    jap_dataset = jap_dataset.filter(lambda x: x["label"] in target_emotions)
    if len(jap_dataset) > 0:
        jap_dataset = jap_dataset.cast_column("label", shared_emotions)
        jap_dataset = jap_dataset.cast_column("audio", Audio(sampling_rate=16000, decode=True))
        train_size = int(0.8 * len(jap_dataset))
        test_size = len(jap_dataset) - train_size
        jap_train_split, jap_test_split = random_split(
            jap_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        jap_train = DataLoader(jap_train_split, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
        jap_test = DataLoader(jap_test_split, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)
    else:
        jap_train = jap_test = None

    # bangla_train = load_dataset(
    #     "json",
    #     data_files="https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus/resolve/main/train.jsonl",
    #     split="train"
    # ).select_columns(["path", "emotional_state"]).rename_column("path", "audio")
    # bangla_train = bangla_train.cast_column("audio", Audio(decode=True))
    # bangla_train = bangla_train.filter(is_audio_valid, load_from_cache_file=False)
    # bangla_train = bangla_train.cast_column("audio", Audio(decode=True))
    # bangla_train = bangla_train.map(lambda x: {
    #     "emotional_state": "angry" if x["emotional_state"].lower().strip() == "anger" else x[
    #         "emotional_state"].lower().strip()
    # })
    # bangla_train = bangla_train.cast_column("emotional_state", shared_emotions)
    # bangla_train = bangla_train.with_format("torch")
    # ban_train_size = int(0.8 * len(bangla_train))
    # ban_test_size = len(bangla_train) - ban_train_size
    # ban_train, ban_test = random_split(
    #     bangla_train, [ban_train_size, ban_test_size], generator=torch.Generator().manual_seed(42)
    # )
    # ban_train = DataLoader(ban_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=speech_collate_fn)
    # ban_test = DataLoader(ban_test, batch_size=64, shuffle=False, num_workers=0, collate_fn=speech_collate_fn)

    chinese_train = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    chinese_train = chinese_train.cast_column("audio", Audio(decode=True))
    def fix_chinese_labels(example):
        lbl_name = chinese_train.features["label"].int2str(example["label"]).lower().strip()
        if lbl_name == "surprised": lbl_name = "surprise"
        return {"label": lbl_name}
    chinese_train = chinese_train.map(fix_chinese_labels)
    chinese_train = chinese_train.cast_column("label", shared_emotions)
    ch_train_size = int(0.8 * len(chinese_train))
    ch_train, ch_test = random_split(
        chinese_train,
        [ch_train_size, len(chinese_train) - ch_train_size],
        generator=torch.Generator().manual_seed(42)
    )
    ch_train = DataLoader(ch_train, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
    ch_test = DataLoader(ch_test, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)

    eng_dataset = load_dataset("En1gma02/english_emotions", split="train")
    eng_dataset = eng_dataset.cast_column("audio", Audio(decode=False))
    eng_dataset = eng_dataset.map(lambda x: {"label": x["style"].lower().strip()})
    eng_dataset = eng_dataset.filter(lambda x: x["label"] in target_emotions)
    eng_dataset = eng_dataset.cast_column("label", shared_emotions)
    eng_size = int(0.8 * len(eng_dataset))
    eng_train_split, eng_test_split = random_split(
        eng_dataset, [eng_size, len(eng_dataset) - eng_size],
        generator=torch.Generator().manual_seed(42)
    )
    eng_train = DataLoader(eng_train_split, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
    eng_test = DataLoader(eng_test_split, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)

    spanish_path = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    spanish = load_dataset("audiofolder", data_dir=spanish_path, split="train")
    spanish = spanish.cast_column("audio", Audio(decode=False))
    spanish_code_map = {
        "ang": "angry",
        "ale": "happy",
        "asc": "disgust",
        "mie": "fear",
        "neu": "neutral",
        "sor": "surprise",
        "tri": "sad"
    }
    def extract_label_from_path(example):
        path = example["audio"]["path"]
        filename = os.path.basename(path).lower()
        mapped_label = "unknown"
        for code, emotion in spanish_code_map.items():
            if code in filename:
                mapped_label = emotion
                break

        return {"label": mapped_label}
    spanish = spanish.map(extract_label_from_path)
    spanish = spanish.filter(lambda x: x["label"] in target_emotions)
    spanish_train = spanish.cast_column("audio", Audio(decode=True))
    spanish_train = spanish.cast_column("label", shared_emotions)
    if len(spanish) > 0:
        span_train_size = int(0.8 * len(spanish))
        span_train, span_test = random_split(
            spanish,
            [span_train_size, len(spanish) - span_train_size],
            generator=torch.Generator().manual_seed(42)
        )
        spanish_train = DataLoader(spanish_train, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
        spanish_test = DataLoader(span_test, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)
    else:
        raise ValueError("Spanish dataset is still empty. Ensure the codes 'ang', 'ale', etc., exist in the filenames.")

    arabic_path = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")
    real_data_dir = None
    for root, dirs, files in os.walk(arabic_path):
        if any(f.endswith('.wav') for f in files):
            real_data_dir = os.path.dirname(root)
            break
    if real_data_dir is None: real_data_dir = arabic_path
    arabic = load_dataset("audiofolder", data_dir=real_data_dir, split="train")
    arabic = arabic.cast_column("audio", Audio(decode=False))
    def add_label_column(x):
        audio_info = x.get("audio")
        if isinstance(audio_info, dict) and "path" in audio_info:
            folder_name = os.path.basename(os.path.dirname(audio_info["path"]))
        else:
            folder_name = "unknown"
        return {"label": folder_name}
    if "label" not in arabic.column_names:
        arabic = arabic.map(add_label_column)
    arabic_norm = {
        "anger": "angry", "0": "angry", "happiness": "happy", "1": "happy",
        "sadness": "sad", "2": "sad", "neutral": "neutral", "3": "neutral",
        "surprised": "surprise", "fearful": "fear", "disgusted": "disgust",
        "exhausted": "neutral"
    }
    def clean_arabic_labels(x):
        label_val = x.get("label")
        if isinstance(label_val, int):
            try:
                label_str = arabic.features["label"].int2str(label_val)
            except:
                label_str = str(label_val)
        else:
            label_str = str(label_val)
        lbl = label_str.lower().strip()
        return {"label": arabic_norm.get(lbl, lbl)}
    arabic = arabic.map(clean_arabic_labels)
    arabic = arabic.filter(lambda x: x["label"] in target_emotions)
    if len(arabic) > 0:
        arabic = arabic.cast_column("label", shared_emotions)
        arabic = arabic.cast_column("audio", Audio(decode=True))
        arabic_size = max(1, int(0.8 * len(arabic)))
        test_size = len(arabic) - arabic_size
        if test_size <= 0:
            arabic_size = len(arabic) - 1
            test_size = 1
        arabic_train_split, arabic_test_split = random_split(
            arabic, [arabic_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        arabic_train = DataLoader(arabic_train_split, batch_size=64, shuffle=True, collate_fn=speech_collate_fn)
        arabic_test = DataLoader(arabic_test_split, batch_size=64, shuffle=False, collate_fn=speech_collate_fn)
    else:
        print("Warning: Arabic dataset is empty after filtering.")
        arabic_train = arabic_test = None

    datasets = {
        "train": {
            "japanese": jap_train,
            "english": eng_train,
            # "bangla": ban_train,
            "spanish": spanish_train,
            "arabic": arabic_train,
            "chinese": ch_train
        },
        "test": {
            "japanese": jap_test,
            "english": eng_test,
#             "bangla": ban_test,
            "spanish": spanish_test,
            "arabic": arabic_test,
            "chinese": ch_test
        }
    }
    return datasets