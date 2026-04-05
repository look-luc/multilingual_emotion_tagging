import kagglehub
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence

from datasets import ClassLabel, Features, Audio

target_emotions = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]
shared_emotions = ClassLabel(names=target_emotions)

def speech_collate_fn(batch):
    audio_tensors = [torch.tensor(item["audio"]["array"]).squeeze() for item in batch]
    audio_padded = pad_sequence(audio_tensors, batch_first=True)
    label_key = next(k for k in batch[0].keys() if k != "audio")
    labels = torch.tensor([item[label_key] for item in batch])

    return audio_padded, labels

def get_data():
    japanese_train = load_dataset(
        "asahi417/jvnv-emotional-speech-corpus",
        split="test"
    )
    japanese_train = japanese_train.with_format("torch", columns=["audio","style"])
    jap_train_size = int(0.8 * len(japanese_train))
    jap_test_size = len(japanese_train) - jap_train_size
    jap_train, jap_test = random_split(
        japanese_train, [jap_train_size, jap_test_size], generator=torch.Generator().manual_seed(42)
    )
    jap_train = DataLoader(jap_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=speech_collate_fn)
    jap_test = DataLoader(jap_test, batch_size=64, shuffle=False, num_workers=0, collate_fn=speech_collate_fn)

    bangla_train = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus/resolve/main/train.jsonl",
        split="train"
    ).select_columns(["path", "emotional_state"])
    bangla_train = bangla_train.with_format("torch")
    bangla_train = bangla_train.map(lambda x: {
        "emotional_state": x["emotional_state"].lower().strip()
    })
    bangla_train = bangla_train.map(lambda x: {
        "emotional_state": "angry" if x["emotional_state"] == "anger" else x["emotional_state"]
    })
    bangla_train = bangla_train.cast_column("emotional_state", shared_emotions)
    ban_train_size = int(0.8 * len(bangla_train))
    ban_test_size = len(bangla_train) - ban_train_size
    ban_train, ban_test = random_split(
        bangla_train, [ban_train_size, ban_test_size], generator=torch.Generator().manual_seed(42)
    )
    ban_train = DataLoader(ban_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=speech_collate_fn)
    ban_test = DataLoader(ban_test, batch_size=64, shuffle=False, num_workers=0, collate_fn=speech_collate_fn)

    chinese_train = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    chinese_train = chinese_train.with_format("torch")
    ch_train_size = int(0.8*len(chinese_train))
    ch_test_size = len(chinese_train) - ch_train_size
    ch_train, ch_test = random_split(
        chinese_train, [ch_train_size, ch_test_size], generator=torch.Generator().manual_seed(42)
    )
    ch_train = DataLoader(ch_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=speech_collate_fn)
    ch_test = DataLoader(ch_test, batch_size=64, shuffle=False, num_workers=0, collate_fn=speech_collate_fn)

    english_train = load_dataset("En1gma02/english_emotions", split="train")
    english_train = english_train.select_columns(["audio", "style"])

    valid_indices = []
    for i, label in enumerate(english_train["style"]):
        if label.lower().strip() in target_emotions:
            valid_indices.append(i)

    english_train = english_train.select(valid_indices)
    english_train = english_train.map(lambda x: {"style": x["style"].lower().strip()})
    english_train = english_train.cast_column("style", shared_emotions)
    english_train = english_train.with_format("torch")
    english_train_size = int(0.8*len(english_train))
    english_test_size = len(english_train) - english_train_size
    english_train, english_test = random_split(
        english_train,
        [english_train_size, english_test_size],
        generator=torch.Generator().manual_seed(42)
    )
    eng_train = DataLoader(english_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=speech_collate_fn)
    eng_test = DataLoader(english_train, batch_size=64, shuffle=False, num_workers=0, collate_fn=speech_collate_fn)

    spanish_path = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    spanish = load_dataset("audiofolder", data_dir=spanish_path, split="train")
    spanish = spanish.with_format("torch")
    spanish_size = int(0.8*len(spanish))
    span_train, span_test = random_split(
        spanish,
        [spanish_size, len(spanish)-spanish_size],
        generator=torch.Generator().manual_seed(42)
    )

    spanish_train = DataLoader(span_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=speech_collate_fn)
    spanish_test = DataLoader(span_test, batch_size=64, shuffle=False, num_workers=0, collate_fn=speech_collate_fn)

    arabic_path = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")
    arabic = load_dataset("audiofolder", data_dir=arabic_path, split="train")
    arabic = arabic.with_format("torch")
    arabic_size = int(0.8*len(arabic))
    arabic_train, arabic_test = random_split(
        arabic,
        [arabic_size, len(arabic)-arabic_size],
        generator=torch.Generator().manual_seed(42)
    )
    arabic_train = DataLoader(arabic_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=speech_collate_fn)
    arabic_test = DataLoader(arabic_test, batch_size=64, shuffle=False, num_workers=0, collate_fn=speech_collate_fn)

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