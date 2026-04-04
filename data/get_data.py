import kagglehub
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset, random_split

def get_data():
    japanese_train = load_dataset(
        "asahi417/jvnv-emotional-speech-corpus",
        split="test"
    ).select_columns(
        ["audio","style"]
    )
    japanese_train = japanese_train.with_format("torch")
    jap_train_size = int(0.8 * len(japanese_train))
    jap_test_size = len(japanese_train) - jap_train_size
    jap_train, jap_test = random_split(
        japanese_train, [jap_train_size, jap_test_size], generator=torch.Generator().manual_seed(42)
    )
    jap_train = DataLoader(jap_train, batch_size=64, shuffle=True, num_workers=0)
    jap_test = DataLoader(jap_test, batch_size=64, shuffle=False, num_workers=0)

    bangla_train = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus/resolve/main/train.jsonl",
        split="train"
    ).select_columns(["path", "emotional_state"])
    bangla_train = bangla_train.with_format("torch")
    ban_train_size = int(0.8 * len(bangla_train))
    ban_test_size = len(japanese_train) - ban_train_size
    ban_train, ban_test = random_split(
        bangla_train, [ban_train_size, ban_test_size], generator=torch.Generator().manual_seed(42)
    )
    ban_train = DataLoader(ban_train, batch_size=64, shuffle=True, num_workers=0)
    ban_test = DataLoader(ban_test, batch_size=64, shuffle=False, num_workers=0)

    chinese_train = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    chinese_train = chinese_train.with_format("torch")
    ch_train_size = int(0.8*len(chinese_train))
    ch_test_size = len(chinese_train) - ch_train_size
    ch_train, ch_test = random_split(
        chinese_train, [ch_train_size, ch_test_size], generator=torch.Generator().manual_seed(42)
    )
    ch_train = DataLoader(ch_train, batch_size=64, shuffle=True, num_workers=0)
    ch_test = DataLoader(ch_test, batch_size=64, shuffle=False, num_workers=0)

    english_train = load_dataset("En1gma02/english_emotions", split="train").select_columns(["audio", "style"])
    english_train = english_train.with_format("torch")
    english_train_size = int(0.8*len(english_train))
    english_test_size = len(english_train) - english_train_size
    english_train, english_test = random_split(
        english_train,
        [english_train_size, english_test_size],
        generator=torch.Generator().manual_seed(42)
    )
    eng_train = DataLoader(english_train, batch_size=64, shuffle=True, num_workers=0)
    eng_test = DataLoader(english_train, batch_size=64, shuffle=False, num_workers=0)

    spanish_path = kagglehub.dataset_download("angeluxarmenta/ses-sd", target_format="huggingface")
    spanish = load_from_disk(spanish_path)
    spanish = spanish.with_format("torch")
    # spanish_size = int(0.8*len(spanish))
    # span_train, span_test = random_split(
    #     spanish,
    #     [spanish_size, len(spanish)],
    #     generator=torch.Generator().manual_seed(42)
    # )
    # spanish_train = DataLoader(span_train, batch_size=64, shuffle=True, num_workers=0)
    # spanish_test = DataLoader(span_test, batch_size=64, shuffle=False, num_workers=0)

    print(spanish)
    # span_train, span_test = DataLoader(spanish, batch_size=64, shuffle=True, num_workers=0)
    arabic = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset", target_format="huggingface")
    print(arabic)

    return japanese_train, bangla_train, chinese_train, english_train, spanish, arabic
