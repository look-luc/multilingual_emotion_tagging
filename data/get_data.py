from datasets import load_dataset
import kagglehub
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data():
    japanese_train = load_dataset("asahi417/jvnv-emotional-speech-corpus",split="test")
    print(f"japanese: {japanese_train}")
    bangla_train = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus/resolve/main/train.jsonl",
        split="train"
    )
    print(f"bangla: {bangla_train}")
    chinese_train = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    print(f"chinese: {chinese_train}")
    english_train = load_dataset("En1gma02/english_emotions", split="train")
    print(f"english: {english_train}")
    spanish = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    print(f"spanish: {spanish}")
    arabic = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")
    print(f"arabic: {arabic}")

    return japanese_train, bangla_train, chinese_train, english_train, spanish, arabic