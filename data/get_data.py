from datasets import load_dataset
import kagglehub

def get_data():
    japanese_train = load_dataset("asahi417/jvnv-emotional-speech-corpus",split="train")
    bangla_train = load_dataset("sustcsenlp/bn_emotion_speech_corpus",split="train")
    chinese_train = load_dataset("BillyLin/CASIA_speech_emotion_recognition", split="train")
    english_train = load_dataset("En1gma02/english_emotions", split="train")
    spanish = kagglehub.dataset_download("angeluxarmenta/ses-sd")
    arabic = kagglehub.dataset_download("a13x10/basic-arabic-vocal-emotions-dataset")

    return japanese_train, bangla_train, chinese_train, english_train, spanish, arabic