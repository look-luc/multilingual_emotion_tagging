import sys
from data import get_data
# from .custom_model import model

def main():
    japanese_train, bangla_train, chinese_train, english_train, spanish, arabic = get_data.get_data()
    return 0

if __name__ == '__main__':
    main()
    sys.exit(0)
