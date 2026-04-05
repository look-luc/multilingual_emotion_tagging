import sys
from data import get_data
# from .custom_model import model

def main():
    dataset = get_data.get_data()
    print(dataset['train']['english'].value())
    return 0

if __name__ == '__main__':
    main()
    sys.exit(0)
