import sys
from data import get_data
# from .custom_model import model

def main():
    dataset = get_data.get_data()
    for batch_idx, (features, labels) in enumerate(dataset['train']['english']):
        print(f"{batch_idx}: feature {features}, label {labels}")
    return 0

if __name__ == '__main__':
    main()
    sys.exit(0)
