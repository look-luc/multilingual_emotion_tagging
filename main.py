import sys
from data import get_data
# from .custom_model import model

def main():
    dataset = get_data.get_data()
    for batch_idx, batch in enumerate(dataset['train']['japanese']):
        if batch is None or (isinstance(batch, tuple) and batch[0].numel() == 0):
            print(f"Skipping empty/corrupted batch at index {batch_idx}")
            continue

        features, labels = batch
        print(f"{batch_idx}: feature {features}, label {labels}")
    return 0

if __name__ == '__main__':
    main()
    sys.exit(0)
