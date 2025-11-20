print("Importing Dependencies")
from datasets import load_dataset
import logging

splits = ['train', 'validation', 'test']
path = 'data/wrime/'

dataset = load_dataset("shunk031/wrime", name="ver1", trust_remote_code=True)

for split in splits:
    df = dataset[split].to_pandas()
    df.to_csv(f"{path}{split}.csv", index=False)
    logging.info("Saved")
