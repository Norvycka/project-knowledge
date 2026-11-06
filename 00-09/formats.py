import os

from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb", split="train")
print(dataset)
print(dataset[0]["text"][:100])

os.makedirs("00-09/data", exist_ok=True)
dataset.to_csv("00-09/data/imdb.csv")
dataset.to_parquet("00-09/data/imdb.parquet")

csv_mb = os.path.getsize("00-09/data/imdb.csv") / 1e6
parquet_mb = os.path.getsize("00-09/data/imdb.parquet") / 1e6
print(f"CSV: {csv_mb:.1f} MB | Parquet: {parquet_mb:.1f} MB")

split = dataset.train_test_split(test_size=0.2, seed=42)  # hold out 20%
print(len(split["train"]), len(split["test"]))
