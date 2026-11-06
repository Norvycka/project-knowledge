import random


def split_data(data, seed):
    items = list(data)  # make a copy, don't change the original
    random.Random(seed).shuffle(items)  # shuffle using this seed

    train_end = int(len(items) * 0.70)
    val_end = int(len(items) * 0.85)

    train = items[:train_end]
    val = items[train_end:val_end]
    test = items[val_end:]
    return train, val, test


data = list(range(100))
train, val, test = split_data(data, seed=7)
print(len(train), len(val), len(test))
print("first test items:", test[:5])
