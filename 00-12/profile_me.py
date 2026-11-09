import time

def load_data():
    time.sleep(0.5)          # pretend: reading files from disk
    return list(range(100_000))

def forward(data):
    return [x * 2 for x in data]   # pretend: the model's math

def train():
    for epoch in range(3):
        data = load_data()
        forward(data)

train()
