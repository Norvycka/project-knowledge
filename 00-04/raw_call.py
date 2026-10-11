import os

def load_env(path):
    with open(path) as f:                # open the file for reading
        for line in f:                   # go through it line by line
            line = line.strip()          # remove the newline/whitespace at the ends
            if not line or line.startswith("#"):
                continue                 # skip empty lines and comments
            name, value = line.split("=")  # split "NAME=value" into two parts at the "="
            os.environ[name] = value     # put it into the environment

load_env(".env")
print("Key loaded, starts with:", os.environ["ANTHROPIC_API_KEY"][:10])
