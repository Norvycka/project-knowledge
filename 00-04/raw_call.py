import os
import json
import urllib.request
import urllib.error

def load_env(path):
    with open(path) as f:                # open the file for reading
        for line in f:                   # go through it line by line
            line = line.strip()          # remove the newline/whitespace at the ends
            if not line or line.startswith("#"):
                continue                 # skip empty lines and comments
            name, value = line.split("=", 1)  # split "NAME=value" into two parts at the "="
            os.environ[name] = value     # put it into the environment

load_env(".env")
#print("Key loaded, starts with:", os.environ["ANTHROPIC_API_KEY"][:10])


# 1. ENDPOINT: where to send it
url = "https://api.anthropic.com/v1/messages"

# 2. HEADERS: metadata about the request, including the KEY
headers = {
    "Content-Type": "application/json",          # "the body is JSON"
    "x-api-key":    os.environ["ANTHROPIC_API_KEY"],            # your key, read from the environment
    "anthropic-version": "2023-06-01",           # which version of the API format
}

# 3. REQUEST BODY: what you want
body = json.dumps({
    "model": "claude-haiku-5-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "whats your purpose?"}],   # your question, as a string
}).encode()

req = urllib.request.Request(url, data=body, headers=headers, method="POST")

# 4. RESPONSE: what you get back (or the error)
try:
    with urllib.request.urlopen(req) as resp:
        print(json.loads(resp.read()))
except urllib.error.HTTPError as e:
    print("Status code:", e.code)
    print("Error body:", e.read().decode())



