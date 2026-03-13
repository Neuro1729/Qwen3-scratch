import torch
from transformers import AutoTokenizer
import os
import urllib.request

# ------------------------
# Download dataset if not exists
# ------------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"

if not os.path.exists(DATA_FILE):
    print("Downloading tiny-shakespeare dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print("Downloaded input.txt")

# ------------------------
# Load tokenizer
# ------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# ------------------------
# Minimal dataloader class
# ------------------------
class Dataloader:
    def __init__(self, B, T):
        """
        B: batch size
        T: sequence length
        """
        self.B = B
        self.T = T

        # ---- Read text ----
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            text = f.read()

        # ---- Tokenize entire text ----
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]  # 1D tensor
        self.tokens = tokens
        self.curr_pos = 0
        self.N = len(self.tokens)  # total number of tokens

    def next_batch(self):
        B, T = self.B, self.T
        end_pos = self.curr_pos + B*T + 1

        # ---- Wrap around if we reach end of dataset ----
        if end_pos > self.N:
            self.curr_pos = 0
            end_pos = B*T + 1

        buf = self.tokens[self.curr_pos:end_pos]

        # ---- Input x and target y ----
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # ---- Advance position for next batch ----
        self.curr_pos += B*T

        return x, y
