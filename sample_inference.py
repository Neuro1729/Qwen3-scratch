import torch
from model import TransformerModel
from dataloader import tokenizer

# ------------------------
# User-configurable parameters
# ------------------------
MODEL_PATH = None  # set path to trained model, or None to use untrained
PROMPT = "Hello who are you"
MAX_TOKENS = 50
NUM_SEQUENCES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Initialize model
# ------------------------
model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    hidden_dim=1024,
    q_dim=2048,
    num_heads_q=16,
    num_heads_kv=8,
    mlp_dim=3072,
    num_layers=28,
    tie_embeddings=True
)

# Load trained weights if provided
if MODEL_PATH is not None:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded trained model from {MODEL_PATH}")
else:
    print("No trained model provided, using untrained model")

model = model.to(DEVICE)
model.eval()

# ------------------------
# Text generation function
# ------------------------
def generate_multiple(model, tokenizer, prompt, max_tokens=50, num_sequences=5, device="cpu"):
    """
    Generate multiple sequences from a prompt using the model.
    """
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)  # [1, T]

    # Repeat prompt for batch
    generated_ids = input_ids.repeat(num_sequences, 1)  # [num_sequences, T]

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(generated_ids)  # [num_sequences, T, vocab_size]
            next_token_logits = logits[:, -1, :]  # [num_sequences, vocab_size]

            # Sampling for diversity
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1)  # [num_sequences, 1]

            # Append to generated sequences
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)  # [num_sequences, T+1]

    # Decode each sequence
    output_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids]
    return output_texts

# ------------------------
# Generate and print sequences
# ------------------------
outputs = generate_multiple(model, tokenizer, PROMPT, max_tokens=MAX_TOKENS,
                            num_sequences=NUM_SEQUENCES, device=DEVICE)

for i, text in enumerate(outputs):
    print(f"Sequence {i+1}:", text)
