import torch
import torch.optim as optim
import time
from model import TransformerModel
from dataloader import Dataloader
from dataloader import tokenizer  

# ------------------------
# User-configurable hyperparameters
# ------------------------
BATCH_SIZE = 4
SEQ_LEN = 128
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 0.01

# ------------------------
# Initialize model
# ------------------------
model = TransformerModel(
    vocab_size=tokenizer.vocab_size,  # vocab size from tokenizer
    hidden_dim=1024,
    q_dim=2048,
    num_heads_q=16,
    num_heads_kv=8,
    mlp_dim=3072,
    num_layers=28,
    tie_embeddings=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.train()

# Optional: compile for faster training (PyTorch 2.0+)
model = torch.compile(model)

# ------------------------
# Initialize dataloader
# ------------------------
train_loader = Dataloader(B=BATCH_SIZE, T=SEQ_LEN)

# ------------------------
# Optimizer
# ------------------------
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ------------------------
# Training loop
# ------------------------
for epoch in range(NUM_EPOCHS):
    for step in range(50):  # or you can compute steps per epoch based on dataset
        t0 = time.time()

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.time()

        dt = (t1 - t0) * 1000  # ms per step
        tokens_per_sec = (BATCH_SIZE * SEQ_LEN) / (t1 - t0)

        print(f"Epoch {epoch+1} Step {step+1} | Loss: {loss.item():.4f} | dt: {dt:.2f} ms | tok/sec: {tokens_per_sec:.2f}")

# ------------------------
# Save the trained model
# ------------------------
torch.save(model.state_dict(), "qwen3_minimal.pth")
print("Training complete, model saved as qwen3_minimal.pth")
