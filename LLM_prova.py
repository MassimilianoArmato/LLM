
# -------------------------------
# 1. Vocabolario manuale
# -------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence

# -------------------------------
# 1. Tokenizer GPT2
# -------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

# -------------------------------
# 2. Encoding
# -------------------------------
def encode_sequence(text):
    return tokenizer(text, return_tensors="pt").input_ids.squeeze()

# -------------------------------
# 3. Embedding Layer
# -------------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids):
        return self.embedding(token_ids)

# -------------------------------
# 4. Positional Encoding
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)

# -------------------------------
# 5. Masked Multi-Head Attention
# -------------------------------
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        return self.attention(x, x, x, attn_mask=mask)[0]

# -------------------------------
# 6. Feed Forward Network
# -------------------------------
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# 7. Blocco Transformer
# -------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = MaskedMultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

# -------------------------------
# 8. Modello LLM Minimale
# -------------------------------
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional = PositionalEncoding(embedding_dim, max_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = self.positional(x)
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.output_layer(x)
        return logits

# -------------------------------
# 9. Addestramento
# -------------------------------
if __name__ == "__main__":
    # Parametri
    embedding_dim = 32
    num_heads = 2
    hidden_dim = 64
    num_layers = 1
    max_len = 128

    model = MiniLLM(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dataset CommonsenseQA
    dataset = load_dataset("commonsense_qa", split="train")

    examples = []
    for item in dataset.select(range(1000)):
        question = item["question"]
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_index = labels.index(item["answerKey"])
        answer = texts[answer_index]

        examples.append((question, answer + tokenizer.eos_token))

    inputs, targets = [], []
    for q, a in examples:
        input_ids = tokenizer(q, return_tensors="pt", truncation=True, max_length=50).input_ids.squeeze()
        target_ids = tokenizer(a, return_tensors="pt", truncation=True, max_length=10).input_ids.squeeze()
        inputs.append(input_ids)
        targets.append(target_ids)

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Training
    model.train()
    num_epochs = 80

    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        for input_ids, target_ids in zip(inputs, targets):
            input_ids = input_ids.unsqueeze(0)
            target_ids = target_ids.unsqueeze(0)

            optimizer.zero_grad()
            logits = model(input_ids)
            last_logits = logits[:, -1, :]
            target_token = target_ids[:, 0]
            loss = criterion(last_logits, target_token)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            tqdm.write(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    # Salvataggio
    torch.save(model.state_dict(), "mini_llm.pth")
    tokenizer.save_pretrained("tokenizer_gpt2")

# -------------------------------
# 10. Inferenza
# -------------------------------
def generate(model, prompt, tokenizer, max_tokens=5):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            if next_token_id == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]])], dim=1)

    new_tokens = generated_ids[0, input_ids.size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

