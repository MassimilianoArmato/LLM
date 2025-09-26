import torch
from transformers import GPT2Tokenizer
from LLM_prova import MiniLLM, generate
from datasets import load_dataset
import json

# Parametri modello (devono combaciare con quelli usati nel training)
embedding_dim = 32
num_heads = 2
hidden_dim = 64
num_layers = 1
max_len = 128
vocab_size = 50257

# Carica tokenizer e modello
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer_gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = MiniLLM(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len)
model.load_state_dict(torch.load("mini_llm.pth"))
model.eval()

# Prompt dinamico
print("Cosa vuoi sapere?")
domanda = "What is committing perjury likely to lead to?"
response = generate(model, domanda, tokenizer)
print("Risposta:", response)


# Carica il dataset CommonsenseQA
dataset = load_dataset("commonsense_qa", split="train")

data = []
for item in dataset.select(range(1500)):
    question = item["question"]
    labels = item["choices"]["label"]
    texts = item["choices"]["text"]
    answer_index = labels.index(item["answerKey"])
    answer = texts[answer_index]

    data.append({
        "domanda": question,
        "risposta": answer,
        "opzioni": texts
    })

with open("dataset_esplorato.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)