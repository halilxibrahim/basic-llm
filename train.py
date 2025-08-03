# train.py

import torch
import torch.nn as nn
from torch.nn import functional as F

# model.py dosyasındaki değişken ve fonksiyonları buraya taşıyalım veya import edelim.
# Bu örnekte daha basit olması için tekrar tanımlıyoruz.

# --- Hiperparametreler ---
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --------------------

torch.manual_seed(1337) # Tekrarlanabilirlik için

# --- Veri Yükleme ve Hazırlama ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Veri bloğu (batch) oluşturma
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# model.py'dan GPT modelini import edelim
from model import GPTLanguageModel 

model = GPTLanguageModel()
model.to(device) # Modeli GPU'ya veya CPU'ya taşı

# Optimizer'ı oluşturalım (AdamW, Transformer'lar için iyi bir seçimdir)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Eğitim döngüsü
for iter in range(max_iters):

    # Her 'eval_interval' adımda bir loss'u değerlendirelim
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Eğitim verisinden bir batch alalım
    xb, yb = get_batch('train')

    # loss'u hesaplayıp gradient'leri güncelleyelim
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # Gradient'leri sıfırla
    loss.backward()                       # Backpropagation
    optimizer.step()                      # Parametreleri güncelle

# Eğitilmiş modeli kaydet
torch.save(model.state_dict(), 'gpt_model.pth')
print("Model eğitimi tamamlandı ve 'gpt_model.pth' olarak kaydedildi.")