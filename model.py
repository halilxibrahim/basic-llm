
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hiperparametreler (Ayarlanabilir Değerler) ---
batch_size = 64         # Paralel olarak işlenecek bağımsız sequence sayısı
block_size = 256        # Bir tahmin için kullanılacak maksimum context (bağlam) uzunluğu
max_iters = 5000        # Eğitim döngüsü sayısı
eval_interval = 500     # Her 'eval_interval' adımda bir loss'u değerlendir
learning_rate = 3e-4    # Öğrenme oranı (AdamW optimizer için genellikle iyi bir başlangıç)
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Varsa GPU kullan, yoksa CPU
eval_iters = 200        # Değerlendirme için kullanılacak batch sayısı
n_embd = 384            # Embedding boyutu (her token'ın vektör uzunluğu)
n_head = 6              # Self-Attention mekanizmasındaki kafa sayısı
n_layer = 6             # Transformer bloklarının (katmanlarının) sayısı
dropout = 0.2           # Eğitim sırasında bazı nöronları rastgele kapatarak ezberlemeyi önler
# ----------------------------------------------------

# --- Veri Yükleme ve Hazırlama ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Metindeki tüm benzersiz karakterleri bulalım
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Karakterleri tamsayılara ve tamsayıları karakterlere dönüştürmek için haritalama
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # string'i alıp tamsayı listesine çevirir
decode = lambda l: ''.join([itos[i] for i in l]) # tamsayı listesini alıp string'e çevirir

# Veriyi train ve validation setlerine ayıralım
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # verinin %90'ı train, %10'u validation
train_data = data[:n]
val_data = data[n:]

# Eğitim için veri bloğu (batch) oluşturma fonksiyonu
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """ Eğitim ve validasyon setleri için ortalama loss'u hesaplar. """
    out = {}
    model.eval() # Modeli değerlendirme moduna al
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Modeli tekrar eğitim moduna al
    return out

# --- Transformer'ın Yapı Taşları ---

class Head(nn.Module):
    """ Self-Attention'ın tek bir 'kafa'sı (Head) """
    def __init__(self, head_size):
        super().__init__()
        # Bir token'ın kendi kimliğini (Query) ve diğer token'ların içeriğini (Key) ve anlamını (Value) öğrenmesi için lineer katmanlar
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 'tril', gelecekteki token'ların iletişim kurmasını engelleyen bir maskedir. Decoder yapısının temelidir.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Query ve Key'lerin ne kadar uyumlu olduğunu hesapla (Attention Skorları)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Geleceğe bakmayı engelle
        wei = F.softmax(wei, dim=-1) # Olasılık dağılımına çevir
        wei = self.dropout(wei)
        
        # Skorlara göre Value'ları ağırlıklandır
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Paralel çalışan birden çok self-attention 'kafa'sı """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # Kafalardan gelen sonuçları birleştiren projeksiyon katmanı
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Tüm kafaların çıktısını birleştir (concatenate)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Basit bir lineer katman ve ardından ReLU aktivasyon fonksiyonu. Attention'dan gelen bilgiyi işler. """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projeksiyon katmanı
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Bir Transformer bloğu: Attention ve FeedForward katmanlarını bir araya getirir. """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # Multi-Head Attention katmanı
        self.ffwd = FeedForward(n_embd)                # Feed-Forward katmanı
        self.ln1 = nn.LayerNorm(n_embd)                # Layer Normalization 1
        self.ln2 = nn.LayerNorm(n_embd)                # Layer Normalization 2

    def forward(self, x):
        # Residual Connection (Artık Bağlantı): x + ... , modelin daha derin öğrenmesini sağlar.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- GPT Dil Modeli ---

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Her token için bir embedding vektörü oluşturur
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Her pozisyon için bir embedding vektörü oluşturur
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Transformer blokları
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Son LayerNorm katmanı
        self.lm_head = nn.Linear(n_embd, vocab_size) # Sonuçları kelime dağarcığı boyutuna haritalar

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx ve targets (B,T) boyutunda tamsayı tensorleridir
        tok_emb = self.token_embedding_table(idx) # (B, T, C) -> (Batch, Time, Channels)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # Token ve pozisyon bilgilerini birleştir
        x = self.blocks(x)    # Transformer bloklarından geçir
        x = self.ln_f(x)      # Son normalizasyon
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # Loss'u (kayıp/hata) hesapla
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx (B, T) boyutunda mevcut bağlamdaki token'ların dizisidir
        for _ in range(max_new_tokens):
            # Tahminler için context'i kırp (block_size'ı geçmemeli)
            idx_cond = idx[:, -block_size:]
            # Tahminleri al
            logits, loss = self(idx_cond)
            # Sadece son adıma odaklan
            logits = logits[:, -1, :] # (B, C) olur
            # Olasılıkları elde etmek için softmax uygula
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Olasılık dağılımından bir örnek seç
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Seçilen örneği diziye ekle
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx