# generate.py

import torch
from model import GPTLanguageModel, decode, device

# Modeli yükle
model = GPTLanguageModel()
model.load_state_dict(torch.load('gpt_model.pth'))
model.to(device)
model.eval() # Modeli değerlendirme moduna al

# Üretimi başlatmak için başlangıç token'ı (yeni satır karakteri \n)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Modeli kullanarak metin üret ve ekrana yazdır
print("Model tarafından üretilen metin:")
generated_chars = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_chars))