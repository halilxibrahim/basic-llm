
import torch
from model import GPTLanguageModel, decode, device

model = GPTLanguageModel()
model.load_state_dict(torch.load('gpt_model.pth'))
model.to(device)
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("Model tarafından üretilen metin:")
generated_chars = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_chars))