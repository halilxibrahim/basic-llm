
import torch
from model import GPTLanguageModel, decode, device, encode

def generate_term_definition(model, start_term="", max_tokens=200):
    model.eval()
    
    if start_term:
        context = torch.tensor(encode(start_term), dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    generated = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    result = decode(generated)
    
    return result

def generate_aviation_dictionary(model, num_entries=5):
    print("🛩️  HAVAcılık VE SAVUNMA SANAYİ TERİMLERİ SÖZLÜĞÜ - AI GENERATED\n")
    print("=" * 60)
    
    aviation_starters = [
        "AIRCRAFT:",
        "MISSILE:",
        "RADAR:",
        "PILOT:",
        "ENGINE:",
        "FIGHTER:",
        "HELICOPTER:",
        "DRONE:",
        "NAVIGATION:",
        "SUPERSONIC:"
    ]
    
    for i in range(min(num_entries, len(aviation_starters))):
        print(f"\n📖 Terim {i+1}:")
        print("-" * 30)
        
        generated_text = generate_term_definition(model, aviation_starters[i], 150)
        
        if len(generated_text) > 300:
            generated_text = generated_text[:300] + "..."
        
        print(generated_text)
        print()

def interactive_dictionary_query(model):
    print("\n🔍 İNTERAKTİF SÖZLÜK SORGUSU")
    print("Çıkmak için 'exit' yazın\n")
    
    while True:
        user_input = input("Hangi terimle ilgili bilgi almak istiyorsunuz? ").strip()
        
        if user_input.lower() in ['exit', 'çıkış', 'quit']:
            print("Sözlük sorgusu sonlandırıldı.")
            break
        
        if user_input:
            print(f"\n📝 '{user_input}' ile ilgili AI tarafından üretilen içerik:")
            print("-" * 50)
            
            formatted_term = user_input.upper() + ":"
            result = generate_term_definition(model, formatted_term, 200)
            
            if len(result) > 400:
                result = result[:400] + "..."
            
            print(result)
            print()

if __name__ == "__main__":
    print("Model yükleniyor...")
    model = GPTLanguageModel()
    
    try:
        model.load_state_dict(torch.load('gpt_model.pth', map_location=device))
        model.to(device)
        print("✅ Model başarıyla yüklendi!\n")
        
        generate_aviation_dictionary(model, num_entries=5)
        
        interactive_dictionary_query(model)
        
    except FileNotFoundError:
        print("❌ Hata: 'gpt_model.pth' dosyası bulunamadı!")
        print("Önce 'python train.py' komutunu çalıştırarak modeli eğitin.")
    except Exception as e:
        print(f"❌ Hata: {str(e)}")