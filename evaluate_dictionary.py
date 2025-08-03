import torch
import re
from model import GPTLanguageModel, decode, device, encode
from collections import Counter

def calculate_vocabulary_coverage(generated_text, reference_terms):
    generated_upper = generated_text.upper()
    found_terms = []
    
    for term in reference_terms:
        if term.upper() in generated_upper:
            found_terms.append(term)
    
    coverage = len(found_terms) / len(reference_terms) if reference_terms else 0
    return coverage, found_terms

def check_term_definition_quality(generated_text):
    lines = generated_text.split('\n')
    quality_metrics = {
        'has_term_format': False,
        'has_definition': False,
        'proper_structure': False,
        'term_count': 0
    }
    
    term_pattern = r'^[A-ZÇĞİÖŞÜ\s]+:\s+[A-ZÇĞİÖŞÜa-zçğıöşü]'
    
    for line in lines:
        line = line.strip()
        if re.match(term_pattern, line):
            quality_metrics['has_term_format'] = True
            quality_metrics['term_count'] += 1
            
            definition_part = line.split(':', 1)[1].strip()
            if len(definition_part) > 20:
                quality_metrics['has_definition'] = True
    
    if quality_metrics['term_count'] > 0 and quality_metrics['has_definition']:
        quality_metrics['proper_structure'] = True
    
    return quality_metrics

def evaluate_technical_terminology_usage(generated_text):
    aviation_keywords = [
        'uçak', 'havacılık', 'motor', 'kanat', 'pilot', 'radar', 'füze',
        'helikopter', 'turbine', 'navigasyon', 'avionik', 'drone',
        'aircraft', 'aviation', 'missile', 'supersonic', 'jet'
    ]
    
    military_keywords = [
        'savunma', 'savaş', 'askeri', 'güvenlik', 'strateji', 'taktik',
        'silah', 'defense', 'military', 'weapon', 'combat'
    ]
    
    text_lower = generated_text.lower()
    
    aviation_count = sum(text_lower.count(keyword) for keyword in aviation_keywords)
    military_count = sum(text_lower.count(keyword) for keyword in military_keywords)
    
    total_words = len(generated_text.split())
    
    return {
        'aviation_density': aviation_count / total_words if total_words > 0 else 0,
        'military_density': military_count / total_words if total_words > 0 else 0,
        'technical_score': (aviation_count + military_count) / total_words if total_words > 0 else 0
    }

def comprehensive_model_evaluation(model, num_samples=3):
    print("🔍 HAVAcılık SÖZLÜĞÜ MODELİ DEĞERLENDİRMESİ")
    print("=" * 60)
    
    reference_terms = [
        'AIRCRAFT', 'MISSILE', 'RADAR', 'PILOT', 'ENGINE', 'FIGHTER',
        'HELICOPTER', 'DRONE', 'NAVIGATION', 'SUPERSONIC', 'AVIONICS',
        'BOMBER', 'INTERCEPTOR', 'TURBINE', 'LANDING GEAR'
    ]
    
    total_coverage = 0
    total_quality_score = 0
    total_technical_score = 0
    
    print(f"\n📊 {num_samples} örnek üzerinde değerlendirme yapılıyor...\n")
    
    for i in range(num_samples):
        print(f"🔬 Örnek {i+1}:")
        print("-" * 30)
        
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=300)[0].tolist()
        generated_text = decode(generated)
        
        coverage, found_terms = calculate_vocabulary_coverage(generated_text, reference_terms)
        total_coverage += coverage
        
        quality_metrics = check_term_definition_quality(generated_text)
        quality_score = (
            quality_metrics['has_term_format'] * 0.3 +
            quality_metrics['has_definition'] * 0.4 +
            quality_metrics['proper_structure'] * 0.3
        )
        total_quality_score += quality_score
        
        tech_metrics = evaluate_technical_terminology_usage(generated_text)
        total_technical_score += tech_metrics['technical_score']
        
        print(f"📈 Kapsama Oranı: {coverage:.2%}")
        print(f"🎯 Kalite Skoru: {quality_score:.2f}/1.0")
        print(f"🛠️  Teknik Terim Yoğunluğu: {tech_metrics['technical_score']:.3f}")
        print(f"✈️  Bulunan Terimler: {', '.join(found_terms[:5])}")
        if len(found_terms) > 5:
            print(f"    ... ve {len(found_terms)-5} diğer terim")
        print()
    
    print("📋 GENEL DEĞERLENDİRME SONUÇLARI")
    print("=" * 40)
    print(f"🎯 Ortalama Kapsama Oranı: {total_coverage/num_samples:.2%}")
    print(f"⭐ Ortalama Kalite Skoru: {total_quality_score/num_samples:.2f}/1.0")
    print(f"🔧 Ortalama Teknik Terim Yoğunluğu: {total_technical_score/num_samples:.3f}")
    
    overall_score = (total_coverage + total_quality_score + total_technical_score) / (3 * num_samples)
    print(f"\n🏆 GENEL PERFORMANS SKORU: {overall_score:.2f}/1.0")
    
    if overall_score > 0.7:
        print("✅ Mükemmel! Model havacılık terminolojisini çok iyi öğrenmiş.")
    elif overall_score > 0.5:
        print("✔️  İyi! Model temel terimleri öğrenmiş, gelişim devam ediyor.")
    elif overall_score > 0.3:
        print("⚠️  Orta! Daha fazla eğitim gerekebilir.")
    else:
        print("❌ Zayıf! Model daha fazla eğitim ve veri gerekiyor.")

if __name__ == "__main__":
    print("Model yükleniyor...")
    model = GPTLanguageModel()
    
    try:
        model.load_state_dict(torch.load('gpt_model.pth', map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model başarıyla yüklendi!\n")
        
        comprehensive_model_evaluation(model, num_samples=5)
        
    except FileNotFoundError:
        print("❌ Hata: 'gpt_model.pth' dosyası bulunamadı!")
        print("Önce 'python train.py' komutunu çalıştırarak modeli eğitin.")
    except Exception as e:
        print(f"❌ Hata: {str(e)}")