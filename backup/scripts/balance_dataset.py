"""
Balance the entire dataset (train, validation, test) by generating more AI samples
"""
import os
from pathlib import Path
from gtts import gTTS
import random
from tqdm import tqdm

# Extended texts for generation
texts_by_language = {
    'tamil': [
        'வணக்கம், இன்று உங்களுக்கு எப்படி உள்ளது?',
        'நான் உங்களுக்கு உதவ இங்கே இருக்கிறேன்',
        'தமிழ் மொழி மிகவும் அழகான மொழி',
        'தொழில்நுட்பம் நமது எதிர்காலம்',
        'கல்வி மிக முக்கியமானது',
        'இசை என்பது உலகளாவிய மொழி',
        'விஞ்ஞானம் முன்னேற்றத்திற்கு வழிவகுக்கிறது',
        'நட்பு என்பது விலைமதிப்பற்றது',
        'உடல்நலம் செல்வம் ஆகும்',
        'நேரம் என்பது பணம்',
        'உங்கள் கனவுகளைப் பின்தொடருங்கள்',
        'வெற்றிக்கு கடின உழைப்பு தேவை',
        'மகிழ்ச்சி உள்ளிருந்து வருகிறது',
        'அறிவு சக்தி',
        'நாளை ஒரு புதிய நாள்',
        'படிப்பு மிகவும் முக்கியம்',
        'உழைப்பே உயர்வுக்கு வழி',
        'நம்பிக்கை வெற்றிக்கு அடிப்படை',
        'பொறுமை கசப்பானது ஆனால் பலன் இனிப்பானது',
        'நல்லவர்களோடு சேர்ந்து இருங்கள்',
    ],
    'english': [
        'Hello, how are you doing today?',
        'I am here to assist you with anything you need',
        'English is a global language spoken worldwide',
        'Technology is shaping our future in many ways',
        'Education is the key to success and growth',
        'Music is a universal language that connects us',
        'Science leads to innovation and progress',
        'Friendship is one of the most valuable things',
        'Health is wealth and should be prioritized',
        'Time is money, use it wisely',
        'Follow your dreams and never give up',
        'Hard work is essential for success',
        'Happiness comes from within yourself',
        'Knowledge is power and opens doors',
        'Tomorrow is a brand new day',
        'Learning is a lifelong journey',
        'Effort leads to achievement',
        'Confidence is the foundation of success',
        'Patience is bitter but its fruit is sweet',
        'Surround yourself with good people',
    ],
    'hindi': [
        'नमस्ते, आज आप कैसे हैं?',
        'मैं आपकी किसी भी तरह से मदद करने के लिए यहाँ हूँ',
        'हिंदी भारत की राष्ट्रीय भाषा है',
        'तकनीकी हमारे भविष्य को आकार दे रही है',
        'शिक्षा सफलता की कुंजी है',
        'संगीत एक सार्वभौमिक भाषा है',
        'विज्ञान नवाचार की ओर ले जाता है',
        'दोस्ती अनमोल है',
        'स्वास्थ्य ही धन है',
        'समय धन है',
        'अपने सपनों का पीछा करें',
        'सफलता के लिए कड़ी मेहनत जरूरी है',
        'खुशी अंदर से आती है',
        'ज्ञान शक्ति है',
        'कल एक नया दिन है',
        'सीखना जीवन भर की यात्रा है',
        'प्रयास उपलब्धि की ओर ले जाता है',
        'आत्मविश्वास सफलता की नींव है',
        'धैर्य कड़वा होता है लेकिन इसका फल मीठा होता है',
        'अच्छे लोगों के साथ रहें',
    ],
    'malayalam': [
        'നമസ്കാരം, ഇന്ന് നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്?',
        'നിങ്ങളെ സഹായിക്കാൻ ഞാൻ ഇവിടെയുണ്ട്',
        'മലയാളം കേരളത്തിന്റെ ഭാഷയാണ്',
        'സാങ്കേതികവിദ്യ നമ്മുടെ ഭാവി രൂപപ്പെടുത്തുന്നു',
        'വിദ്യാഭ്യാസം വിജയത്തിന്റെ താക്കോലാണ്',
        'സംഗീതം ഒരു സാർവത്രിക ഭാഷയാണ്',
        'ശാസ്ത്രം നവീകരണത്തിലേക്ക് നയിക്കുന്നു',
        'സൗഹൃദം വിലമതിക്കാനാവാത്തതാണ്',
        'ആരോഗ്യമാണ് സമ്പത്ത്',
        'സമയം പണമാണ്',
        'നിങ്ങളുടെ സ്വപ്നങ്ങളെ പിന്തുടരുക',
        'വിജയത്തിന് കഠിനാധ്വാനം ആവശ്യമാണ്',
        'സന്തോഷം ഉള്ളിൽ നിന്ന് വരുന്നു',
        'അറിവ് ശക്തിയാണ്',
        'നാളെ ഒരു പുതിയ ദിവസമാണ്',
        'പഠനം ജീവിതകാലം മുഴുവൻ യാത്രയാണ്',
        'പരിശ്രമം നേട്ടത്തിലേക്ക് നയിക്കുന്നു',
        'ആത്മവിശ്വാസം വിജയത്തിന്റെ അടിത്തറയാണ്',
        'ക്ഷമ കയ്പുള്ളതാണ് എന്നാൽ അതിന്റെ ഫലം മധുരമാണ്',
        'നല്ല ആളുകളുമായി ചുറ്റിക്കൊള്ളുക',
    ],
    'telugu': [
        'నమస్కారం, ఈరోజు మీరు ఎలా ఉన్నారు?',
        'మీకు సహాయం చేయడానికి నేను ఇక్కడ ఉన్నాను',
        'తెలుగు తెలుగు రాష్ట్రాల భాష',
        'సాంకేతికత మన భవిష్యత్తును రూపొందిస్తోంది',
        'విద్య విజయానికి కీలకం',
        'సంగీతం సార్వత్రిక భాష',
        'శాస్త్రం ఆవిష్కరణకు దారితీస్తుంది',
        'స్నేహం అమూల్యమైనది',
        'ఆరోగ్యమే సంపద',
        'సమయం డబ్బు',
        'మీ కలలను అనుసరించండి',
        'విజయానికి కష్టపడి పని చేయడం అవసరం',
        'ఆనందం లోపలి నుండి వస్తుంది',
        'జ్ఞానం శక్తి',
        'రేపు కొత్త రోజు',
        'నేర్చుకోవడం జీవితకాల ప్రయాణం',
        'ప్రయత్నం సాధనకు దారి తీస్తుంది',
        'విశ్వాసం విజయానికి పునాది',
        'సహనం చేదుగా ఉంటుంది కానీ దాని ఫలితం తీయగా ఉంటుంది',
        'మంచి వ్యక్తులతో ఉండండి',
    ]
}

def generate_balanced_dataset():
    """Balance all splits: train, validation, test"""
    
    language_codes = {
        'tamil': 'ta',
        'english': 'en',
        'hindi': 'hi',
        'malayalam': 'ml',
        'telugu': 'te'
    }
    
    # Target counts for each split
    targets = {
        'train': 800,      # 800 per language = 4000 total (matches human)
        'validation': 100, # 100 per language = 500 total (matches human)
        'test': 100        # 100 per language = 500 total (matches human)
    }
    
    print("="*60)
    print("🎯 BALANCING DATASET - ALL SPLITS")
    print("="*60)
    print("\nTargets:")
    print(f"  Train: {targets['train']} per language ({targets['train']*5} total)")
    print(f"  Validation: {targets['validation']} per language ({targets['validation']*5} total)")
    print(f"  Test: {targets['test']} per language ({targets['test']*5} total)")
    print("="*60 + "\n")
    
    total_generated = 0
    
    for split, target_per_language in targets.items():
        print(f"\n{'='*60}")
        print(f"📂 Processing {split.upper()} split")
        print(f"{'='*60}")
        
        for language, texts in texts_by_language.items():
            output_dir = Path(f'dataset/{split}/ai_generated/{language}')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Count existing files
            existing_files = list(output_dir.glob('*.mp3')) + list(output_dir.glob('*.wav'))
            existing = len(existing_files)
            needed = target_per_language - existing
            
            if needed <= 0:
                print(f"✅ {language:10} | Has {existing:4} samples (sufficient)")
                continue
            
            print(f"🔄 {language:10} | Has {existing:4}, generating {needed:4} more...")
            
            # Generate samples with progress bar
            success_count = 0
            pbar = tqdm(range(needed), desc=f"   Generating {language}", ncols=80)
            
            for i in pbar:
                try:
                    # Randomly select text
                    text = random.choice(texts)
                    
                    # Add variation with speed
                    speed = random.choice([True, False])
                    
                    # Generate TTS
                    tts = gTTS(text=text, lang=language_codes[language], slow=speed)
                    
                    # Save
                    filename = f'gen_{split}_{existing + success_count:04d}.mp3'
                    filepath = output_dir / filename
                    tts.save(str(filepath))
                    
                    success_count += 1
                    total_generated += 1
                    
                except Exception as e:
                    pbar.write(f"   ⚠️  Error: {e}")
                    continue
            
            final_count = len(list(output_dir.glob('*.mp3')) + list(output_dir.glob('*.wav')))
            print(f"✅ {language:10} | Complete! Total: {final_count:4} samples\n")
    
    print("\n" + "="*60)
    print(f"✅ GENERATION COMPLETE!")
    print(f"   Total new samples generated: {total_generated}")
    print("="*60 + "\n")
    
    # Show final comprehensive stats
    print("="*60)
    print("📊 FINAL DATASET STATISTICS")
    print("="*60)
    
    grand_total_human = 0
    grand_total_ai = 0
    
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()}:")
        print("-" * 60)
        
        split_human = 0
        split_ai = 0
        
        # Count by language
        for language in ['tamil', 'english', 'hindi', 'malayalam', 'telugu']:
            human_path = Path(f'dataset/{split}/human/{language}')
            ai_path = Path(f'dataset/{split}/ai_generated/{language}')
            
            human_count = 0
            ai_count = 0
            
            if human_path.exists():
                human_count = len(list(human_path.glob('*.mp3')) + list(human_path.glob('*.wav')))
                split_human += human_count
            
            if ai_path.exists():
                ai_count = len(list(ai_path.glob('*.mp3')) + list(ai_path.glob('*.wav')))
                split_ai += ai_count
            
            print(f"  {language:10} | Human: {human_count:4} | AI: {ai_count:4}")
        
        print(f"  {'-'*58}")
        print(f"  {'TOTAL':10} | Human: {split_human:4} | AI: {split_ai:4}")
        
        grand_total_human += split_human
        grand_total_ai += split_ai
        
        # Show balance
        if split_human > 0 and split_ai > 0:
            balance = (split_ai / split_human) * 100
            print(f"  Balance: {balance:.1f}% {'✅ Good!' if 80 <= balance <= 120 else '⚠️  Imbalanced'}")
    
    print("\n" + "="*60)
    print("GRAND TOTAL:")
    print(f"  Human: {grand_total_human}")
    print(f"  AI: {grand_total_ai}")
    print(f"  Total: {grand_total_human + grand_total_ai}")
    
    if grand_total_human > 0 and grand_total_ai > 0:
        overall_balance = (grand_total_ai / grand_total_human) * 100
        print(f"  Overall Balance: {overall_balance:.1f}%")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    generate_balanced_dataset()