"""
Phase 6a: Generate Obfuscated Benign Dataset
Creates benign queries with realistic obfuscation patterns.
"""

import json
from pathlib import Path
from typing import List, Dict
import random

random.seed(42)


class ObfuscatedBenignGenerator:
    """Generates benign queries with various obfuscation patterns."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.phase6a_dir = self.script_dir.parent
        self.data_dir = self.phase6a_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Base benign queries
        self.base_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does deep learning work?",
            "What is natural language processing?",
            "Tell me about computer vision",
            "What are transformers in AI?",
            "Explain the attention mechanism",
            "What is reinforcement learning?",
            "How does backpropagation work?",
            "What are convolutional neural networks?",
            "Explain recurrent neural networks",
            "What is transfer learning?",
            "How does gradient descent work?",
            "What is overfitting?",
            "Explain regularization techniques",
            "What is cross-validation?",
            "How does clustering work?",
            "What is dimensionality reduction?",
            "Explain feature engineering",
            "What is data normalization?",
        ]
    
    def add_unicode_emoji(self, query: str) -> str:
        """Add Unicode characters and emoji."""
        emoji_map = {
            "machine learning": "machine 🤖 learning",
            "neural networks": "neural 🧠 networks",
            "deep learning": "deep 📊 learning",
            "data": "dâta",  # Accented
            "learning": "lëarning",  # Accented
            "algorithm": "àlgorithm",  # Accented
        }
        
        result = query
        for key, val in emoji_map.items():
            if key in result.lower():
                result = result.replace(key, val)
                break
        
        return result
    
    def add_homoglyphs(self, query: str) -> str:
        """Add homoglyph characters (Cyrillic/Greek lookalikes)."""
        homoglyph_map = {
            'a': 'а',  # Cyrillic a
            'e': 'е',  # Cyrillic e
            'o': 'о',  # Cyrillic o
            'p': 'р',  # Cyrillic p
            'c': 'с',  # Cyrillic s
            'x': 'х',  # Cyrillic h
        }
        
        result = list(query)
        # Replace 20% of letters with homoglyphs
        letter_indices = [i for i, c in enumerate(result) if c.isalpha() and c.lower() in homoglyph_map]
        for idx in random.sample(letter_indices, max(1, len(letter_indices) // 5)):
            result[idx] = homoglyph_map[result[idx].lower()]
        
        return ''.join(result)
    
    def add_zero_width_chars(self, query: str) -> str:
        """Add zero-width characters between words."""
        zwj = '\u200d'  # Zero-width joiner
        words = query.split()
        return zwj.join(words)
    
    def add_mixed_script(self, query: str) -> str:
        """Add legitimate non-Latin script."""
        additions = [
            f"{query} (в русском языке)",  # Russian
            f"{query} (在中文中)",  # Chinese
            f"{query} (باللغة العربية)",  # Arabic
            f"{query} (en français)",  # French
        ]
        return random.choice(additions)
    
    def add_formatting_obfuscation(self, query: str) -> str:
        """Add unusual formatting."""
        formatting_options = [
            query.replace(" ", "    "),  # Multiple spaces
            query.replace(" ", "\n"),  # Line breaks
            query.upper(),  # All caps
            ''.join(c.upper() if i % 2 else c.lower() for i, c in enumerate(query)),  # Mixed case
            query.replace(" ", "_"),  # Underscores
            query.replace(" ", "-"),  # Dashes
        ]
        return random.choice(formatting_options)
    
    def add_symbol_obfuscation(self, query: str) -> str:
        """Add unusual symbols and punctuation."""
        symbol_options = [
            query + "???",  # Repeated punctuation
            query.replace(" ", "..."),  # Ellipsis
            query.replace("?", "?!?!"),  # Repeated punctuation
            f">>> {query} <<<",  # Angle brackets
            f"*** {query} ***",  # Asterisks
        ]
        return random.choice(symbol_options)
    
    def generate_dataset(self) -> List[Dict]:
        """Generate obfuscated benign dataset."""
        dataset = []
        
        print("Generating obfuscated benign dataset...")
        
        # 1. Unicode/Emoji Obfuscation (30 samples)
        print("  Adding Unicode/emoji obfuscation (30 samples)...")
        for i in range(30):
            query = random.choice(self.base_queries)
            obfuscated = self.add_unicode_emoji(query)
            dataset.append({
                "id": len(dataset),
                "query": obfuscated,
                "is_benign": True,
                "obfuscation_type": "unicode_emoji",
                "original": query,
            })
        
        # 2. Homoglyph Obfuscation (30 samples)
        print("  Adding homoglyph obfuscation (30 samples)...")
        for i in range(30):
            query = random.choice(self.base_queries)
            obfuscated = self.add_homoglyphs(query)
            dataset.append({
                "id": len(dataset),
                "query": obfuscated,
                "is_benign": True,
                "obfuscation_type": "homoglyph",
                "original": query,
            })
        
        # 3. Zero-Width Character Obfuscation (30 samples)
        print("  Adding zero-width character obfuscation (30 samples)...")
        for i in range(30):
            query = random.choice(self.base_queries)
            obfuscated = self.add_zero_width_chars(query)
            dataset.append({
                "id": len(dataset),
                "query": obfuscated,
                "is_benign": True,
                "obfuscation_type": "zero_width",
                "original": query,
            })
        
        # 4. Mixed-Script Obfuscation (30 samples)
        print("  Adding mixed-script obfuscation (30 samples)...")
        for i in range(30):
            query = random.choice(self.base_queries)
            obfuscated = self.add_mixed_script(query)
            dataset.append({
                "id": len(dataset),
                "query": obfuscated,
                "is_benign": True,
                "obfuscation_type": "mixed_script",
                "original": query,
            })
        
        # 5. Formatting Obfuscation (30 samples)
        print("  Adding formatting obfuscation (30 samples)...")
        for i in range(30):
            query = random.choice(self.base_queries)
            obfuscated = self.add_formatting_obfuscation(query)
            dataset.append({
                "id": len(dataset),
                "query": obfuscated,
                "is_benign": True,
                "obfuscation_type": "formatting",
                "original": query,
            })
        
        # 6. Symbol/Punctuation Obfuscation (30 samples)
        print("  Adding symbol/punctuation obfuscation (30 samples)...")
        for i in range(30):
            query = random.choice(self.base_queries)
            obfuscated = self.add_symbol_obfuscation(query)
            dataset.append({
                "id": len(dataset),
                "query": obfuscated,
                "is_benign": True,
                "obfuscation_type": "symbol",
                "original": query,
            })
        
        # 7. Legitimate Non-Latin Queries (40 samples)
        print("  Adding legitimate non-Latin queries (40 samples)...")
        non_latin_queries = [
            "Что такое машинное обучение?",  # Russian
            "Как работают нейронные сети?",  # Russian
            "什么是深度学习?",  # Chinese
            "神经网络如何工作?",  # Chinese
            "ما هو التعلم الآلي?",  # Arabic
            "كيف تعمل الشبكات العصبية?",  # Arabic
            "Qu'est-ce que l'apprentissage automatique?",  # French
            "Comment fonctionnent les réseaux de neurones?",  # French
        ]
        
        for i in range(40):
            query = random.choice(non_latin_queries)
            dataset.append({
                "id": len(dataset),
                "query": query,
                "is_benign": True,
                "obfuscation_type": "non_latin",
                "original": query,
            })
        
        # 8. Clean Benign Queries (40 samples - baseline)
        print("  Adding clean benign queries (40 samples)...")
        for i in range(40):
            query = random.choice(self.base_queries)
            dataset.append({
                "id": len(dataset),
                "query": query,
                "is_benign": True,
                "obfuscation_type": "clean",
                "original": query,
            })
        
        print(f"✓ Generated {len(dataset)} obfuscated benign samples")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict]):
        """Save dataset to JSON."""
        output_file = self.data_dir / "obfuscated_benign_queries.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved dataset to {output_file}")
        
        # Print summary
        print(f"\nDataset Summary:")
        obfuscation_counts = {}
        for sample in dataset:
            obf_type = sample["obfuscation_type"]
            obfuscation_counts[obf_type] = obfuscation_counts.get(obf_type, 0) + 1
        
        for obf_type, count in sorted(obfuscation_counts.items()):
            print(f"  {obf_type}: {count} samples")
    
    def run(self):
        """Generate and save dataset."""
        print("\n" + "="*70)
        print("PHASE 6A: GENERATE OBFUSCATED BENIGN DATASET")
        print("="*70)
        
        dataset = self.generate_dataset()
        self.save_dataset(dataset)
        
        print("\n" + "="*70)
        print("✅ DATASET GENERATION COMPLETE")
        print("="*70)


def main():
    """Main entry point."""
    generator = ObfuscatedBenignGenerator()
    generator.run()


if __name__ == "__main__":
    main()
