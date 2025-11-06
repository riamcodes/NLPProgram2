"""
CS 5/7322 - Programming Homework 2: Extra Credit
Repeat Part 1 and Part 2 with SciBERT model
"""

from transformers import BertModel, BertTokenizer
from hw2 import (
    genBERTVector, cosine_similarity, 
    part1_analysis, part2_analysis
)
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os


def part1_analysis_with_prefix(model, tokenizer, file_prefix=''):
    """
    Wrapper for part1_analysis that adds file prefix to avoid overwriting original files.
    This is a modified version that saves files with a prefix.
    """
    # Import the word sentences from part1
    from hw2 import part1_analysis
    
    # Call original function but manually handle file naming
    # duplicate some logic to add the prefix
    import numpy as np
    from itertools import combinations
    
    # Get the word sentences (same as in hw2.py)
    word_sentences = {
        "park": [
            "The park next to my house offers a nice walk.",
            "The residents are trying to build a park for the kids to play.",
            "We had a picnic in the central park last weekend.",
            "The city council approved funding for a new park downtown.",
            "Children love playing on the swings at the park.",
            "The park has beautiful flowers and tall trees.",
            "Many people exercise in the park every morning.",
            "The park closes at sunset for safety reasons.",
            "There is a small pond in the middle of the park.",
            "The community organized a cleanup day at the park."
        ],
        "bank": [
            "I need to deposit money at the bank today.",
            "The bank approved my loan application yesterday.",
            "She works as a teller at the local bank.",
            "The bank charges a monthly fee for this account.",
            "I opened a savings account at the new bank.",
            "The bank is closed on weekends and holidays.",
            "He withdrew cash from the bank this morning.",
            "The bank offers competitive interest rates.",
            "I forgot my bank card at home today.",
            "The bank has excellent customer service."
        ],
        "bat": [
            "The bat flew out of the cave at dusk.",
            "A bat can see in the dark using echolocation.",
            "The bat hung upside down from the tree branch.",
            "Scientists study the bat to understand its behavior.",
            "The bat found insects to eat during the night.",
            "A small bat was sleeping in the attic.",
            "The bat uses sound waves to navigate.",
            "I saw a bat circling the streetlight.",
            "The bat colony lives in the old barn.",
            "A fruit bat feeds on nectar and fruits."
        ],
        "spring": [
            "Spring is my favorite season because of the flowers.",
            "The weather is perfect during spring.",
            "I love seeing the cherry blossoms in spring.",
            "Spring brings warmer temperatures and longer days.",
            "We plan our garden planting for early spring.",
            "Spring cleaning is a tradition in our household.",
            "The birds return from migration in spring.",
            "Spring marks the end of winter.",
            "I enjoy taking walks during spring mornings.",
            "Spring festivals celebrate the new growing season."
        ],
        "ring": [
            "She received a beautiful diamond ring for her birthday.",
            "The engagement ring has a stunning sapphire.",
            "He lost his class ring at the beach.",
            "The wedding ring symbolizes eternal commitment.",
            "She wears a gold ring on her left hand.",
            "The ring was too tight and needed resizing.",
            "I bought a silver ring from the jewelry store.",
            "The antique ring belonged to her grandmother.",
            "The ring sparkled in the sunlight.",
            "He gave her a promise ring before leaving."
        ],
        "train": [
            "The train arrived at the station on time.",
            "I take the train to work every morning.",
            "The train travels through mountains and valleys.",
            "Passengers boarded the train for the long journey.",
            "The train whistle could be heard from miles away.",
            "The bullet train reaches speeds of 200 miles per hour.",
            "We missed the last train and had to wait.",
            "The train tracks cross through the city center.",
            "The train conductor checked everyone's tickets.",
            "The train derailed near the rural station."
        ],
        "mouse": [
            "The mouse scurried across the kitchen floor.",
            "A tiny mouse built a nest in the garden.",
            "The cat chased the mouse around the house.",
            "The mouse found cheese in the trap.",
            "I saw a mouse running along the wall.",
            "The mouse is nocturnal and active at night.",
            "A field mouse lives in the grass.",
            "The mouse squeaked when it was frightened.",
            "The mouse has large ears and a long tail.",
            "Scientists use the mouse for laboratory experiments."
        ],
        "crane": [
            "The crane stood gracefully on one leg in the marsh.",
            "A beautiful crane flew overhead during migration.",
            "The crane is known for its elegant dancing courtship.",
            "We spotted a rare crane at the wildlife sanctuary.",
            "The crane has a long neck and pointed beak.",
            "The Japanese crane is a symbol of good luck.",
            "The crane waded through the shallow water.",
            "The crane pair nested near the lake together.",
            "The crane spread its wings to take flight.",
            "Birdwatchers travel far to see the crane."
        ]
    }
    
    print("\n" + "="*80)
    print("PART 1: Comparing BERT vectors for same word + same meaning")
    print("="*80)
    
    all_word_results = {}
    all_similarities = []
    
    for word, sentences in word_sentences.items():
        print(f"\n{'='*80}")
        print(f"Processing word: {word.upper()}")
        print(f"{'='*80}")
        
        vectors = genBERTVector(model, tokenizer, word, sentences)
        valid_vectors = [v for v in vectors if len(v) > 0]
        
        if len(valid_vectors) != 10:
            print(f"WARNING: Expected 10 vectors, got {len(valid_vectors)}")
        
        similarities = []
        for i, j in combinations(range(len(valid_vectors)), 2):
            sim = cosine_similarity(valid_vectors[i], valid_vectors[j])
            similarities.append(sim)
        
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        all_word_results[word] = {
            'sentences': sentences,
            'vectors': valid_vectors,
            'similarities': similarities,
            'mean': mean_sim,
            'std': std_sim
        }
        
        all_similarities.extend(similarities)
        
        print(f"\nSentences for '{word}':")
        for idx, sent in enumerate(sentences, 1):
            print(f"  {idx}. {sent}")
        
        print(f"\nStatistics for '{word}':")
        print(f"  Mean cosine similarity: {mean_sim:.4f}")
        print(f"  Standard deviation: {std_sim:.4f}")
        print(f"  Number of pairs: {len(similarities)}")
        
        # Create histogram with prefix
        plt.figure(figsize=(10, 6))
        bins = np.arange(0, 1.1, 0.1)
        counts, edges, patches = plt.hist(similarities, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Cosine Similarities for Word: "{word}" (SciBERT)')
        plt.xticks(bins)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{file_prefix}part1_histogram_{word}.png', dpi=150)
        print(f"\nSaved histogram: {file_prefix}part1_histogram_{word}.png")
        plt.close()
    
    # Combined analysis
    print(f"\n{'='*80}")
    print("COMBINED ANALYSIS (All 8 words, 360 pairs total)")
    print(f"{'='*80}")
    
    overall_mean = np.mean(all_similarities)
    overall_std = np.std(all_similarities)
    
    print(f"\nOverall Statistics:")
    print(f"  Total pairs: {len(all_similarities)}")
    print(f"  Overall mean cosine similarity: {overall_mean:.4f}")
    print(f"  Overall standard deviation: {overall_std:.4f}")
    
    # Create combined histogram with prefix
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.1, 0.1)
    counts, edges, patches = plt.hist(all_similarities, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Cosine Similarities (Combined - All 8 Words, 360 pairs) - SciBERT')
    plt.xticks(bins)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}part1_histogram_combined.png', dpi=150)
    print(f"\nSaved combined histogram: {file_prefix}part1_histogram_combined.png")
    plt.close()
    
    # Save results with prefix
    with open(f'{file_prefix}part1_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PART 1 RESULTS: Comparing BERT vectors for same word + same meaning (SciBERT)\n")
        f.write("="*80 + "\n\n")
        
        for word, results in all_word_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Word: {word.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Sentences for '{word}':\n")
            for idx, sent in enumerate(results['sentences'], 1):
                f.write(f"  {idx}. {sent}\n")
            
            f.write(f"\nStatistics for '{word}':\n")
            f.write(f"  Mean cosine similarity: {results['mean']:.4f}\n")
            f.write(f"  Standard deviation: {results['std']:.4f}\n")
            f.write(f"  Number of pairs: {len(results['similarities'])}\n")
            f.write("\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("COMBINED ANALYSIS (All 8 words, 360 pairs total)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Overall Statistics:\n")
        f.write(f"  Total pairs: {len(all_similarities)}\n")
        f.write(f"  Overall mean cosine similarity: {overall_mean:.4f}\n")
        f.write(f"  Overall standard deviation: {overall_std:.4f}\n")
    
    print(f"\nSaved detailed results to: {file_prefix}part1_results.txt")
    
    return all_word_results, overall_mean, overall_std


def part2_analysis_with_prefix(part1_results, file_prefix=''):
    """
    Wrapper for part2_analysis that adds file prefix.
    """
    import random
    
    print("\n" + "="*80)
    print("PART 2: Comparing BERT vectors for different words")
    print("="*80)
    
    all_labeled_vectors = []
    word_index_map = {}
    
    words_list = list(part1_results.keys())
    for word_idx, (word, results) in enumerate(part1_results.items()):
        word_index_map[word] = word_idx
        for sent_idx, vector in enumerate(results['vectors']):
            all_labeled_vectors.append((vector, word_idx, word, sent_idx))
    
    print(f"\nTotal vectors available: {len(all_labeled_vectors)}")
    print(f"Total words: {len(words_list)}")
    print(f"Expected pairs: 360")
    
    # Generate all possible pairs where vectors are from different words
    all_different_word_pairs = []
    
    for i in range(len(all_labeled_vectors)):
        vec1, word_idx1, word1, sent_idx1 = all_labeled_vectors[i]
        for j in range(i + 1, len(all_labeled_vectors)):
            vec2, word_idx2, word2, sent_idx2 = all_labeled_vectors[j]
            if word_idx1 != word_idx2:
                all_different_word_pairs.append((vec1, vec2, word1, word2))
    
    print(f"Total possible different-word pairs: {len(all_different_word_pairs)}")
    
    # Randomly sample 360 pairs
    random.seed(42)
    
    if len(all_different_word_pairs) >= 360:
        different_word_pairs = random.sample(all_different_word_pairs, 360)
    else:
        print(f"WARNING: Only {len(all_different_word_pairs)} pairs available, using all of them")
        different_word_pairs = all_different_word_pairs
    
    print(f"\nGenerated {len(different_word_pairs)} pairs of vectors from different words")
    
    # Calculate cosine similarities
    similarities = []
    for vec1, vec2, word1, word2 in different_word_pairs:
        sim = cosine_similarity(vec1, vec2)
        similarities.append(sim)
    
    # Calculate statistics
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    print(f"\nStatistics for different-word pairs:")
    print(f"  Total pairs: {len(similarities)}")
    print(f"  Mean cosine similarity: {mean_sim:.4f}")
    print(f"  Standard deviation: {std_sim:.4f}")
    
    # Create histogram with prefix
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.1, 0.1)
    counts, edges, patches = plt.hist(similarities, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cosine Similarities (Different Words, 360 pairs) - SciBERT')
    plt.xticks(bins)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}part2_histogram.png', dpi=150)
    print(f"\nSaved histogram: {file_prefix}part2_histogram.png")
    plt.close()
    
    # Save results with prefix
    with open(f'{file_prefix}part2_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PART 2 RESULTS: Comparing BERT vectors for different words (SciBERT)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total pairs analyzed: {len(similarities)}\n")
        f.write(f"Mean cosine similarity: {mean_sim:.4f}\n")
        f.write(f"Standard deviation: {std_sim:.4f}\n")
        f.write("\n")
        f.write("Note: Each pair consists of vectors from different words.\n")
    
    print(f"\nSaved detailed results to: {file_prefix}part2_results.txt")
    
    return similarities, mean_sim, std_sim


def extra_credit_scibert_analysis():
    """
    Extra Credit: Repeat Part 1 and Part 2 with SciBERT
    """
    print("\n" + "="*80)
    print("EXTRA CREDIT: SciBERT Analysis (Part 1 & 2)")
    print("="*80)
    
    # Load SciBERT model and tokenizer
    print("\nLoading SciBERT model and tokenizer...")
    print("This may take a few minutes on first run (downloading model)...")
    
    try:
        # SciBERT model: allenai/scibert_scivocab_uncased
        model_name = 'allenai/scibert_scivocab_uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        print("SciBERT model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading SciBERT: {e}")
        print("Please make sure you have internet connection and transformers library is up to date.")
        return None, None, None, None
    
    # Run Part 1 with SciBERT (with prefix to avoid overwriting original files)
    print("\n" + "="*80)
    print("Running Part 1 with SciBERT...")
    print("="*80)
    part1_results_scibert, part1_mean_scibert, part1_std_scibert = part1_analysis_with_prefix(
        model, tokenizer, file_prefix='scibert_'
    )
    
    # Run Part 2 with SciBERT (with prefix to avoid overwriting original files)
    print("\n" + "="*80)
    print("Running Part 2 with SciBERT...")
    print("="*80)
    part2_similarities_scibert, part2_mean_scibert, part2_std_scibert = part2_analysis_with_prefix(
        part1_results_scibert, file_prefix='scibert_'
    )
    
    # Save comparison results
    with open('scibert_comparison.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXTRA CREDIT: SciBERT vs BERT-base Comparison\n")
        f.write("="*80 + "\n\n")
        f.write("SciBERT Results:\n")
        f.write(f"  Part 1 (Same word, same meaning):\n")
        f.write(f"    Mean: {part1_mean_scibert:.4f}, Std: {part1_std_scibert:.4f}\n")
        f.write(f"  Part 2 (Different words, different meaning):\n")
        f.write(f"    Mean: {part2_mean_scibert:.4f}, Std: {part2_std_scibert:.4f}\n")
        f.write("\n")
        f.write("Note: Compare these results with BERT-base results from main analysis (hw2.py).\n")
        f.write("BERT-base results are typically:\n")
        f.write("  Part 1 Mean: ~0.72, Part 2 Mean: ~0.25\n")
    
    print("\n" + "="*80)
    print("EXTRA CREDIT COMPLETE")
    print("="*80)
    print(f"\nSciBERT Results:")
    print(f"  Part 1 (Same word, same meaning):")
    print(f"    Mean: {part1_mean_scibert:.4f}, Std: {part1_std_scibert:.4f}")
    print(f"  Part 2 (Different words, different meaning):")
    print(f"    Mean: {part2_mean_scibert:.4f}, Std: {part2_std_scibert:.4f}")
    print("\nComparison saved to: scibert_comparison.txt")
    
    return part1_mean_scibert, part1_std_scibert, part2_mean_scibert, part2_std_scibert


if __name__ == "__main__":
    extra_credit_scibert_analysis()


