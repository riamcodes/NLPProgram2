"""
CS 5/7322 - Programming Homework 2: Examining contextualized Word Embeddings for BERT
"""

from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
import random
try:
    from nltk.corpus import wordnet as wn
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("WARNING: NLTK not available. WordNet functionality will be limited.")


def genBERTVector(model, tokenizer, word, sentence_list):
    """
    Generate BERT vectors for a specific word across multiple sentences.
    
    Args:
        model: Pre-trained BERT model
        tokenizer: BERT tokenizer
        word: The word to extract vectors for
        sentence_list: List of sentences (strings) that should contain the word
        
    Returns:
        List of vectors (numpy arrays), where each vector corresponds to the word
        in each sentence. Returns empty list for sentences where word doesn't appear.
        If word appears multiple times, takes the vector for the first appearance.
    """
    vectors = []
    
    # Tokenize word to find its subword tokens
    word_tokens = tokenizer.tokenize(word)
    if not word_tokens:
        # Word not in vocabulary - return empty lists for all sentences
        return [[] for _ in sentence_list]
    
    # Process each sentence individually to accurately find word positions
    for sentence in sentence_list:
        # Tokenize the sentence
        tokens = tokenizer.tokenize(sentence)
        
        # Find the word in the tokenized sentence
        word_found = False
        word_position = None
        
        # Check if the word appears as subwords in the tokenized sentence
        for j in range(len(tokens) - len(word_tokens) + 1):
            if tokens[j:j+len(word_tokens)] == word_tokens:
                # Found the word, position is j+1 (+1 for [CLS] token that will be added)
                word_position = j + 1
                word_found = True
                break
        
        if not word_found:
            vectors.append([])
            continue
        
        # Now tokenize the sentence properly (with [CLS] and [SEP])
        inputs = tokenizer(sentence, return_tensors="pt")
        
        # Forward pass through BERT
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the hidden states (embeddings) of the last layer
        last_hidden_states = outputs.last_hidden_state.numpy()
        
        # Extract the vector for the word (first 50 dimensions as specified)
        if word_position < last_hidden_states.shape[1]:
            word_vector = last_hidden_states[0, word_position, :50]
            vectors.append(word_vector)
        else:
            vectors.append([])
    
    return vectors


def cosine_similarity(v1, v2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1, v2: numpy arrays (vectors)
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def part1_analysis(model, tokenizer):
    """
    Part 1: Comparing BERT vectors for same word + same meaning across sentences
    """
    print("\n" + "="*80)
    print("PART 1: Comparing BERT vectors for same word + same meaning")
    print("="*80)
    
    # Define 8 words with their 10 sentences each
    # Each word uses the SAME meaning across all 10 sentences
    word_sentences = {
        # park: outdoor recreational area (NOT parking a car)
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
        
        # bank: financial institution (NOT river bank)
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
        
        # bat: flying mammal animal (NOT sports equipment)
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
        
        # spring: season of the year (NOT coil or water source)
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
        
        # ring: jewelry worn on finger (NOT sound or circular object)
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
        
        # train: locomotive vehicle (NOT to teach/educate)
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
        
        # mouse: small rodent animal (NOT computer device)
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
        
        # crane: large bird (NOT construction equipment)
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
    
    # Extract vectors for each word
    all_word_results = {}
    all_similarities = []  # For combined histogram (360 pairs total)
    
    for word, sentences in word_sentences.items():
        print(f"\n{'='*80}")
        print(f"Processing word: {word.upper()}")
        print(f"{'='*80}")
        
        # Get vectors for this word
        vectors = genBERTVector(model, tokenizer, word, sentences)
        
        # Check that we got valid vectors
        valid_vectors = [v for v in vectors if len(v) > 0]
        if len(valid_vectors) != 10:
            print(f"WARNING: Expected 10 vectors, got {len(valid_vectors)}")
            print(f"Invalid vectors at indices: {[i for i, v in enumerate(vectors) if len(v) == 0]}")
        
        # Calculate cosine similarities between all pairs
        similarities = []
        for i, j in combinations(range(len(valid_vectors)), 2):
            sim = cosine_similarity(valid_vectors[i], valid_vectors[j])
            similarities.append(sim)
        
        # Store results
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        all_word_results[word] = {
            'sentences': sentences,
            'vectors': valid_vectors,
            'similarities': similarities,
            'mean': mean_sim,
            'std': std_sim
        }
        
        # Add to combined list
        all_similarities.extend(similarities)
        
        # Print results for this word
        print(f"\nSentences for '{word}':")
        for idx, sent in enumerate(sentences, 1):
            print(f"  {idx}. {sent}")
        
        print(f"\nStatistics for '{word}':")
        print(f"  Mean cosine similarity: {mean_sim:.4f}")
        print(f"  Standard deviation: {std_sim:.4f}")
        print(f"  Number of pairs: {len(similarities)}")
        
        # Create histogram for this word
        plt.figure(figsize=(10, 6))
        bins = np.arange(0, 1.1, 0.1)  # [0, 0.1, 0.2, ..., 1.0]
        counts, edges, patches = plt.hist(similarities, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Cosine Similarities for Word: "{word}"')
        plt.xticks(bins)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'part1_histogram_{word}.png', dpi=150)
        print(f"\nSaved histogram: part1_histogram_{word}.png")
        plt.close()
    
    # Combined analysis (all 360 pairs)
    print(f"\n{'='*80}")
    print("COMBINED ANALYSIS (All 8 words, 360 pairs total)")
    print(f"{'='*80}")
    
    overall_mean = np.mean(all_similarities)
    overall_std = np.std(all_similarities)
    
    print(f"\nOverall Statistics:")
    print(f"  Total pairs: {len(all_similarities)}")
    print(f"  Overall mean cosine similarity: {overall_mean:.4f}")
    print(f"  Overall standard deviation: {overall_std:.4f}")
    
    # Create combined histogram
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.1, 0.1)
    counts, edges, patches = plt.hist(all_similarities, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cosine Similarities (Combined - All 8 Words, 360 pairs)')
    plt.xticks(bins)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('part1_histogram_combined.png', dpi=150)
    print(f"\nSaved combined histogram: part1_histogram_combined.png")
    plt.close()
    
    # Save results to a text file for report
    with open('part1_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PART 1 RESULTS: Comparing BERT vectors for same word + same meaning\n")
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
    
    print("\nSaved detailed results to: part1_results.txt")
    
    return all_word_results, overall_mean, overall_std


def part2_analysis(part1_results):
    """
    Part 2: Comparing BERT vectors for different words and different meaning across sentences
    """
    print("\n" + "="*80)
    print("PART 2: Comparing BERT vectors for different words")
    print("="*80)
    
    # Collect all vectors with their word labels
    # Structure: (vector, word_label, sentence_index)
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
    # Then randomly sample 360 from them
    all_different_word_pairs = []
    
    for i in range(len(all_labeled_vectors)):
        vec1, word_idx1, word1, sent_idx1 = all_labeled_vectors[i]
        for j in range(i + 1, len(all_labeled_vectors)):
            vec2, word_idx2, word2, sent_idx2 = all_labeled_vectors[j]
            # Only add if they're from different words
            if word_idx1 != word_idx2:
                all_different_word_pairs.append((vec1, vec2, word1, word2))
    
    print(f"Total possible different-word pairs: {len(all_different_word_pairs)}")
    
    # Randomly sample 360 pairs
    random.seed(42)  # For reproducibility
    
    if len(all_different_word_pairs) >= 360:
        different_word_pairs = random.sample(all_different_word_pairs, 360)
    else:
        print(f"WARNING: Only {len(all_different_word_pairs)} pairs available, using all of them")
        different_word_pairs = all_different_word_pairs
    
    print(f"\nGenerated {len(different_word_pairs)} pairs of vectors from different words")
    
    # Verification: Check a few sample pairs to ensure they're from different words
    print("\nVerification - Sample pairs (first 5):")
    for i, (vec1, vec2, word1, word2) in enumerate(different_word_pairs[:5]):
        print(f"  Pair {i+1}: {word1} vs {word2} (different: {word1 != word2})")
    
    # Calculate cosine similarities for all pairs
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
    
    # Create histogram (same format as Part 1)
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.1, 0.1)  # [0, 0.1, 0.2, ..., 1.0]
    counts, edges, patches = plt.hist(similarities, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cosine Similarities (Different Words, 360 pairs)')
    plt.xticks(bins)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('part2_histogram.png', dpi=150)
    print(f"\nSaved histogram: part2_histogram.png")
    plt.close()
    
    # Save results to a text file for report
    with open('part2_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PART 2 RESULTS: Comparing BERT vectors for different words\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total pairs analyzed: {len(similarities)}\n")
        f.write(f"Mean cosine similarity: {mean_sim:.4f}\n")
        f.write(f"Standard deviation: {std_sim:.4f}\n")
        f.write("\n")
        f.write("Note: Each pair consists of vectors from different words.\n")
    
    print("\nSaved detailed results to: part2_results.txt")
    
    return similarities, mean_sim, std_sim


def find_synset_word_pairs():
    """
    Find 8 synsets from WordNet and extract 2 words from each.
    Returns list of tuples: [(word1, word2, synset_name), ...]
    """
    if not NLTK_AVAILABLE:
        print("ERROR: NLTK not available. Please install nltk and download wordnet data.")
        print("Run: pip install nltk && python -c 'import nltk; nltk.download(\"wordnet\")'")
        return []
    
    word_pairs = []
    
    # Manually selected synsets with good word pairs
    # Each tuple: (synset_name, word1, word2)
    synset_pairs = [
        # car, auto, automobile, machine, motorcar
        ("car.n.01", "car", "automobile"),
        
        # house, home, abode, domicile, dwelling
        ("house.n.01", "house", "home"),
        
        # big, large, great
        ("big.a.01", "big", "large"),
        
        # happy, glad, joyful
        ("happy.a.01", "happy", "glad"),
        
        # fast, quick, rapid
        ("fast.a.01", "fast", "quick"),
        
        # beautiful, pretty, lovely
        ("beautiful.a.01", "beautiful", "pretty"),
        
        # small, little, tiny
        ("small.a.01", "small", "little"),
        
        # smart, intelligent, clever
        ("smart.a.01", "smart", "intelligent")
    ]
    
    # Verify synsets exist and words are in them
    verified_pairs = []
    for synset_name, word1, word2 in synset_pairs:
        try:
            # Try to find the synset
            synsets = wn.synsets(word1)
            found = False
            for syn in synsets:
                if word2 in [lemma.name().replace('_', ' ') for lemma in syn.lemmas()]:
                    verified_pairs.append((word1, word2, syn.name()))
                    found = True
                    break
            if not found:
                # Use the pair anyway if both are valid words
                print(f"Note: '{word1}' and '{word2}' may not be in same synset, using pair anyway")
                verified_pairs.append((word1, word2, "custom"))
        except Exception as e:
            print(f"Warning: Could not verify pair ({word1}, {word2}): {e}")
            verified_pairs.append((word1, word2, "custom"))
    
    return verified_pairs[:8]  # Return exactly 8 pairs


def part3_analysis(model, tokenizer):
    """
    Part 3: Compare BERT vectors for different words and same meaning across sentences
    """
    print("\n" + "="*80)
    print("PART 3: Comparing BERT vectors for different words with same meaning")
    print("="*80)
    
    # Find 8 word pairs from WordNet synsets
    print("\nFinding word pairs from WordNet synsets...")
    word_pairs = find_synset_word_pairs()
    
    if not word_pairs:
        print("ERROR: Could not find word pairs. Using predefined pairs.")
        # Fallback: use predefined pairs
        word_pairs = [
            ("car", "automobile", "custom"),
            ("house", "home", "custom"),
            ("big", "large", "custom"),
            ("happy", "glad", "custom"),
            ("fast", "quick", "custom"),
            ("beautiful", "pretty", "custom"),
            ("small", "little", "custom"),
            ("smart", "intelligent", "custom")
        ]
    
    print(f"Found {len(word_pairs)} word pairs")
    
    # Define 5 sentences for each word in each pair
    # Sentences should be different and not just substitute one word for the other
    pair_sentences = {
        ("car", "automobile"): {
            "car": [
                "The red car is very fast.",
                "I need to buy a new car soon.",
                "The car broke down on the highway.",
                "She drives a sports car to work.",
                "The car's engine makes a loud noise."
            ],
            "automobile": [
                "This is a turbocharged automobile.",
                "The automobile industry has grown rapidly.",
                "He collects vintage automobile models.",
                "The automobile manufacturer recalled many vehicles.",
                "Electric automobile sales are increasing."
            ]
        },
        ("house", "home"): {
            "house": [
                "The house has three bedrooms and two bathrooms.",
                "We painted the house bright yellow last summer.",
                "The house was built in 1950.",
                "She owns a beautiful house by the lake.",
                "The house needs major repairs."
            ],
            "home": [
                "There is no place like a home.",
                "The home was decorated with modern furniture.",
                "She returned to her home after a long journey.",
                "The new home has a spacious backyard.",
                "They made their home in the countryside."
            ]
        },
        ("big", "large"): {
            "big": [
                "That is a very big tree in the park.",
                "The big dog ran across the field.",
                "She has a big collection of books.",
                "The big meeting is scheduled for tomorrow.",
                "A big storm is approaching the city."
            ],
            "large": [
                "The large building dominates the skyline.",
                "He inherited a large fortune from his family.",
                "The large crowd gathered at the stadium.",
                "She wore a large hat to the party.",
                "The large pizza could feed ten people."
            ]
        },
        ("happy", "glad"): {
            "happy": [
                "I am very happy with the results.",
                "The happy child played in the garden.",
                "She felt happy after receiving good news.",
                "They celebrated with happy smiles.",
                "The happy couple got married last week."
            ],
            "glad": [
                "I am glad you could make it to the party.",
                "The glad news spread quickly through the town.",
                "She was glad to see her old friend again.",
                "He seemed glad about the opportunity.",
                "We are glad the weather improved."
            ]
        },
        ("fast", "quick"): {
            "fast": [
                "The fast runner won the race easily.",
                "Time passes very fast when you are having fun.",
                "The fast food restaurant was crowded.",
                "She speaks too fast for me to understand.",
                "The fast train arrived on schedule."
            ],
            "quick": [
                "He gave a quick answer to the question.",
                "The quick decision saved us time.",
                "She has a quick mind for mathematics.",
                "We need a quick solution to this problem.",
                "The quick glance revealed the truth."
            ]
        },
        ("beautiful", "pretty"): {
            "beautiful": [
                "The beautiful sunset painted the sky orange.",
                "She wore a beautiful dress to the ball.",
                "The beautiful garden was full of flowers.",
                "The beautiful music filled the concert hall.",
                "The beautiful painting hung on the wall."
            ],
            "pretty": [
                "The pretty flower bloomed in spring.",
                "She looked pretty in her new outfit.",
                "The pretty little girl played with dolls.",
                "The pretty garden attracted many visitors.",
                "He gave her a pretty necklace."
            ]
        },
        ("small", "little"): {
            "small": [
                "The small bird flew into the nest.",
                "She lives in a small apartment downtown.",
                "The small child could not reach the shelf.",
                "A small mistake caused big problems.",
                "The small town had only one grocery store."
            ],
            "little": [
                "The little dog barked at the mailman.",
                "He had little time to finish the project.",
                "The little girl loved her teddy bear.",
                "There is little chance of success.",
                "The little house was cozy and warm."
            ]
        },
        ("smart", "intelligent"): {
            "smart": [
                "The smart student aced all her exams.",
                "He made a smart investment in technology.",
                "The smart solution solved the complex problem.",
                "She gave a smart answer to the tricky question.",
                "The smart design saved space and money."
            ],
            "intelligent": [
                "The intelligent machine learned from experience.",
                "She is an intelligent researcher in physics.",
                "The intelligent system predicted the outcome.",
                "He gave an intelligent response to the debate.",
                "The intelligent approach simplified the process."
            ]
        }
    }
    
    all_pair_results = {}
    all_similarities = []  # For combined histogram
    
    # Process each word pair
    for word1, word2, synset_name in word_pairs:
        print(f"\n{'='*80}")
        print(f"Processing pair: {word1.upper()} - {word2.upper()}")
        print(f"{'='*80}")
        
        # Get sentences for this pair
        if (word1, word2) in pair_sentences:
            sentences_word1 = pair_sentences[(word1, word2)][word1]
            sentences_word2 = pair_sentences[(word1, word2)][word2]
        else:
            print(f"WARNING: No sentences defined for pair ({word1}, {word2})")
            continue
        
        # Extract vectors for word1
        print(f"\nExtracting vectors for '{word1}'...")
        vectors_word1 = genBERTVector(model, tokenizer, word1, sentences_word1)
        valid_vectors_word1 = [v for v in vectors_word1 if len(v) > 0]
        
        # Extract vectors for word2
        print(f"Extracting vectors for '{word2}'...")
        vectors_word2 = genBERTVector(model, tokenizer, word2, sentences_word2)
        valid_vectors_word2 = [v for v in vectors_word2 if len(v) > 0]
        
        if len(valid_vectors_word1) != 5 or len(valid_vectors_word2) != 5:
            print(f"WARNING: Expected 5 vectors for each word, got {len(valid_vectors_word1)} and {len(valid_vectors_word2)}")
        
        # Calculate similarity between every pair (one from each word)
        # 5 vectors × 5 vectors = 25 pairs
        similarities = []
        for v1 in valid_vectors_word1:
            for v2 in valid_vectors_word2:
                sim = cosine_similarity(v1, v2)
                similarities.append(sim)
        
        # Store results
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        all_pair_results[(word1, word2)] = {
            'sentences_word1': sentences_word1,
            'sentences_word2': sentences_word2,
            'vectors_word1': valid_vectors_word1,
            'vectors_word2': valid_vectors_word2,
            'similarities': similarities,
            'mean': mean_sim,
            'std': std_sim
        }
        
        # Add to combined list
        all_similarities.extend(similarities)
        
        # Print results
        print(f"\nSentences for '{word1}':")
        for idx, sent in enumerate(sentences_word1, 1):
            print(f"  {idx}. {sent}")
        
        print(f"\nSentences for '{word2}':")
        for idx, sent in enumerate(sentences_word2, 1):
            print(f"  {idx}. {sent}")
        
        print(f"\nStatistics for pair ({word1}, {word2}):")
        print(f"  Mean cosine similarity: {mean_sim:.4f}")
        print(f"  Standard deviation: {std_sim:.4f}")
        print(f"  Number of pairs: {len(similarities)}")
        
        # Create histogram for this pair (same format as Part 1)
        plt.figure(figsize=(10, 6))
        bins = np.arange(0, 1.1, 0.1)
        counts, edges, patches = plt.hist(similarities, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Cosine Similarities for Pair: "{word1}" - "{word2}"')
        plt.xticks(bins)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'part3_histogram_{word1}_{word2}.png', dpi=150)
        print(f"\nSaved histogram: part3_histogram_{word1}_{word2}.png")
        plt.close()
    
    # Combined analysis (all 8 pairs, 25 pairs each = 200 pairs total)
    print(f"\n{'='*80}")
    print("COMBINED ANALYSIS (All 8 word pairs, 200 pairs total)")
    print(f"{'='*80}")
    
    overall_mean = np.mean(all_similarities)
    overall_std = np.std(all_similarities)
    
    print(f"\nOverall Statistics:")
    print(f"  Total pairs: {len(all_similarities)}")
    print(f"  Overall mean cosine similarity: {overall_mean:.4f}")
    print(f"  Overall standard deviation: {overall_std:.4f}")
    
    # Create combined histogram
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.1, 0.1)
    counts, edges, patches = plt.hist(all_similarities, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cosine Similarities (Combined - All 8 Word Pairs, 200 pairs)')
    plt.xticks(bins)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('part3_histogram_combined.png', dpi=150)
    print(f"\nSaved combined histogram: part3_histogram_combined.png")
    plt.close()
    
    # Save results to a text file for report
    with open('part3_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PART 3 RESULTS: Comparing BERT vectors for different words with same meaning\n")
        f.write("="*80 + "\n\n")
        
        for (word1, word2), results in all_pair_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Word Pair: {word1.upper()} - {word2.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Sentences for '{word1}':\n")
            for idx, sent in enumerate(results['sentences_word1'], 1):
                f.write(f"  {idx}. {sent}\n")
            
            f.write(f"\nSentences for '{word2}':\n")
            for idx, sent in enumerate(results['sentences_word2'], 1):
                f.write(f"  {idx}. {sent}\n")
            
            f.write(f"\nStatistics for pair ({word1}, {word2}):\n")
            f.write(f"  Mean cosine similarity: {results['mean']:.4f}\n")
            f.write(f"  Standard deviation: {results['std']:.4f}\n")
            f.write(f"  Number of pairs: {len(results['similarities'])}\n")
            f.write("\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("COMBINED ANALYSIS (All 8 word pairs, 200 pairs total)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Overall Statistics:\n")
        f.write(f"  Total pairs: {len(all_similarities)}\n")
        f.write(f"  Overall mean cosine similarity: {overall_mean:.4f}\n")
        f.write(f"  Overall standard deviation: {overall_std:.4f}\n")
    
    print("\nSaved detailed results to: part3_results.txt")
    
    return all_pair_results, overall_mean, overall_std


if __name__ == "__main__":
    # Load BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    print("Model loaded successfully!")
    
    # Run Part 1 analysis
    part1_results, overall_mean, overall_std = part1_analysis(model, tokenizer)
    
    print("\n" + "="*80)
    print("PART 1 COMPLETE")
    print("="*80)
    print("\nHistograms saved as PNG files.")
    
    # Run Part 2 analysis
    part2_similarities, part2_mean, part2_std = part2_analysis(part1_results)
    
    print("\n" + "="*80)
    print("PART 2 COMPLETE")
    print("="*80)
    print("\nPart 2 histogram saved as PNG file.")
    print("\n" + "="*80)
    print("COMPARISON SUMMARY (Part 1 & 2)")
    print("="*80)
    print(f"\nPart 1 (Same word, same meaning):")
    print(f"  Mean: {overall_mean:.4f}, Std: {overall_std:.4f}")
    print(f"\nPart 2 (Different words):")
    print(f"  Mean: {part2_mean:.4f}, Std: {part2_std:.4f}")
    
    # Run Part 3 analysis
    part3_results, part3_mean, part3_std = part3_analysis(model, tokenizer)
    
    print("\n" + "="*80)
    print("PART 3 COMPLETE")
    print("="*80)
    print("\n" + "="*80)
    print("FINAL SUMMARY (All Parts)")
    print("="*80)
    print(f"\nPart 1 (Same word, same meaning):")
    print(f"  Mean: {overall_mean:.4f}, Std: {overall_std:.4f}")
    print(f"\nPart 2 (Different words, different meaning):")
    print(f"  Mean: {part2_mean:.4f}, Std: {part2_std:.4f}")
    print(f"\nPart 3 (Different words, same meaning):")
    print(f"  Mean: {part3_mean:.4f}, Std: {part3_std:.4f}")
   


