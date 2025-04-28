# Full RSC Semantic Convergence Experiment
# Built for Agent A ↔ B Learning Validation
# Author: Erich Curtis
# Date: 2025-04-28

import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ------------------- 1. Vocabulary and Semantic Field Setup -------------------

semantic_fields = {
    "Living Things": ['dog', 'cat', 'bird', 'fish', 'whale', 'wolf', 'tiger', 'horse', 'rabbit', 'bear'],
    "Transportation": ['car', 'truck', 'bicycle', 'airplane', 'train', 'boat', 'ship', 'bridge', 'tunnel', 'road'],
    "Natural Features": ['river', 'mountain', 'forest', 'desert', 'beach', 'lake', 'island', 'valley', 'waterfall', 'canyon'],
    "Human Structures": ['house', 'castle', 'hut', 'skyscraper', 'tent', 'apartment', 'building', 'school', 'hospital', 'station'],
    "Communication Devices": ['phone', 'radio', 'television', 'computer', 'internet', 'camera', 'microphone', 'speaker', 'satellite', 'signal'],
    "Weather and Sky": ['rain', 'snow', 'storm', 'cloud', 'sun', 'moon', 'star', 'wind', 'hurricane', 'tornado'],
    "Objects and Tools": ['chair', 'table', 'bed', 'lamp', 'pen', 'pencil', 'hammer', 'screwdriver', 'rope', 'backpack']
}

# Mix some fields (e.g., river belongs to both Natural Features and Transportation)
field_overlaps = {
    'river': ['Natural Features', 'Transportation'],
    'mountain': ['Natural Features', 'Weather and Sky'],
    'phone': ['Communication Devices', 'Objects and Tools'],
    'house': ['Human Structures', 'Objects and Tools'],
}

# Create master vocabulary
vocab = []
for field, words in semantic_fields.items():
    vocab.extend(words)
vocab = list(set(vocab))  # Remove any duplicates

# ------------------- 2. Agent Initialization -------------------

random.seed(42)

# Agent A: baseline mappings (symbol to symbol)
agent_a_groundings = {word: word for word in vocab}

# Agent B: slightly noisy starting mappings
agent_b_groundings = {}
for word in vocab:
    if random.random() < 0.8:
        agent_b_groundings[word] = word  # 80% correct
    else:
        agent_b_groundings[word] = random.choice(vocab)  # 20% wrong guess

# ------------------- 3. Learning and Negotiation Phase -------------------

# Parameters
total_frames = 200
batch_size = 5  # Number of proposals per frame

# Tracking
proposals_log = []
accepted_relations = set()
accuracy_over_time = []

# Helper: simple relational consistency check
def validate_relation(w1, w2):
    """Accept relation if words belong to compatible semantic fields."""
    fields_w1 = [field for field, words in semantic_fields.items() if w1 in words]
    fields_w2 = [field for field, words in semantic_fields.items() if w2 in words]
    # Check for field overlap
    return bool(set(fields_w1) & set(fields_w2))

# Learning Loop
for frame in range(total_frames):
    proposals = random.sample(vocab, batch_size)
    accepted = 0
    for w1 in proposals:
        guess_w2 = agent_b_groundings[w1]
        if validate_relation(w1, guess_w2):
            accepted_relations.add((w1, guess_w2))
            accepted += 1
            proposals_log.append(f"Frame {frame}: PROPOSED ({w1} ↔ {guess_w2}) → ACCEPTED")
        else:
            proposals_log.append(f"Frame {frame}: PROPOSED ({w1} ↔ {guess_w2}) → REJECTED")
    accuracy = accepted / batch_size
    accuracy_over_time.append(accuracy)

# ------------------- 4. Save Negotiation Log -------------------

with open('negotiation_log.txt', 'w') as log_file:
    for line in proposals_log:
        log_file.write(line + '\n')

print("Negotiation log saved to 'negotiation_log.txt'.")

# ------------------- 5. Final Graph Construction -------------------

# Build Graph
G = nx.Graph()
for w1, w2 in accepted_relations:
    G.add_edge(w1, w2)

# Layout
pos = nx.spring_layout(G, seed=42)

# Color Edges Based on Validation Order
num_edges = len(G.edges())
colors = plt.cm.viridis(np.linspace(0, 1, num_edges))

# Draw Final Semantic Graph
plt.figure(figsize=(20, 14))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
nx.draw_networkx_labels(G, pos, font_size=8)

# Sort edges by order for color mapping
edges = list(G.edges())
for idx, edge in enumerate(edges):
    nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=[colors[idx]], width=2)

plt.title("Final Structured Semantic Field Graph: Agent A ↔ B", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.show()

# ------------------- 6. Accuracy Over Time Plot -------------------

# Cumulative average accuracy over time
cumulative_accuracy = [sum(accuracy_over_time[:i+1])/(i+1) for i in range(len(accuracy_over_time))]

plt.figure(figsize=(12, 6))
plt.plot(range(total_frames), cumulative_accuracy, color='green', linewidth=2)
plt.xlabel('Learning Frame', fontsize=14)
plt.ylabel('Cumulative Accuracy', fontsize=14)
plt.title('Accuracy Over Time: Structured Relational Learning', fontsize=16)
plt.grid(True)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
