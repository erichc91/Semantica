# Full RSC Semantic Convergence + Syntax Emergence Experiment
# Author: Erich Curtis
# Date: 2025-04-28

import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ------------------- 1. Vocabulary and Expanded Relations -------------------

semantic_fields = {
    "Living Things": ['dog', 'cat', 'horse', 'bird', 'fish', 'teacher', 'parent', 'child', 'friend', 'enemy'],
    "Actions": ['run', 'walk', 'swim', 'teach', 'learn', 'jump', 'help', 'hurt', 'love', 'eat'],
    "Objects": ['chair', 'table', 'cup', 'pen', 'phone', 'backpack', 'house', 'bridge', 'road', 'tree'],
    "Locations": ['forest', 'mountain', 'river', 'valley', 'school', 'city', 'field', 'beach', 'island'],
    "Quantities": ['one', 'two', 'few', 'many', 'all', 'none', 'some'],
    "Modifiers": ['big', 'small', 'fast', 'slow', 'loud', 'quiet', 'happy', 'sad'],
    "Prepositions": ['across', 'under', 'over', 'through', 'beside', 'near', 'around'],
    "Scientific": ['earth', 'sun', 'water', 'ice', 'orbit', 'gravity', 'star', 'cloud'],
    "Social": ['neighbor', 'stranger', 'friend', 'parent', 'enemy'],
    "Emotions": ['happy', 'sad', 'angry', 'tired', 'calm', 'excited', 'afraid', 'safe'],
    "Conditions": ['if', 'when', 'because', 'although', 'unless']
}

# Flat vocabulary
vocab = list(set([item for sublist in semantic_fields.values() for item in sublist]))

# Relations - Simple, Medium, Complex, Conditional
relations = [
    ('dog', 'is', 'mammal'),
    ('car', 'moves_on', 'road'),
    ('bird', 'flies_over', 'river'),
    ('teacher', 'teaches', 'child'),
    ('rain', 'causes', 'wet_ground'),
    ('river', 'flows_into', 'lake'),
    ('phone', 'connects_to', 'internet'),
    ('earth', 'orbits', 'sun'),
    ('sun', 'provides', 'light'),
    ('friend', 'helps', 'friend'),
    ('child', 'plays_with', 'dog'),
    ('parent', 'loves', 'child'),
    ('earth', 'has', 'gravity'),
    ('water', 'freezes_at', '0C'),
    ('mountain', 'stands_over', 'valley'),
    ('wind', 'moves', 'clouds'),
    ('child', 'crosses', 'bridge', 'during rain'),
    ('friend', 'helps', 'friend', 'because of danger'),
    ('sun', 'rises', 'over mountain', 'at morning'),
    ('hunger', 'leads_to', 'eating'),
    ('fish', 'cannot', 'walk'),
    ('if rain', 'then', 'ground becomes wet')
]

# ------------------- 2. Agent Initialization -------------------

random.seed(42)

agent_a_groundings = {word: word for word in vocab}

agent_b_groundings = {}
for word in vocab:
    if random.random() < 0.8:
        agent_b_groundings[word] = word
    else:
        agent_b_groundings[word] = random.choice(vocab)

agent_c_groundings = {}
for word in vocab:
    if random.random() < 0.3:
        agent_c_groundings[word] = word
    else:
        agent_c_groundings[word] = random.choice(vocab)

# ------------------- 3. Learning Phase with RSC Stability -------------------

total_frames = 1500
batch_size = 3

proposals_log_ab = []
proposals_log_ac = []
anchor_log_ab = []
anchor_log_ac = []
accepted_relations_ab = set()
accepted_relations_ac = set()
stability_counter_ab = {}
stability_counter_ac = {}
anchors_ab = set()
anchors_ac = set()
accuracy_over_time_ab = []
accuracy_over_time_ac = []

stability_threshold = 3  # Validations needed to anchor

# Use the ground truth list to validate
ground_truth_set = set(relations)

def validate_relation(*args):
    return args in ground_truth_set


def try_validate(groundings, *words):
    mapped = [groundings[w] if w in groundings else w for w in words]
    return validate_relation(*mapped)

# Learning Loop
for frame in tqdm(range(total_frames), desc="Learning Progress", unit="frame"):
    proposals = random.sample(relations, batch_size)
    
    accepted_ab = 0
    accepted_ac = 0
    
    for proposal in proposals:
        # Agent B Validation
        if try_validate(agent_b_groundings, *proposal):
            accepted_relations_ab.add(proposal)
            stability_counter_ab[proposal] = stability_counter_ab.get(proposal, 0) + 1
            accepted_ab += 1
            proposals_log_ab.append(f"Frame {frame}: {proposal} → ACCEPTED (A↔B)")
            if stability_counter_ab[proposal] == stability_threshold:
                anchors_ab.update(proposal)
                anchor_log_ab.append(f"Frame {frame}: ANCHORED {proposal}")
        else:
            proposals_log_ab.append(f"Frame {frame}: {proposal} → REJECTED (A↔B)")
        
        # Agent C Validation
        if try_validate(agent_c_groundings, *proposal):
            accepted_relations_ac.add(proposal)
            stability_counter_ac[proposal] = stability_counter_ac.get(proposal, 0) + 1
            accepted_ac += 1
            proposals_log_ac.append(f"Frame {frame}: {proposal} → ACCEPTED (A↔C)")
            if stability_counter_ac[proposal] == stability_threshold:
                anchors_ac.update(proposal)
                anchor_log_ac.append(f"Frame {frame}: ANCHORED {proposal}")
        else:
            proposals_log_ac.append(f"Frame {frame}: {proposal} → REJECTED (A↔C)")
    
    accuracy_over_time_ab.append(accepted_ab / batch_size)
    accuracy_over_time_ac.append(accepted_ac / batch_size)
    
    # Drift
    if frame % 25 == 0 and frame != 0:
        for word in random.sample(list(agent_c_groundings.keys()), 5):
            agent_c_groundings[word] = random.choice(vocab)

# ------------------- 4. Save Logs -------------------

with open('negotiation_log_ab.txt', 'w', encoding='utf-8') as f:
    for line in proposals_log_ab:
        f.write(line + '\n')

with open('negotiation_log_ac.txt', 'w', encoding='utf-8') as f:
    for line in proposals_log_ac:
        f.write(line + '\n')

with open('anchor_log_ab.txt', 'w', encoding='utf-8') as f:
    for line in anchor_log_ab:
        f.write(line + '\n')

with open('anchor_log_ac.txt', 'w', encoding='utf-8') as f:
    for line in anchor_log_ac:
        f.write(line + '\n')

print("All logs saved.")

# ------------------- 5. Plot Accuracy -------------------

plt.figure(figsize=(12, 6))
plt.plot(range(total_frames), [sum(accuracy_over_time_ab[:i+1])/(i+1) for i in range(total_frames)], label='Agent A ↔ B', color='green', linewidth=2)
plt.plot(range(total_frames), [sum(accuracy_over_time_ac[:i+1])/(i+1) for i in range(total_frames)], label='Agent A ↔ C', color='red', linewidth=2)
plt.xlabel('Learning Frame', fontsize=14)
plt.ylabel('Cumulative Accuracy', fontsize=14)
plt.title('Accuracy Over Time: Agent A ↔ B vs Agent A ↔ C (with RSC Stability)', fontsize=16)
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
