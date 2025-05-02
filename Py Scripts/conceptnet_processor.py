import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyArrowPatch
import random
import io
import pandas as pd
from PIL import Image, ImageDraw
import os
import time
from tqdm import tqdm
import uuid
import warnings
import traceback
import json
from scipy.spatial.distance import cosine
import matplotlib
import requests
from io import StringIO
import gdown
matplotlib.use('Agg')  # Use non-interactive backend for better memory handling-interactive backend for better memory handling

class ConceptNetProcessor:
    """
    Process ConceptNet data for semantic visualization
    """
    def __init__(self, english_data=None, german_data=None):
        self.english_data = english_data
        self.german_data = german_data
        self.semantic_graph = nx.DiGraph()
        self.concept_vectors = {}
        self.relation_types = set()
        
        print("ConceptNetProcessor initialized")
    
    def clean_concept_name(self, concept_str):
        """Extract clean concept name from ConceptNet format"""
        if not isinstance(concept_str, str):
            return "unknown"
            
        # Extract the concept name from the ConceptNet URI format
        parts = concept_str.split('/')
        if len(parts) >= 4:
            # Format is typically /c/LANG/CONCEPT
            concept = parts[-1]
            # Remove part-of-speech tags if present
            if '/' in concept:
                concept = concept.split('/')[0]
            return concept
        return concept_str
    
    def extract_relation_type(self, relation_str):
        """Extract relation type from ConceptNet format"""
        if not isinstance(relation_str, str):
            return "unknown"
            
        parts = relation_str.split('/')
        if len(parts) >= 3:
            # Format is typically /r/RELATION_TYPE
            return parts[-1]
        return relation_str
    
    def extract_language(self, concept_str):
        """Extract language from ConceptNet concept URI"""
        if not isinstance(concept_str, str):
            return "unknown"
            
        parts = concept_str.split('/')
        if len(parts) >= 4:
            # Format is typically /c/LANG/CONCEPT
            return parts[2]
        return "unknown"
    
    def parse_weight(self, weight_str):
        """Parse weight JSON string to extract numeric weight"""
        if not isinstance(weight_str, str):
            return 1.0
            
        try:
            weight_data = json.loads(weight_str)
            # ConceptNet weights are typically in 'weight' field
            return float(weight_data.get('weight', 1.0))
        except:
            return 1.0
    
    def save_preprocessed_data(self, english_file='english_conceptnet_preprocessed.csv', german_file='german_conceptnet_preprocessed.csv'):
        """Save preprocessed ConceptNet data to CSV files"""
        if self.english_data is not None:
            self.english_data.to_csv(english_file, index=False)
            print(f"Saved English data to {english_file}")
        
        if self.german_data is not None:
            self.german_data.to_csv(german_file, index=False)
        print(f"Saved German data to {german_file}")
    
    def build_semantic_graph(self, max_concepts=200, min_weight=1.0, sample_size=0.25):
        """Build semantic graph from ConceptNet data"""
        print("Building semantic graph from ConceptNet data...")
        
        if self.english_data is None and self.german_data is None:
            print("No ConceptNet data provided.")
            return
        
        # Combine datasets
        all_data = []
        if self.english_data is not None:
            print(f"Processing {len(self.english_data)} English ConceptNet assertions...")
            sample_size_en = int(len(self.english_data) * sample_size)
            print(f"Will sample {sample_size_en} English assertions")
            all_data.append(('en', self.english_data))
        
        if self.german_data is not None:
            print(f"Processing {len(self.german_data)} German ConceptNet assertions...")
            sample_size_de = int(len(self.german_data) * sample_size)
            print(f"Will sample {sample_size_de} German assertions")
            all_data.append(('de', self.german_data))
        
        # Track concepts and their occurrence count
        concept_counts = {}
        
        # Process each language dataset
        for lang, data in all_data:
            curr_sample_size = int(len(data) * sample_size)
            data_sample = data.sample(n=curr_sample_size, random_state=42)
            print(f"Sampling {curr_sample_size} assertions from {len(data)} {lang} assertions")
            
            # Process assertions
            for _, row in tqdm(data_sample.iterrows(), desc=f"Processing {lang} assertions", total=len(data_sample)):
                try:
                    # Extract source and target concepts
                    source_concept = self.clean_concept_name(row['start'])
                    target_concept = self.clean_concept_name(row['end'])
                    
                    # Extract relation type
                    relation_type = self.extract_relation_type(row['rel'])
                    self.relation_types.add(relation_type)
                    
                    # Extract languages
                    source_lang = self.extract_language(row['start'])
                    target_lang = self.extract_language(row['end'])
                    
                    # Parse weight
                    weight = self.parse_weight(row['weight'])
                    
                    # Skip low-weight relationships
                    if weight < min_weight:
                        continue
                    
                    # Track concept occurrences
                    concept_counts[source_concept] = concept_counts.get(source_concept, 0) + 1
                    concept_counts[target_concept] = concept_counts.get(target_concept, 0) + 1
                    
                    # Add to graph
                    self.semantic_graph.add_node(
                        source_concept,
                        lang=source_lang,
                        count=concept_counts[source_concept]
                    )
                    
                    self.semantic_graph.add_node(
                        target_concept,
                        lang=target_lang,
                        count=concept_counts[target_concept]
                    )
                    
                    # Add edge with relation data
                    self.semantic_graph.add_edge(
                        source_concept,
                        target_concept,
                        relation=relation_type,
                        weight=weight
                    )
                    
                except Exception as e:
                    warnings.warn(f"Error processing assertion: {e}")
        
        # Limit to top concepts if needed
        if len(concept_counts) > max_concepts:
            print(f"Limiting graph to top {max_concepts} concepts...")
            top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:max_concepts]
            top_concept_names = {c[0] for c in top_concepts}
            
            # Create subgraph with only top concepts
            subgraph = nx.DiGraph()
            
            for node in top_concept_names:
                if self.semantic_graph.has_node(node):
                    subgraph.add_node(
                        node,
                        **self.semantic_graph.nodes[node]
                    )
            
            for source, target, data in self.semantic_graph.edges(data=True):
                if source in top_concept_names and target in top_concept_names:
                    subgraph.add_edge(
                        source,
                        target,
                        **data
                    )
            
            self.semantic_graph = subgraph
        
        print(f"Semantic graph built with {self.semantic_graph.number_of_nodes()} nodes and {self.semantic_graph.number_of_edges()} edges")
        
        # Infer semantic categories
        self.infer_semantic_categories()
        
        return self.semantic_graph
    
    def infer_semantic_categories(self):
        """Infer semantic categories for concepts based on relationships"""
        print("Inferring semantic categories...")
        categories = {}
        
        # Count relationship types for each concept
        for node in self.semantic_graph.nodes():
            # Initialize as generic
            categories[node] = 'generic'
            
            # Get all relationships involving this concept
            in_edges = self.semantic_graph.in_edges(node, data=True)
            out_edges = self.semantic_graph.out_edges(node, data=True)
            
            # Count relationship types
            person_relations = 0
            place_relations = 0
            animal_relations = 0
            
            for _, _, data in in_edges:
                rel = data.get('relation', '')
                if rel in {'IsA/person', 'CapableOf', 'HasA'}:
                    person_relations += 1
                elif rel in {'AtLocation', 'LocatedNear', 'HasA'}:
                    place_relations += 1
                elif rel in {'IsA/animal', 'CapableOf'}:
                    animal_relations += 1
            
            for _, _, data in out_edges:
                rel = data.get('relation', '')
                if rel in {'IsA/person', 'CapableOf', 'HasA'}:
                    person_relations += 1
                elif rel in {'AtLocation', 'LocatedNear', 'HasA'}:
                    place_relations += 1
                elif rel in {'IsA/animal', 'CapableOf'}:
                    animal_relations += 1
            
            # Assign category based on dominant relationships
            max_relations = max(person_relations, place_relations, animal_relations)
            if max_relations > 0:
                if max_relations == person_relations:
                    categories[node] = 'person'
                elif max_relations == place_relations:
                    categories[node] = 'place'
                elif max_relations == animal_relations:
                    categories[node] = 'animal'
        
        # Update graph with categories
        nx.set_node_attributes(self.semantic_graph, categories, 'category')
        
        # Print category statistics
        category_counts = {}
        for cat in categories.values():
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"Inferred categories: {category_counts}")
    
    def compute_important_relationships(self, threshold=0.5, max_relationships=30):
        """Compute the most important relationships between concepts based on vector similarity"""
        important_relationships = []
        
        # Get all pairs of concepts
        concepts = list(self.concept_vectors.keys())
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Get vectors
                vec1 = self.concept_vectors[concept1]['vector']
                vec2 = self.concept_vectors[concept2]['vector']
                
                # Compute cosine similarity
                similarity = 1 - cosine(vec1, vec2)
                
                if similarity > threshold:
                    important_relationships.append({
                        'source': concept1,
                        'target': concept2,
                        'similarity': similarity
                    })
        
        # Sort by similarity and take top N
        important_relationships.sort(key=lambda x: x['similarity'], reverse=True)
        return important_relationships[:max_relationships]
    
    def generate_concept_vectors(self, dimensions=5):
        """Generate concept vectors based on graph structure"""
        print(f"Generating {dimensions}-dimensional concept vectors...")
        
        # Use node2vec or similar embedding
        nodes = list(self.semantic_graph.nodes())
        
        # Simple embedding based on connectivity patterns
        adjacency_matrix = nx.adjacency_matrix(self.semantic_graph).todense()
        
        # Use SVD to reduce dimensionality
        U, _, _ = np.linalg.svd(adjacency_matrix)
        embeddings = U[:, :dimensions]
        
        # Normalize embeddings
        embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
        
        # Convert to dictionary with structured data
        for i, node in enumerate(nodes):
            self.concept_vectors[node] = {
                'vector': embeddings[i],
                'category': self.semantic_graph.nodes[node].get('category', 'generic')
            }
        
        print(f"Generated vectors for {len(self.concept_vectors)} concepts")
        return self.concept_vectors
    
class ConceptNetProcessor_v2:
    """
    Process ConceptNet data for semantic visualization
    """
    def __init__(self, english_data=None, german_data=None):
        self.english_data = english_data
        self.german_data = german_data
        self.semantic_graph = nx.DiGraph()
        self.concept_vectors = {}
        self.relation_types = set()
        
        print("ConceptNetProcessor initialized")
    
    def save_preprocessed_data(self, english_file='english_conceptnet_preprocessed.csv', german_file='german_conceptnet_preprocessed.csv'):
        """Save preprocessed ConceptNet data to CSV files"""
        if self.english_data is not None:
            self.english_data.to_csv(english_file, index=False)
            print(f"Saved English data to {english_file}")
        
        if self.german_data is not None:
            self.german_data.to_csv(german_file, index=False)
        print(f"Saved German data to {german_file}")
    
    def clean_concept_name(self, concept_str):
        """Extract clean concept name from ConceptNet format"""
        if not isinstance(concept_str, str):
            return "unknown"

        # Extract the concept name from the ConceptNet URI format
        parts = concept_str.split('/')
        if len(parts) >= 4:
            # Format is typically /c/LANG/CONCEPT
            concept = parts[-1]
            # Remove part-of-speech tags if present
            if '/' in concept:
                concept = concept.split('/')[0]
            return concept.lower().strip()  # Normalize to lowercase and strip whitespace
        return concept_str.lower().strip()
    
    def extract_relation_type(self, relation_str):
        """Extract relation type from ConceptNet format"""
        if not isinstance(relation_str, str):
            return "unknown"
            
        parts = relation_str.split('/')
        if len(parts) >= 3:
            # Format is typically /r/RELATION_TYPE
            return parts[-1]
        return relation_str
    
    def extract_language(self, concept_str):
        """Extract language from ConceptNet concept URI"""
        if not isinstance(concept_str, str):
            return "unknown"
            
        parts = concept_str.split('/')
        if len(parts) >= 4:
            # Format is typically /c/LANG/CONCEPT
            return parts[2]
        return "unknown"
    
    def parse_weight(self, weight_str):
        """Parse weight JSON string to extract numeric weight"""
        if not isinstance(weight_str, str):
            return 1.0
            
        try:
            weight_data = json.loads(weight_str)
            # ConceptNet weights are typically in 'weight' field
            return float(weight_data.get('weight', 1.0))
        except:
            return 1.0
    
    def build_semantic_graph(self, max_concepts=200, min_weight=0.5, sample_size=0.25):
        """Build semantic graph from ConceptNet data"""
        print("Building semantic graph from ConceptNet data...")

        if self.english_data is None and self.german_data is None:
            print("No ConceptNet data provided.")
            return

        # Combine datasets
        all_data = []
        if self.english_data is not None:
            print(f"Processing {len(self.english_data)} English ConceptNet assertions...")
            sample_size_en = int(len(self.english_data) * sample_size)
            print(f"Will sample {sample_size_en} English assertions")
            all_data.append(('en', self.english_data))

        if self.german_data is not None:
            print(f"Processing {len(self.german_data)} German ConceptNet assertions...")
            sample_size_de = int(len(self.german_data) * sample_size)
            print(f"Will sample {sample_size_de} German assertions")
            all_data.append(('de', self.german_data))

        # Track concepts and their occurrence count
        concept_counts = {}

        # Process each language dataset
        for lang, data in all_data:
            curr_sample_size = int(len(data) * sample_size)
            data_sample = data.sample(n=curr_sample_size, random_state=42)
            print(f"Sampling {curr_sample_size} assertions from {len(data)} {lang} assertions")

            # Process assertions
            for _, row in tqdm(data_sample.iterrows(), desc=f"Processing {lang} assertions", total=len(data_sample)):
                try:
                    # Extract source and target concepts
                    source_concept = self.clean_concept_name(row['start'])
                    target_concept = self.clean_concept_name(row['end'])

                    # Extract relation type
                    relation_type = self.extract_relation_type(row['rel'])
                    self.relation_types.add(relation_type)

                    # Extract languages
                    source_lang = self.extract_language(row['start'])
                    target_lang = self.extract_language(row['end'])

                    # Parse weight
                    weight = self.parse_weight(row['meta'])

                    # Skip low-weight relationships
                    if weight < min_weight:
                        continue

                    # Track concept occurrences
                    concept_counts[source_concept] = concept_counts.get(source_concept, 0) + 1
                    concept_counts[target_concept] = concept_counts.get(target_concept, 0) + 1

                    # Add to graph
                    self.semantic_graph.add_node(
                        source_concept,
                        lang=source_lang,
                        count=concept_counts[source_concept]
                    )

                    self.semantic_graph.add_node(
                        target_concept,
                        lang=target_lang,
                        count=concept_counts[target_concept]
                    )

                    # Add edge with relation data
                    self.semantic_graph.add_edge(
                        source_concept,
                        target_concept,
                        relation=relation_type,
                        weight=weight
                    )

                except Exception as e:
                    warnings.warn(f"Error processing assertion: {e}")

        # Limit to top concepts if needed
        if len(concept_counts) > max_concepts:
            print(f"Limiting graph to top {max_concepts} concepts...")
            top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:max_concepts]
            top_concept_names = {c[0] for c in top_concepts}

            # Create subgraph with only top concepts
            subgraph = nx.DiGraph()

            for node in top_concept_names:
                if self.semantic_graph.has_node(node):
                    subgraph.add_node(
                        node,
                        **self.semantic_graph.nodes[node]
                    )

            for source, target, data in self.semantic_graph.edges(data=True):
                if source in top_concept_names and target in top_concept_names:
                    subgraph.add_edge(
                        source,
                        target,
                        **data
                    )

            self.semantic_graph = subgraph

        print(f"Semantic graph built with {self.semantic_graph.number_of_nodes()} nodes and {self.semantic_graph.number_of_edges()} edges")

        # Infer semantic categories
        self.infer_semantic_categories()

        return self.semantic_graph

    def infer_semantic_categories(self):
        """Infer semantic categories for concepts based on relationships"""
        print("Inferring semantic categories...")
        categories = {}

        # Count relationship types for each concept
        for node in self.semantic_graph.nodes():
            # Initialize as generic
            categories[node] = 'generic'

            # Get all relationships involving this concept
            in_edges = self.semantic_graph.in_edges(node, data=True)
            out_edges = self.semantic_graph.out_edges(node, data=True)

            # Count relationship types
            person_relations = 0
            place_relations = 0
            animal_relations = 0

            for _, _, data in in_edges:
                rel = data.get('relation', '')
                if rel in {'IsA/person', 'CapableOf', 'HasA'}:
                    person_relations += 1
                elif rel in {'AtLocation', 'LocatedNear', 'HasA'}:
                    place_relations += 1
                elif rel in {'IsA/animal', 'CapableOf'}:
                    animal_relations += 1

            for _, _, data in out_edges:
                rel = data.get('relation', '')
                if rel in {'IsA/person', 'CapableOf', 'HasA'}:
                    person_relations += 1
                elif rel in {'AtLocation', 'LocatedNear', 'HasA'}:
                    place_relations += 1
                elif rel in {'IsA/animal', 'CapableOf'}:
                    animal_relations += 1

            # Assign category based on dominant relationships
            max_relations = max(person_relations, place_relations, animal_relations)
            if max_relations > 0:
                if max_relations == person_relations:
                    categories[node] = 'person'
                elif max_relations == place_relations:
                    categories[node] = 'place'
                elif max_relations == animal_relations:
                    categories[node] = 'animal'

        # Update graph with categories
        nx.set_node_attributes(self.semantic_graph, categories, 'category')

        # Print category statistics
        category_counts = {}
        for cat in categories.values():
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"Inferred categories: {category_counts}")

    def compute_important_relationships(self, threshold=0.5, max_relationships=30):
        """Compute the most important relationships between concepts based on vector similarity"""
        important_relationships = []

        # Get all pairs of concepts
        concepts = list(self.concept_vectors.keys())
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Get vectors
                vec1 = self.concept_vectors[concept1]['vector']
                vec2 = self.concept_vectors[concept2]['vector']

                # Compute cosine similarity
                similarity = 1 - cosine(vec1, vec2)

                if similarity > threshold:
                    important_relationships.append({
                        'source': concept1,
                        'target': concept2,
                        'similarity': similarity
                    })

        # Sort by similarity and take top N
        important_relationships.sort(key=lambda x: x['similarity'], reverse=True)
        return important_relationships[:max_relationships]

    def generate_concept_vectors(self, dimensions=5):
        """Generate concept vectors based on graph structure"""
        print(f"Generating {dimensions}-dimensional concept vectors...")

        # Use node2vec or similar embedding
        nodes = list(self.semantic_graph.nodes())

        # Simple embedding based on connectivity patterns
        adjacency_matrix = nx.adjacency_matrix(self.semantic_graph).todense()

        # Use SVD to reduce dimensionality
        U, _, _ = np.linalg.svd(adjacency_matrix)
        embeddings = U[:, :dimensions]

        # Normalize embeddings
        embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)

        # Convert to dictionary with structured data
        for i, node in enumerate(nodes):
            self.concept_vectors[node] = {
                'vector': embeddings[i],
                'category': self.semantic_graph.nodes[node].get('category', 'generic')
            }

        print(f"Generated vectors for {len(self.concept_vectors)} concepts")
        return self.concept_vectors