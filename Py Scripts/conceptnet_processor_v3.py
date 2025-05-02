import pandas as pd
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List, Tuple
import json
from tqdm import tqdm
import warnings

class ConceptNetStreamProcessor:
    """
    Efficient streaming processor for ConceptNet data that handles large files
    through chunked processing and disk-based operations.
    """
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the processor with input and output directories
        
        Args:
            input_dir: Directory containing raw ConceptNet TSV files
            output_dir: Directory for processed output files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Column names for ConceptNet TSV files
        self.columns = [
            'URI', 'rel', 'start', 'end', 'weight', 
            'source', 'id', 'dataset', 'surfaceText'
        ]
        
    def clean_concept_name(self, concept_str: str) -> str:
        """Extract clean concept name from ConceptNet format"""
        if not isinstance(concept_str, str):
            return "unknown"
        
        parts = concept_str.split('/')
        if len(parts) >= 4:
            concept = parts[-1]
            if '/' in concept:
                concept = concept.split('/')[0]
            return concept.lower().strip()
        return concept_str.lower().strip()
    
    def extract_relation_type(self, relation_str: str) -> str:
        """Extract relation type from ConceptNet format"""
        if not isinstance(relation_str, str):
            return "unknown"
        
        parts = relation_str.split('/')
        if len(parts) >= 3:
            return parts[-1]
        return relation_str
    
    def extract_language(self, concept_str: str) -> str:
        """Extract language from ConceptNet concept URI"""
        if not isinstance(concept_str, str):
            return "unknown"
        
        parts = concept_str.split('/')
        if len(parts) >= 4:
            return parts[2]
        return "unknown"
    
    def parse_weight(self, weight_str: str) -> float:
        """Parse weight value, handling different formats"""
        if isinstance(weight_str, (int, float)):
            return float(weight_str)
        if not isinstance(weight_str, str):
            return 1.0
        
        try:
            return float(weight_str)
        except:
            try:
                weight_data = json.loads(weight_str)
                return float(weight_data.get('weight', 1.0))
            except:
                return 1.0
    
    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of ConceptNet data"""
        try:
            # Clean and extract features
            processed = pd.DataFrame({
                'clean_start': chunk['start'].apply(self.clean_concept_name),
                'clean_end': chunk['end'].apply(self.clean_concept_name),
                'relation': chunk['rel'].apply(self.extract_relation_type),
                'weight': chunk['weight'].apply(self.parse_weight),
                'source_lang': chunk['start'].apply(self.extract_language),
                'target_lang': chunk['end'].apply(self.extract_language)
            })
            
            # Filter invalid entries
            mask = (
                (processed['clean_start'] != 'unknown') &
                (processed['clean_end'] != 'unknown') &
                (processed['source_lang'] != 'unknown') &
                (processed['target_lang'] != 'unknown') &
                (processed['weight'] > 0)
            )
            
            return processed[mask]
            
        except Exception as e:
            warnings.warn(f"Error processing chunk: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_file(
        self,
        lang: str,
        chunk_size: int = 10000,
        min_weight: float = 1.0
    ) -> None:
        """
        Preprocess a ConceptNet file for a specific language
        
        Args:
            lang: Language code (e.g., 'en', 'de')
            chunk_size: Number of rows to process at once
            min_weight: Minimum weight threshold
        """
        input_file = self.input_dir / f'conceptnet-assertions-5.7.0.{lang}.tsv'
        output_file = self.output_dir / f'conceptnet_{lang}_processed.parquet'
        
        if not input_file.exists():
            print(f"Input file not found: {input_file}")
            return
        
        print(f"Processing {lang} ConceptNet data...")
        
        # Process in chunks using dask
        ddf = dd.read_csv(
            input_file,
            sep='\t',
            names=self.columns,
            blocksize='64MB'
        )
        
        # Apply processing to each partition
        processed_ddf = ddf.map_partitions(
            self.process_chunk,
            meta={
                'clean_start': 'str',
                'clean_end': 'str',
                'relation': 'str',
                'weight': 'float64',
                'source_lang': 'str',
                'target_lang': 'str'
            }
        )
        
        # Filter by weight
        processed_ddf = processed_ddf[processed_ddf['weight'] >= min_weight]
        
        # Save to parquet with optimized settings
        processed_ddf.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',
            write_index=False
        )
        
        print(f"Saved processed {lang} data to {output_file}")
    
    def load_processed_data(
        self,
        lang: str,
        sample_size: Optional[float] = None,
        max_rows: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load processed data for a language, optionally with sampling
        
        Args:
            lang: Language code
            sample_size: If provided, fraction of data to sample
            max_rows: If provided, maximum number of rows to load
        """
        input_file = self.output_dir / f'conceptnet_{lang}_processed.parquet'
        
        if not input_file.exists():
            print(f"Processed file not found: {input_file}")
            return None
        
        try:
            # First try using dask for efficient loading
            ddf = dd.read_parquet(str(input_file))
            
            if sample_size:
                # Use frac parameter instead of n for dask sampling
                ddf = ddf.sample(frac=sample_size, random_state=42)
            
            if max_rows:
                ddf = ddf.head(max_rows)
            
            # Convert to pandas DataFrame
            return ddf.compute()
            
        except Exception as e:
            # Fallback to pandas if dask encounters an error
            print(f"Dask loading failed with error: {str(e)}")
            print("Falling back to pandas for data loading")
            
            try:
                # Read directly with pandas
                df = pd.read_parquet(str(input_file))
                
                if sample_size:
                    # Sample using pandas
                    df = df.sample(frac=sample_size, random_state=42)
                
                if max_rows:
                    df = df.head(max_rows)
                
                return df
            except Exception as e2:
                print(f"Failed to load data: {str(e2)}")
                return None

def preprocess_conceptnet(
    input_dir: str = "Data/Input",
    output_dir: str = "Data/Processed",
    languages: List[str] = ['en', 'de'],
    min_weight: float = 1.0
) -> None:
    """
    Helper function to preprocess ConceptNet data for multiple languages
    
    Args:
        input_dir: Directory containing raw ConceptNet files
        output_dir: Directory for processed output
        languages: List of language codes to process
        min_weight: Minimum weight threshold
    """
    processor = ConceptNetStreamProcessor(input_dir, output_dir)
    
    for lang in languages:
        processor.preprocess_file(lang, min_weight=min_weight)
    
    print("Preprocessing complete. Use load_processed_data() to load the results.")