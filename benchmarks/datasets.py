"""Benchmark dataset loaders and processors."""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from pathlib import Path


def collate_benchmark_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function for benchmark datasets.
    
    Handles variable-length sequences and different data types.
    """
    # Check what keys we have
    keys = batch[0].keys()
    
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # Handle tensors
        if isinstance(values[0], torch.Tensor):
            # Stack if same size, otherwise keep as list
            try:
                collated[key] = torch.stack(values)
            except:
                collated[key] = values
        
        # Handle strings
        elif isinstance(values[0], str):
            collated[key] = values
        
        # Handle lists
        elif isinstance(values[0], list):
            collated[key] = values
        
        # Handle dicts
        elif isinstance(values[0], dict):
            collated[key] = values
        
        # Handle numbers
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        
        else:
            collated[key] = values
    
    return collated


class CLEVRDataset(Dataset):
    """CLEVR dataset for visual reasoning.
    
    CLEVR (Compositional Language and Elementary Visual Reasoning) tests
    compositional question answering over synthetic images.
    """
    
    def __init__(self, split: str = 'train', data_dir: str = './data/clevr',
                 max_samples: Optional[int] = None):
        self.split = split
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        
        self.questions = self._generate_synthetic_clevr()
        
        print(f"Loaded CLEVR {split}: {len(self.questions)} samples")
    
    def _generate_synthetic_clevr(self, num_samples: int = 1000) -> List[Dict]:
        """Generate synthetic CLEVR-style data for testing."""
        questions = []
        question_types = ['count', 'exist', 'compare_attribute', 'query_attribute']
        
        for i in range(num_samples if not self.max_samples else min(num_samples, self.max_samples)):
            qtype = np.random.choice(question_types)
            
            if qtype == 'count':
                question = "How many red spheres are there?"
                answer = str(np.random.randint(0, 5))
                program = ['filter_color[red]', 'filter_shape[sphere]', 'count']
            elif qtype == 'exist':
                question = "Is there a blue cube?"
                answer = np.random.choice(['yes', 'no'])
                program = ['filter_color[blue]', 'filter_shape[cube]', 'exist']
            elif qtype == 'compare_attribute':
                question = "Is the red sphere larger than the blue cube?"
                answer = np.random.choice(['yes', 'no'])
                program = ['filter_color[red]', 'filter_shape[sphere]', 'filter_color[blue]', 
                          'filter_shape[cube]', 'compare_size']
            else:
                question = "What color is the large sphere?"
                answer = np.random.choice(['red', 'blue', 'green', 'yellow'])
                program = ['filter_size[large]', 'filter_shape[sphere]', 'query_color']
            
            questions.append({
                'question': question,
                'answer': answer,
                'question_family_index': question_types.index(qtype),
                'program': program,
                'image_filename': f'CLEVR_synthetic_{i:06d}.png'
            })
        
        return questions
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        question_data = self.questions[idx]
        
        # Generate synthetic visual features (512-dim) with fixed size
        visual_features = torch.randn(16, 512)  # Fixed sequence length
        
        return {
            'visual_features': visual_features,
            'question': question_data['question'],
            'answer': question_data['answer'],
            'program': question_data.get('program', []),
            'question_type': question_data.get('question_family_index', 0)
        }


class BAbIDataset(Dataset):
    """bAbI dataset for reasoning tasks."""
    
    def __init__(self, task_id: int = 1, split: str = 'train',
                 data_dir: str = './data/babi', max_samples: Optional[int] = None):
        self.task_id = task_id
        self.split = split
        self.max_samples = max_samples
        
        self.stories = self._generate_synthetic_babi()
        print(f"Loaded bAbI task {task_id} {split}: {len(self.stories)} samples")
    
    def _generate_synthetic_babi(self, num_samples: int = 500) -> List[Dict]:
        """Generate synthetic bAbI-style reasoning data."""
        stories = []
        entities = ['Mary', 'John', 'Sandra', 'Daniel']
        locations = ['kitchen', 'bedroom', 'bathroom', 'garden', 'office']
        objects = ['apple', 'football', 'milk']
        
        for i in range(num_samples if not self.max_samples else min(num_samples, self.max_samples)):
            entity = np.random.choice(entities)
            loc1 = np.random.choice(locations)
            loc2 = np.random.choice(locations)
            
            context = [
                f"{entity} went to the {loc1}.",
                f"{entity} picked up the {np.random.choice(objects)}.",
                f"{entity} travelled to the {loc2}."
            ]
            
            question = f"Where is {entity}?"
            answer = loc2
            
            stories.append({
                'context': context,
                'question': question,
                'answer': answer,
                'supporting_facts': [1, 3]
            })
        
        return stories
    
    def __len__(self) -> int:
        return len(self.stories)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        story = self.stories[idx]
        
        # Fixed size features for batching
        features = torch.randn(16, 512)
        
        return {
            'features': features,
            'context': story['context'],
            'question': story['question'],
            'answer': story['answer'],
            'supporting_facts': story['supporting_facts']
        }


class VisualQADataset(Dataset):
    """Visual Question Answering dataset wrapper."""
    
    def __init__(self, split: str = 'train', data_dir: str = './data/vqa',
                 max_samples: Optional[int] = None):
        self.split = split
        self.max_samples = max_samples
        
        self.data = self._load_data()
        print(f"Loaded VQA {split}: {len(self.data)} samples")
    
    def _load_data(self) -> List[Dict]:
        """Generate synthetic VQA data."""
        num_samples = 1000 if not self.max_samples else self.max_samples
        data = []
        
        question_templates = [
            "What color is the {}?",
            "How many {} are in the image?",
            "Is there a {} in the image?",
            "Where is the {}?"
        ]
        
        objects = ['car', 'person', 'dog', 'cat', 'tree', 'building']
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
        
        for i in range(num_samples):
            obj = np.random.choice(objects)
            q_template = np.random.choice(question_templates)
            question = q_template.format(obj)
            
            if 'color' in question:
                answer = np.random.choice(colors)
            elif 'many' in question:
                answer = str(np.random.randint(0, 10))
            elif 'Is there' in question:
                answer = np.random.choice(['yes', 'no'])
            else:
                answer = np.random.choice(['left', 'right', 'center', 'top', 'bottom'])
            
            data.append({
                'question': question,
                'answer': answer,
                'image_id': f'img_{i:06d}'
            })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Fixed size visual features for batching
        visual_features = torch.randn(16, 512)
        
        return {
            'visual_features': visual_features,
            'question': item['question'],
            'answer': item['answer']
        }


def load_benchmark_dataset(name: str, split: str = 'train', 
                          max_samples: Optional[int] = None) -> Dataset:
    """Load a benchmark dataset by name.
    
    Args:
        name: Dataset name ('clevr', 'babi', 'vqa')
        split: Data split ('train', 'val', 'test')
        max_samples: Maximum number of samples to load
        
    Returns:
        Dataset instance
    """
    if name.lower() == 'clevr':
        return CLEVRDataset(split=split, max_samples=max_samples)
    elif name.lower() == 'babi':
        return BAbIDataset(task_id=1, split=split, max_samples=max_samples)
    elif name.lower() == 'vqa':
        return VisualQADataset(split=split, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {name}")
