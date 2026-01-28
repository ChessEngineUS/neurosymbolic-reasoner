"""Benchmark dataset loaders and processors."""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from pathlib import Path


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
        
        self.questions = self._load_questions()
        self.scene_graphs = self._load_scene_graphs()
        
        print(f"Loaded CLEVR {split}: {len(self.questions)} samples")
    
    def _load_questions(self) -> List[Dict]:
        """Load CLEVR questions."""
        questions_file = self.data_dir / f"questions/CLEVR_{self.split}_questions.json"
        
        if not questions_file.exists():
            print(f"Warning: {questions_file} not found. Generating synthetic data.")
            return self._generate_synthetic_clevr()
        
        with open(questions_file, 'r') as f:
            data = json.load(f)
            questions = data['questions']
            
        if self.max_samples:
            questions = questions[:self.max_samples]
            
        return questions
    
    def _load_scene_graphs(self) -> Dict:
        """Load or generate scene graphs."""
        scenes_file = self.data_dir / f"scenes/CLEVR_{self.split}_scenes.json"
        
        if not scenes_file.exists():
            return {}
        
        with open(scenes_file, 'r') as f:
            data = json.load(f)
            
        return {scene['image_filename']: scene for scene in data['scenes']}
    
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
        
        # Generate synthetic visual features (512-dim)
        visual_features = torch.randn(49, 512)  # 7x7 spatial grid
        
        # Extract reasoning program
        program = question_data.get('program', [])
        
        # Get scene graph if available
        scene_graph = self.scene_graphs.get(question_data['image_filename'], {})
        
        return {
            'visual_features': visual_features,
            'question': question_data['question'],
            'answer': question_data['answer'],
            'program': program,
            'scene_graph': scene_graph,
            'question_type': question_data.get('question_family_index', 0)
        }


class BAbIDataset(Dataset):
    """bAbI dataset for reasoning tasks.
    
    20 tasks testing various reasoning capabilities:
    - Single supporting fact
    - Two supporting facts  
    - Three supporting facts
    - Temporal reasoning
    - Path finding
    - etc.
    """
    
    def __init__(self, task_id: int = 1, split: str = 'train',
                 data_dir: str = './data/babi', max_samples: Optional[int] = None):
        self.task_id = task_id
        self.split = split
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        
        self.stories = self._load_stories()
        print(f"Loaded bAbI task {task_id} {split}: {len(self.stories)} samples")
    
    def _load_stories(self) -> List[Dict]:
        """Load bAbI stories."""
        task_file = self.data_dir / f"qa{self.task_id}_{self.split}.txt"
        
        if not task_file.exists():
            print(f"Warning: {task_file} not found. Generating synthetic data.")
            return self._generate_synthetic_babi()
        
        stories = []
        with open(task_file, 'r') as f:
            current_story = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(' ', 1)
                idx = int(parts[0])
                
                if idx == 1 and current_story:
                    stories.append(self._process_story(current_story))
                    current_story = []
                
                current_story.append(parts[1] if len(parts) > 1 else '')
            
            if current_story:
                stories.append(self._process_story(current_story))
        
        if self.max_samples:
            stories = stories[:self.max_samples]
            
        return stories
    
    def _process_story(self, story: List[str]) -> Dict:
        """Process a single story into context, question, answer."""
        context = []
        question = None
        answer = None
        supporting_facts = []
        
        for line in story:
            if '?' in line:
                parts = line.split('\t')
                question = parts[0]
                if len(parts) > 1:
                    answer = parts[1]
                if len(parts) > 2:
                    supporting_facts = [int(x) for x in parts[2].split()]
            else:
                context.append(line)
        
        return {
            'context': context,
            'question': question,
            'answer': answer,
            'supporting_facts': supporting_facts
        }
    
    def _generate_synthetic_babi(self, num_samples: int = 500) -> List[Dict]:
        """Generate synthetic bAbI-style reasoning data."""
        stories = []
        entities = ['Mary', 'John', 'Sandra', 'Daniel']
        locations = ['kitchen', 'bedroom', 'bathroom', 'garden', 'office']
        objects = ['apple', 'football', 'milk']
        
        for i in range(num_samples if not self.max_samples else min(num_samples, self.max_samples)):
            entity = np.random.choice(entities)
            loc1 = np.random.choice(locations)
            loc2 = np.random.choice(locations, replace=False)
            
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
        
        # Convert text to features
        context_str = ' '.join(story['context'])
        combined_text = context_str + ' ' + story['question']
        
        # Simple bag-of-words features (can be replaced with BERT embeddings)
        features = torch.randn(len(story['context']) + 1, 512)
        
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
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        
        self.data = self._load_data()
        print(f"Loaded VQA {split}: {len(self.data)} samples")
    
    def _load_data(self) -> List[Dict]:
        """Load or generate VQA data."""
        # Generate synthetic VQA data
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
        
        # Generate synthetic visual features
        visual_features = torch.randn(196, 512)  # 14x14 grid
        
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
