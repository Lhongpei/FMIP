from torch_geometric.data import InMemoryDataset
import os
import torch
import pandas as pd
from tqdm import tqdm
import pickle
import math
import hashlib
import json
import fcntl
import tempfile
from contextlib import contextmanager
from fmip.utils.milp_reader import MIPmodel
from torch_geometric.data import HeteroData, Batch
from fmip.utils.utils import *


@contextmanager
def _cache_lock(path):
    """Serialize initialization across Slurm ranks and independent processes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _atomic_write(path, writer):
    """Publish a complete cache file without exposing a partial serialization."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.cache-', dir=directory)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _problem_inventory(root):
    directory = os.path.join(root, 'problem')
    if not os.path.isdir(directory):
        # Without a raw inventory, callers need the original explicit file list
        # to locate a cache that was keyed by that inventory.
        return None
    supported = {'.lp', '.mps', '.mps.gz', '.proto.lp'}
    return sorted(name for name in os.listdir(directory)
                  if get_file_extension(name) in supported)


class GraphMILP(HeteroData):
    ''''
    The class for the MILP graph data
    The structure of the graph is as follows:
    - cons: constraints node [right-hand side]
    - Cvars: continuous variables node [varLB, varUB, obj_coef, hasLB, hasUB, current_sol]
    - Ivars: integer variables node [varLB, varUB, obj_coef, hasLB, hasUB, current_sol]
    '''
    def __init__(self, **kwargs):
        super(GraphMILP, self).__init__(**kwargs)
        self.only_discrete = False
        
    def extract_Ivars_label(self):
        return self['Ivars'].x[:, -1].clone()
    
    def extract_Cvars_label(self):
        return self['Cvars'].x[:, -1].clone()
    
    def input_Ivars_label_(self, label):
        self['Ivars'].x[:, -1] = label
        
    def input_Cvars_label_(self, label):
        self['Cvars'].x[:, -1] = label
        
    def projection_(self):
        proj_Ivars, proj_Cvars = self.projection()
        self['Ivars'].x[:, -1] = proj_Ivars
        self['Cvars'].x[:, -1] = proj_Cvars

    def _variable_bounds(self, node_type):
        features = self[node_type].x
        # Missing bounds are encoded as zero, with separate presence flags.
        lower = torch.where(features[:, 3].bool(), features[:, 0], -torch.inf)
        upper = torch.where(features[:, 4].bool(), features[:, 1], torch.inf)
        return lower, upper
        
    def projection(self):
        proj_Cvars = self.projection_Cvars(self['Cvars'].x[:, -1])
        lower, upper = self._variable_bounds('Ivars')
        proj_Ivars = torch.clip(self['Ivars'].x[:, -1], lower, upper)
        return proj_Ivars, proj_Cvars
        
    def projection_Cvars(self, Cvars):
        lower, upper = self._variable_bounds('Cvars')
        proj_Cvars = torch.clip(Cvars, lower, upper)
        return proj_Cvars
    
    def projection_Ivars(self, Ivars):
        if not torch.isfinite(Ivars).all() or (Ivars < 0).any():
            raise ValueError('Integer probabilities must be finite and nonnegative')
        classes = torch.arange(Ivars.shape[-1], device=Ivars.device)
        lower, upper = self._variable_bounds('Ivars')
        valid = (classes >= lower.unsqueeze(-1)) & (classes <= upper.unsqueeze(-1))
        if not valid.any(dim=-1).all():
            raise ValueError('Integer bounds contain no class represented by the model')
        probs = Ivars.masked_fill(~valid, 0)
        total = probs.sum(dim=-1, keepdim=True)
        fallback = valid.to(Ivars.dtype) / valid.sum(dim=-1, keepdim=True)
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        return torch.where(total > 0, probs / denominator, fallback)
    
def toHeteroData(x_cons, x_Cvars, x_Ivars, 
                 C2cons_edge_attr, I2cons_edge_attr, 
                 C2cons_edge_index, I2cons_edge_index, model_sense=1, objective_constant=0.):
    data = GraphMILP()
    data['cons'].x = x_cons
    data['Cvars'].x = x_Cvars
    data['Ivars'].x = x_Ivars
    data['Cvars', 'to', 'cons'].edge_index = C2cons_edge_index
    data['Ivars', 'to', 'cons'].edge_index = I2cons_edge_index
    data['Cvars', 'to', 'cons'].edge_attr = C2cons_edge_attr
    data['Ivars', 'to', 'cons'].edge_attr = I2cons_edge_attr
    data.model_sense = model_sense
    data.objective_constant = objective_constant
    return data

def one_hot_transform_label(x: torch.Tensor) -> torch.Tensor:

    label = x[:, -1].long()
    one_hot_label = torch.nn.functional.one_hot(label, num_classes=2).float()
    x_new = torch.cat([x[:, :-1], one_hot_label], dim=1)
    return x_new
class GraphMILPDataset(InMemoryDataset):

    def __init__(self, root, saved_path, reprocess=False, file_name_list = None, given_solution = True, prefix = ''):
        self.root = root
        self.max_integer_num = None
        self.saved_path = saved_path
        self.problem_root = os.path.join(root, 'problem')
        self.solution_root = os.path.join(root, 'solution') if given_solution else None
        self.file_name_list = (_problem_inventory(root) if file_name_list is None
                               else list(file_name_list))
        self.given_solution = given_solution
        self.prefix = prefix
        with _cache_lock(self.processed_dir + '.lock'):
            if reprocess:
                self._clear_processed_files()
                print('Reprocessing the dataset')
            print(f'Initializing dataset from {root}')
            super(GraphMILPDataset, self).__init__(root)

            # This is our own local cache of PyG objects, not a weights-only file.
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            self.calculate_max_integer_()

    @property
    def processed_dir(self):
        return self.cache_directory(self.root, self.saved_path, self.prefix,
                                    self.file_name_list, self.given_solution)

    @staticmethod
    def cache_directory(root, saved_path, prefix='', file_name_list=None, given_solution=True):
        selection = _problem_inventory(root) if file_name_list is None else list(file_name_list)
        identity = [os.path.realpath(root), selection, bool(given_solution)]
        digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
        return os.path.join(saved_path, prefix + '_processed', digest)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # v3 also records objective sense/constant and validates solver labels.
        # Keep old files intact, but never silently reuse those graph encodings.
        return ['data_v3.pt']

    def calculate_max_integer_(self):
        features = self._data['Ivars'].x
        if features.shape[0] == 0:
            raise ValueError('The dataset must contain integer variables')
        slices = getattr(self, 'slices', None)
        if slices is not None:
            pointers = slices['Ivars']['x']
            if ((pointers[1:] - pointers[:-1]) == 0).any():
                raise ValueError('Every FMIP dataset graph must contain at least one integer variable')
        labels = features[:, -1]
        if not torch.isfinite(labels).all() or not torch.isclose(labels, labels.round(), atol=1e-5, rtol=0).all():
            raise ValueError('Integer solution labels must be finite and integral within tolerance')
        # Cached solver labels may be 0.999999; .long() would truncate to 0.
        features[:, -1] = labels.round()
        if (features[:, -1] < 0).any():
            raise ValueError('FMIP requires nonnegative integer solution labels')
        # A solution need not use the largest allowed class (even for binaries).
        bounded = features[:, 4].bool() & torch.isfinite(features[:, 1])
        largest = features[:, -1].round().max()
        if bounded.any():
            largest = torch.maximum(largest, features[bounded, 1].floor().max())
        self.max_integer_num = int(largest.item()) + 1
    
    def download(self):
        pass

    def process(self):
        # load normalize statistics
        data_list = []
        problem_path_list = (sorted(name for name in os.listdir(self.problem_root)
                                    if get_file_extension(name) in {'.lp', '.mps', '.mps.gz', '.proto.lp'})
                             if self.file_name_list is None else self.file_name_list)
        print(f'Processing {len(problem_path_list)} files')
        for problem in tqdm(problem_path_list, desc='Adding data to dataset'):
            try:
                if self.solution_root is not None:
                    extension = get_file_extension(problem)
                    solution_path = os.path.join(self.solution_root, problem[:-len(extension)] + '_sol.pkl')
                    if not os.path.exists(solution_path):
                        print(f'No solution file for {problem}')
                        raise ValueError(f'No solution file for {problem}')
                    data = MIPmodel(os.path.join(self.problem_root, problem)).generGraphCom(solution_path)
                else:
                    data = MIPmodel(os.path.join(self.problem_root, problem)).generGraphCom()
                if data['x_Ivars'].shape[0] == 0:
                    raise ValueError('Every FMIP dataset graph must contain at least one integer variable')
                assert (data['x_Ivars'][:, -1] >= 0).all()
            except Exception as e:
                raise ValueError(f'Cannot encode {problem}: {e}') from e
            data_hetero = toHeteroData(data['x_cons'], data['x_Cvars'], data['x_Ivars'], 
                                       data['C2cons_edge_attr'], data['I2cons_edge_attr'], 
                                       data['C2cons_edge_index'], data['I2cons_edge_index'],
                                       model_sense=data.get('model_sense', 1),
                                       objective_constant=data.get('objective_constant', 0.))
            # print(data['C2cons_edge_index'])
            if data['C2cons_edge_index'].shape[1] > 0:
                assert data['C2cons_edge_index'][0].max() < data['x_Cvars'].shape[0]
            data_hetero['name'] = problem
            data_list.append(data_hetero)
        if len(data_list) == 0:
            raise ValueError('No data in the dataset')

        # Concatenate the list of `Data` objects into a single `Data` object
        data, slices = self.collate(data_list)
        _atomic_write(self.processed_paths[0], lambda stream: torch.save((data, slices), stream))
    
    def clear_processed_files(self):
        """remove processed files"""
        with _cache_lock(self.processed_dir + '.lock'):
            self._clear_processed_files()

    def _clear_processed_files(self):
        if not os.path.exists(self.processed_dir):
            print('No processed files to delete')
            return
        for file_name in os.listdir(self.processed_dir):
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        print("Processed files have been deleted.")

class SplitSchedule:
    def __init__(self, root, saved_path, split_ratio = [0.9, 0.05, 0.05], split_type = 'random'):
        solution_root = os.path.join(root, 'solution')
        problem_root = os.path.join(root, 'problem')
        supported = {'.lp', '.mps', '.mps.gz', '.proto.lp'}
        candidates = {}
        for filename in sorted(os.listdir(problem_root)):
            extension = get_file_extension(filename)
            if extension in supported:
                solution_name = filename[:-len(extension)] + '_sol.pkl'
                if solution_name in candidates:
                    raise ValueError(f'Ambiguous problem files for {solution_name}')
                candidates[solution_name] = filename
        self.dataset = []
        for solution_name in sorted(os.listdir(solution_root)):
            if not solution_name.endswith('_sol.pkl'):
                continue
            if solution_name not in candidates:
                raise ValueError(f'No corresponding problem file for {solution_name}')
            self.dataset.append(candidates[solution_name])
        if len(split_ratio) != 3 or any(ratio < 0 for ratio in split_ratio) or not math.isclose(sum(split_ratio), 1.):
            raise ValueError('split_ratio must contain three nonnegative fractions summing to one')
        self.split_ratio = list(split_ratio)
        self.split_type = split_type
        self.num_data = len(self.dataset)
        self.saved_path = saved_path
        os.makedirs(saved_path, exist_ok=True)
        self.split_file_name = 'split.pkl'
        self.metadata_path = os.path.join(saved_path, 'split.meta.json')
        self.lock_path = os.path.join(saved_path, '.split.lock')
        self.metadata = {'version': 1, 'root': os.path.realpath(root),
                         'split_ratio': self.split_ratio, 'split_type': split_type,
                         'problems': sorted(self.dataset)}
        with _cache_lock(self.lock_path):
            if not os.path.exists(os.path.join(self.saved_path, self.split_file_name)):
                self.split_data()
                self._save_split(self._generated_split())
            else:
                self._load_split()
                print('Loading split from saved file')
                if not os.path.exists(self.metadata_path):
                    # Preserve a validated legacy partition when adding provenance.
                    self._save_metadata()
        
    def split_data(self):
        if self.split_type == 'random':
            indices = torch.randperm(self.num_data)
        else:
            raise ValueError('Invalid split type')
        active = [i for i, ratio in enumerate(self.split_ratio) if ratio > 0]
        if self.num_data < len(active):
            raise ValueError(f'At least {len(active)} labeled problems are required for nonempty requested splits')
        counts = [math.floor(self.num_data * ratio) for ratio in self.split_ratio]
        remainder = self.num_data - sum(counts)
        order = sorted(active, key=lambda i: self.num_data * self.split_ratio[i] - counts[i], reverse=True)
        for i in order[:remainder]:
            counts[i] += 1
        for i in active:
            if counts[i] == 0:
                donor = max(active, key=lambda j: counts[j])
                counts[donor] -= 1
                counts[i] = 1
        self.train_indices = indices[:counts[0]]
        self.valid_indices = indices[counts[0]:counts[0] + counts[1]]
        self.test_indices = indices[counts[0] + counts[1]:]
        
    def save_split(self):
        with _cache_lock(self.lock_path):
            split_dict = self._generated_split() if hasattr(self, 'train_indices') else self._load_split()
            self._save_split(split_dict)

    def _save_metadata(self):
        payload = json.dumps(self.metadata, sort_keys=True).encode('utf-8')
        _atomic_write(self.metadata_path, lambda stream: stream.write(payload))

    def _save_split(self, split_dict):
        _atomic_write(os.path.join(self.saved_path, self.split_file_name),
                      lambda stream: pickle.dump(split_dict, stream, protocol=pickle.HIGHEST_PROTOCOL))
        self._save_metadata()
            
    def load_split(self):
        with _cache_lock(self.lock_path):
            return self._load_split()

    def _load_split(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, encoding='utf-8') as stream:
                metadata = json.load(stream)
            if metadata != self.metadata:
                raise ValueError('Saved split source, inventory, or configuration changed; '
                                 'use a new cache directory or regenerate split.pkl')
        with open(os.path.join(self.saved_path, self.split_file_name), 'rb') as f:
            split_dict = pickle.load(f)
        if not isinstance(split_dict, dict) or set(split_dict) != {'train', 'valid', 'test'}:
            raise ValueError('Saved split must contain train, valid, and test entries')
        if any(not isinstance(values, (list, tuple)) or
               any(not isinstance(name, str) for name in values)
               for values in split_dict.values()):
            raise ValueError('Saved split entries must be lists of problem filenames')
        names = [name for split in split_dict.values() for name in split]
        if len(names) != len(set(names)) or set(names) != set(self.dataset):
            raise ValueError('Saved split contains duplicates or missing problems; regenerate split.pkl')
        for name, ratio in zip(('train', 'valid', 'test'), self.split_ratio):
            if ratio > 0 and not split_dict[name]:
                raise ValueError(f'Saved {name} split is empty; regenerate split.pkl with enough labeled problems')
        return split_dict
        
    def get_split(self):
        return self.load_split()

    def _generated_split(self):
        return {'train': [self.dataset[i] for i in self.train_indices],
                'valid': [self.dataset[i] for i in self.valid_indices],
                'test': [self.dataset[i] for i in self.test_indices]}

if __name__ == '__main__':
    toy_file_root = '/data/GM4MILP/datasets/load_balancing'
    saved_path = 'dataset_cache/load_balancing'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    split = SplitSchedule(toy_file_root, saved_path, split_ratio=[0.0005, 0.2, 0.8])
    split_dict = split.get_split()
    print(split_dict)
    train_dataset = GraphMILPDataset(toy_file_root, saved_path, file_name_list = split_dict['train'], prefix = 'train', given_solution=True)
    valid_dataset = GraphMILPDataset(toy_file_root, saved_path, file_name_list = split_dict['valid'], prefix = 'valid', given_solution=True)
    test_dataset = GraphMILPDataset(toy_file_root, saved_path, file_name_list = split_dict['test'], prefix = 'test', given_solution=True)
