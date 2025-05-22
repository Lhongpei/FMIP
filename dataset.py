from torch_geometric.data import InMemoryDataset
import os
import torch
import pandas as pd
from tqdm import tqdm
import pickle
from utils.milp_reader import MIPmodel
from torch_geometric.data import HeteroData, Batch
from utils.utils import *
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
        self['Ivars'].x[:, -1] = torch.clip(self['Ivars'].x[:, -1], 0, 1) #Assume the integer variable is binary, need to be modified for general case
        self['Cvars'].x[:, -1] = torch.clip(self['Cvars'].x[:, -1], self['Cvars'].x[:, 0], self['Cvars'].x[:, 1])
        self['Cvars'].x[:, -1] = torch.clip(self['Ivars'].x[:, -1], self['Ivars'].x[:, 0], self['Ivars'].x[:, 1])
        
    def projection(self):
        proj_Ivars = torch.clip(self['Ivars'].x[:, -1], 0, 1) #Assume the integer variable is binary, need to be modified for general case
        proj_Cvars = torch.clip(self['Cvars'].x[:, -1], self['Cvars'].x[:, 0], self['Cvars'].x[:, 1])
        proj_Ivars = torch.clip(self['Ivars'].x[:, -1], self['Ivars'].x[:, 0], self['Ivars'].x[:, 1])
        return proj_Ivars, proj_Cvars
        
    def projection_Cvars(self, Cvars):
        proj_Cvars = torch.clip(Cvars, self['Cvars'].x[:, 0], self['Cvars'].x[:, 1])
        return proj_Cvars
    
    def projection_Ivars(self, Ivars):
        mask = torch.repeat_interleave(torch.arange(Ivars.shape[-1]).unsqueeze(0), self['Ivars'].x.shape[0], dim=0).to(Ivars.device)
        mask_bool = (mask < self['Ivars'].x[:, 0].unsqueeze(-1)) | (mask > self['Ivars'].x[:, 1].unsqueeze(-1))
        proj_Ivars = torch.where(mask_bool, 0, Ivars)
        return proj_Ivars
    
def toHeteroData(x_cons, x_Cvars, x_Ivars, 
                 C2cons_edge_attr, I2cons_edge_attr, 
                 C2cons_edge_index, I2cons_edge_index):
    data = GraphMILP()
    data['cons'].x = x_cons
    data['Cvars'].x = x_Cvars
    data['Ivars'].x = x_Ivars
    data['Cvars', 'to', 'cons'].edge_index = C2cons_edge_index
    data['Ivars', 'to', 'cons'].edge_index = I2cons_edge_index
    data['Cvars', 'to', 'cons'].edge_attr = C2cons_edge_attr
    data['Ivars', 'to', 'cons'].edge_attr = I2cons_edge_attr
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
        self.file_name_list = file_name_list
        self.prefix = prefix
        if reprocess:
            self.clear_processed_files()
            print('Reprocessing the dataset')
        print(f'Initializing dataset from {root}')
        super(GraphMILPDataset, self).__init__(root)
            
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.calculate_max_integer_()

    @property
    def processed_dir(self):
        return os.path.join(self.saved_path, self.prefix + '_processed')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def calculate_max_integer_(self):
        self.max_integer_num = int(torch.max(self._data['Ivars'].x[:, -1]).item() + 1)
    
    def download(self):
        pass

    def process(self):
        # load normalize statistics
        data_list = []
        problem_path_list = os.listdir(self.problem_root) if self.file_name_list is None else self.file_name_list
        print(f'Processing {len(problem_path_list)} files')
        for problem in tqdm(problem_path_list, desc='Adding data to dataset'):
            try:
                if self.solution_root is not None:
                    solution_path = find_respect_solution(self.solution_root, problem)
                    if not os.path.exists(solution_path):
                        print(f'No solution file for {problem}')
                        raise ValueError(f'No solution file for {problem}')
                    data = MIPmodel(os.path.join(self.problem_root, problem)).generGraphCom(solution_path)
                else:
                    data = MIPmodel(os.path.join(self.problem_root, problem)).generGraphCom()
                assert (data['x_Ivars'][:, -1] >= 0).all()
            except Exception as e:
                print(f'Error in processing {problem}, Error: {e}')
                continue
            data_hetero = toHeteroData(data['x_cons'], data['x_Cvars'], data['x_Ivars'], 
                                       data['C2cons_edge_attr'], data['I2cons_edge_attr'], 
                                       data['C2cons_edge_index'], data['I2cons_edge_index'])
            # print(data['C2cons_edge_index'])
            if data['C2cons_edge_index'].shape[1] > 0:
                assert data['C2cons_edge_index'][0].max() <= data['x_Cvars'].shape[0]
            data_hetero['name'] = problem
            data_list.append(data_hetero)
        if len(data_list) == 0:
            raise ValueError('No data in the dataset')

        # Concatenate the list of `Data` objects into a single `Data` object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def clear_processed_files(self):
        """remove processed files"""
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
        sol_dataset = os.listdir(os.path.join(root, 'solution'))
        problem_root = os.path.join(root, 'problem')
        first_problem = os.path.join(problem_root, os.listdir(problem_root)[0])
        type_file = get_file_extension(first_problem)
        assert type_file in ['.lp', '.mps', '.mps.gz', '.proto.lp'], f'Invalid file type {type_file}, only lp, mps, mps.gz are supported'
        self.dataset = [f.replace('_sol.pkl', type_file) for f in sol_dataset]
        self.split_ratio = split_ratio
        self.split_type = split_type
        self.num_data = len(self.dataset)
        self.saved_path = saved_path
        self.split_file_name = f'split.pkl'    
        if not os.path.exists(os.path.join(self.saved_path, self.split_file_name)):
            self.split_data()
            self.save_split()
        else:
            print('Loading split from saved file')
        
    def split_data(self):
        if self.split_type == 'random':
            indices = torch.randperm(self.num_data)
        else:
            raise ValueError('Invalid split type')
        self.train_indices = indices[:int(self.num_data * self.split_ratio[0])]
        self.valid_indices = indices[int(self.num_data * self.split_ratio[0]):int(self.num_data * (self.split_ratio[0] + self.split_ratio[1]))]
        self.test_indices = indices[int(self.num_data * (self.split_ratio[0] + self.split_ratio[1])):]
        
    def save_split(self):
        dict_sp = self.get_split()
        split_dict = {
            'train': dict_sp['train'],
            'valid': dict_sp['valid'],
            'test': dict_sp['test']
        }
        with open(os.path.join(self.saved_path, self.split_file_name), 'wb') as f:
            pickle.dump(split_dict, f)
            
    def load_split(self):
        with open(os.path.join(self.saved_path, self.split_file_name), 'rb') as f:
            split_dict = pickle.load(f)
        return split_dict
        
    def get_split(self):
        if not os.path.exists(os.path.join(self.saved_path, self.split_file_name)):
            train_data = [self.dataset[i] for i in self.train_indices]
            valid_data = [self.dataset[i] for i in self.valid_indices]
            test_data = [self.dataset[i] for i in self.test_indices]
            split = {
                'train': train_data,
                'valid': valid_data,
                'test': test_data
            }
        else:
            split = self.load_split()
        return split    

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