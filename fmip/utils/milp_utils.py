import torch
from torch_geometric.data import Batch
import copy
import torch.nn.functional as F
def calculate_objective():
        pass
    
def objective_value(coef, sol):
    return torch.dot(coef, sol)


def from_edge_to_matrix(edge_index, edge_attr, shape):
    #sparse matrix
    sparse_matrix = torch.sparse_coo_tensor(edge_index, edge_attr.squeeze(-1) , shape, device=edge_index.device)
    return sparse_matrix

def cat_Ivars_Cvars_edges(I2cons_edge_index, C2cons_edge_index, I2cons_edge_attr, C2cons_edge_attr, I2cons_shape, C2cons_shape):
    I2cons_sparse = from_edge_to_matrix(I2cons_edge_index, I2cons_edge_attr, I2cons_shape)
    C2cons_sparse = from_edge_to_matrix(C2cons_edge_index, C2cons_edge_attr, C2cons_shape)
    return torch.cat([I2cons_sparse, C2cons_sparse], dim=0).t()

def graph_to_adj_matrix(graph):
    I2cons_index, I2cons_attr = graph['Ivars', 'to', 'cons'].edge_index, graph['Ivars', 'to', 'cons'].edge_attr
    C2cons_index, C2cons_attr = graph['Cvars', 'to', 'cons'].edge_index, graph['Cvars', 'to', 'cons'].edge_attr
    C2cons_sparse = from_edge_to_matrix(C2cons_index, C2cons_attr, (graph['Cvars'].x.shape[0], graph['cons'].x.shape[0]))
    I2cons_sparse = from_edge_to_matrix(I2cons_index, I2cons_attr, (graph['Ivars'].x.shape[0], graph['cons'].x.shape[0]))
    return torch.cat([I2cons_sparse, C2cons_sparse], dim=0).t()

def cal_objective(graph):
    coef = objective_value(graph['Cvars'].x[:, 2], graph['Cvars'].x[:, -1]) + objective_value(graph['Ivars'].x[:, 2], graph['Ivars'].x[:, -1])
    return coef

def cal_feasibility(graph):
    A = graph_to_adj_matrix(graph)
    b = graph['cons'].x[:, -1]
    sol = torch.cat([graph['Ivars'].x[:, -1], graph['Cvars'].x[:, -1]])
    return feasibility(A, b, sol)


def cal_feasibility_batch(graph, Ivars, Cvars, reduction = True, if_Ivars_batch = False, if_Cvars_batch = False):
    I2cons_index, I2cons_attr = graph['Ivars', 'to', 'cons'].edge_index, graph['Ivars', 'to', 'cons'].edge_attr
    C2cons_index, C2cons_attr = graph['Cvars', 'to', 'cons'].edge_index, graph['Cvars', 'to', 'cons'].edge_attr
    A_I = from_edge_to_matrix(I2cons_index, I2cons_attr, (graph['Ivars'].x.shape[0], graph['cons'].x.shape[0])).t()
    A_C = from_edge_to_matrix(C2cons_index, C2cons_attr, (graph['Cvars'].x.shape[0], graph['cons'].x.shape[0])).t()
    Ax = torch.mm(A_I, Ivars) + torch.mm(A_C, Cvars)
    AIx = torch.mm(A_I, Ivars) if if_Ivars_batch else torch.mv(A_I, Ivars)
    ACx = torch.mm(A_C, Cvars) if if_Cvars_batch else torch.mv(A_C, Cvars)
    
    b = graph['cons'].x[:, -1]
    if reduction:
        return torch.norm(torch.relu(Ax - b))
    else:
        batch_index = graph['cons'].batch
        batch_size = len(graph['name'])
        return (torch.scatter_add(torch.zeros(batch_size, device=graph['cons'].x.device), dim=0, index=batch_index, src=torch.relu(Ax - b)**2))**0.5
    
def cal_objective_batch(graph, Ivars, Cvars, reduction = True):
    if not graph.only_discrete:
        obj_C = torch.sum((Cvars * graph['Cvars'].x[:, 2]), dim=-1)
    obj_I = torch.sum((Ivars * graph['Ivars'].x[:, 2]), dim=-1)
    if reduction:
        return torch.sum(obj_C) + torch.sum(obj_I) if not graph.only_discrete else torch.sum(obj_I)
    else:
        return obj_C + obj_I if not graph.only_discrete else obj_I

def cal_feasibility_graph(graph, reduction = True):
    flag = graph.only_discrete if type(graph.only_discrete) == bool else graph.only_discrete[0]
    if not flag:
        I2cons_index, I2cons_attr = graph['Ivars', 'to', 'cons'].edge_index, graph['Ivars', 'to', 'cons'].edge_attr
        C2cons_index, C2cons_attr = graph['Cvars', 'to', 'cons'].edge_index, graph['Cvars', 'to', 'cons'].edge_attr
        A_I = from_edge_to_matrix(I2cons_index, I2cons_attr, (graph['Ivars'].x.shape[0], graph['cons'].x.shape[0])).t()
        A_C = from_edge_to_matrix(C2cons_index, C2cons_attr, (graph['Cvars'].x.shape[0], graph['cons'].x.shape[0])).t()
        Ax = torch.mv(A_I, graph['Ivars'].x[:, -1]) + torch.mv(A_C, graph['Cvars'].x[:, -1])
    else:
        I2cons_index, I2cons_attr = graph['Ivars', 'to', 'cons'].edge_index, graph['Ivars', 'to', 'cons'].edge_attr
        A_I = from_edge_to_matrix(I2cons_index, I2cons_attr, (graph['Ivars'].x.shape[0], graph['cons'].x.shape[0])).t()
        Ax = torch.mv(A_I, graph['Ivars'].x[:, -1])
    b = graph['cons'].x[:, -1]
    if reduction:
        return torch.norm(torch.relu(Ax - b))
    else:
        batch_index = graph['cons'].batch
        batch_size = len(graph['name'])
        return (torch.scatter_add(torch.zeros(batch_size, device=graph['cons'].x.device), dim=0, index=batch_index, src=torch.relu(Ax - b)**2))**0.5

def cal_objective_graph(graph, reduction = True):
    obj_C = graph['Cvars'].x[:, 2] * graph['Cvars'].x[:, -1]
    obj_I = graph['Ivars'].x[:, 2] * graph['Ivars'].x[:, -1]
    if reduction:
        return torch.sum(obj_C) + torch.sum(obj_I)
    else:
        batch_index_I = graph['Ivars'].batch
        batch_index_C = graph['Cvars'].batch
        batch_size = len(graph['name'])
        return torch.scatter_add(torch.zeros(batch_size, device=graph['cons'].x.device), dim=0, index=batch_index_C, src=obj_C) + \
                torch.scatter_add(torch.zeros(batch_size, device=graph['cons'].x.device), dim=0, index=batch_index_I, src=obj_I)

# def guidance(graph, weight):
#     '''
#     The guidance function for the MILP graph
#     target_function = objective + weight * feasibility
#     return guidance value (gradient of the target function)
#     '''
def target_function_batch(graph, Ivars, Cvars, weight = 10, reduction = True):
    '''
    The guidance function for the MILP graph
    target_function = objective + weight * feasibility'''
    coef = cal_objective_batch(graph, Ivars, Cvars, reduction)
    feas = cal_feasibility_batch(graph, Ivars, Cvars, reduction)
    return coef + weight * feas
    
def target_function(graph, weight = 10):
    '''
    The guidance function for the MILP graph
    target_function = objective + weight * feasibility'''
    coef = cal_objective(graph)
    feas = cal_feasibility(graph)
    return coef + weight * feas

def gradient_function(graph, weight = 10):
    graph['Cvars'].x.requires_grad = True
    graph['Ivars'].x.requires_grad = True
    target = target_function(graph, weight)
    target.backward()
    return graph['Ivars'].x.grad[:, -1], graph['Cvars'].x.grad[:, -1]

def target_function_graph(graph, weight = 10, reduction = True):
    return cal_objective_graph(graph, reduction) + weight * cal_feasibility_graph(graph, reduction)

def duplicate_hetero_data(heterodata, times):
    """Duplicate the edge index (in sparse graphs) for parallel sampling."""

    return Batch.from_data_list(
        [copy.deepcopy(heterodata) for _ in range(times)]
    )
    

class feasibility:
    def __init__(self, if_only_graph, if_reduction, if_Ivars_batch, if_Cvars_batch):
        self.if_only_graph = if_only_graph
        self.if_reduction = if_reduction
        self.if_Ivars_batch = if_Ivars_batch
        self.if_Cvars_batch = if_Cvars_batch
        assert not (if_only_graph and (if_Ivars_batch or if_Cvars_batch)), 'if_only_graph and if_Ivars_batch or if_Cvars_batch cannot be True at the same time'
    def __call__(self, graph, Ivars, Cvars):
        if self.if_only_graph:
            return cal_feasibility_graph(graph, self.if_reduction)
        else:
            I2cons_index, I2cons_attr = graph['Ivars', 'to', 'cons'].edge_index, graph['Ivars', 'to', 'cons'].edge_attr
            A_I = from_edge_to_matrix(I2cons_index, I2cons_attr, (graph['Ivars'].x.shape[0], graph['cons'].x.shape[0]))
            AIx = torch.mm(Ivars, A_I) if self.if_Ivars_batch else torch.mv(A_I.t(), Ivars)
            
            if (not graph.only_discrete) or (Cvars is not None):
                C2cons_index, C2cons_attr = graph['Cvars', 'to', 'cons'].edge_index, graph['Cvars', 'to', 'cons'].edge_attr
                A_C = from_edge_to_matrix(C2cons_index, C2cons_attr, (graph['Cvars'].x.shape[0], graph['cons'].x.shape[0]))
                ACx = torch.mm(Cvars, A_C) if self.if_Cvars_batch else torch.mv(A_C.t(), Cvars)
                Ax = AIx + ACx
            else:
                Ax = AIx
            b = graph['cons'].x[:, -1]
            if self.if_reduction:
                return torch.norm(torch.relu(Ax - b))
            else:
                return torch.norm(torch.relu(Ax - b), dim=1)
            
class objective:
    def __init__(self, if_only_graph, if_reduction):
        self.if_only_graph = if_only_graph
        self.if_reduction = if_reduction
    def __call__(self, graph, Ivars, Cvars):
        if self.if_only_graph:
            return cal_objective_graph(graph, self.if_reduction)
        else:
            return cal_objective_batch(graph, Ivars, Cvars, self.if_reduction)
        
class target_func:
    def __init__(self, weight_c, weight_f, if_reduction, if_only_graph, if_Ivars_batch, if_Cvars_batch):
        self.weight_c, self.weight_f = weight_c, weight_f
        self.if_reduction = if_reduction
        self.if_only_graph = if_only_graph
        self.if_Ivars_batch = if_Ivars_batch
        self.if_Cvars_batch = if_Cvars_batch
        self.feasibility_cal = feasibility(if_only_graph=if_only_graph, if_reduction=if_reduction, if_Ivars_batch=if_Ivars_batch, if_Cvars_batch=if_Cvars_batch)
        self.objective_cal = objective(if_only_graph=if_only_graph, if_reduction=if_reduction)
    def __call__(self, graph, Ivars, Cvars):
        if self.weight_c == 0:
            return self.weight_f * self.feasibility_cal(graph, Ivars, Cvars)
        return self.weight_c * self.objective_cal(graph, Ivars, Cvars) + self.weight_f * self.feasibility_cal(graph, Ivars, Cvars)
    
def rescale_grad(
    grad: torch.Tensor, clip_scale=1.0
): 

    scale = (grad ** 2).mean(dim=-1)
    scale: torch.Tensor = scale.sum(dim=-1)
    clipped_scale = torch.clamp(scale, max=clip_scale)
    co_ef = clipped_scale / (scale + 1e-9)  # [B]
    grad = grad * co_ef.view(-1, 1)

    return grad        
        
        
if __name__ == '__main__':
    import os
    file_root = '/data/GM4MILP/datasets/load_balancing_tiny/problem/'
    sol_root = '/data/GM4MILP/datasets/load_balancing_tiny/solution/'
    sol_path = os.path.join(sol_root, os.listdir(sol_root)[0])
    file_path = os.path.join(file_root, os.listdir(sol_root)[0].replace('_sol.pkl', '.mps.gz'))
    from utils.milp_reader import MIPmodel
    from dataset import toHeteroData
    graph_com = MIPmodel(file_path).generGraphCom(sol_path)
    graph = toHeteroData(graph_com['x_cons'], graph_com['x_Cvars'], graph_com['x_Ivars'], graph_com['C2cons_edge_attr'], graph_com['I2cons_edge_attr'], graph_com['C2cons_edge_index'], graph_com['I2cons_edge_index'])
    duplicate_Ivars = torch.stack([graph['Ivars'].x[:, -1] for _ in range(10)])
    duplicate_Cvars = torch.stack([graph['Cvars'].x[:, -1] for _ in range(10)])
    feasibility_cal = feasibility(if_only_graph=False, if_reduction=False, if_Ivars_batch=True, if_Cvars_batch=False)
    objective_cal = objective(if_only_graph=False, if_reduction=False)
    print(feasibility_cal(graph, duplicate_Ivars, graph['Cvars'].x[:, -1]))
    print(objective_cal(graph, duplicate_Ivars, graph['Cvars'].x[:, -1]))
    