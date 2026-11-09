import torch
from torch_geometric.data import Batch
import copy
import operator
import torch.nn.functional as F
def calculate_objective():
    raise NotImplementedError('Use cal_objective_graph(graph) or cal_objective_batch(graph, Ivars, Cvars)')


def _only_discrete(graph):
    flag = getattr(graph, 'only_discrete', False)
    if isinstance(flag, (list, tuple)):
        if any(bool(value) != bool(flag[0]) for value in flag):
            raise ValueError('Cannot mix discrete-only and mixed graph modes in one batch')
        return bool(flag[0]) if flag else False
    if isinstance(flag, torch.Tensor):
        if flag.numel() == 0:
            return False
        if not (flag == flag.reshape(-1)[0]).all():
            raise ValueError('Cannot mix discrete-only and mixed graph modes in one batch')
        return bool(flag.reshape(-1)[0])
    return bool(flag)


def _objective_constants(graph, reference):
    return torch.as_tensor(getattr(graph, 'objective_constant', 0.),
                           device=reference.device, dtype=reference.dtype).reshape(-1)


def _graph_count(graph):
    names = graph.get('name', None)
    if isinstance(names, (list, tuple)):
        return len(names)
    return getattr(graph, 'num_graphs', 1)


def _node_batch(graph, kind):
    return getattr(graph[kind], 'batch', torch.zeros(graph[kind].x.shape[0],
                                                   device=graph[kind].x.device, dtype=torch.long))
    
def objective_value(coef, sol):
    return torch.dot(coef, sol)


def constraint_matmul(matrix, values):
    """Keep solver coefficient arithmetic out of unsupported sparse AMP kernels."""
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        if matrix.dtype in (torch.float16, torch.bfloat16):
            matrix = matrix.float()
        values = values.to(dtype=matrix.dtype)
        return torch.mv(matrix, values) if values.ndim == 1 else torch.mm(matrix, values)


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
    return cal_objective_graph(graph)

def cal_feasibility(graph):
    A = graph_to_adj_matrix(graph)
    b = graph['cons'].x[:, -1]
    sol = torch.cat([graph['Ivars'].x[:, -1], graph['Cvars'].x[:, -1]])
    return torch.linalg.vector_norm(torch.relu(constraint_matmul(A, sol) - b))


def cal_feasibility_batch(graph, Ivars, Cvars, reduction = True, if_Ivars_batch = False, if_Cvars_batch = False):
    """Constraint violations for vectors or sample-by-variable solution matrices."""
    I2cons_index, I2cons_attr = graph['Ivars', 'to', 'cons'].edge_index, graph['Ivars', 'to', 'cons'].edge_attr
    A_I = from_edge_to_matrix(I2cons_index, I2cons_attr, (graph['Ivars'].x.shape[0], graph['cons'].x.shape[0])).t()
    Ax = constraint_matmul(A_I, Ivars.t()).t() if if_Ivars_batch or Ivars.ndim == 2 else constraint_matmul(A_I, Ivars)
    if not _only_discrete(graph):
        C2cons_index, C2cons_attr = graph['Cvars', 'to', 'cons'].edge_index, graph['Cvars', 'to', 'cons'].edge_attr
        A_C = from_edge_to_matrix(C2cons_index, C2cons_attr, (graph['Cvars'].x.shape[0], graph['cons'].x.shape[0])).t()
        ACx = constraint_matmul(A_C, Cvars.t()).t() if if_Cvars_batch or Cvars.ndim == 2 else constraint_matmul(A_C, Cvars)
        Ax = Ax + ACx
    
    b = graph['cons'].x[:, -1]
    if reduction:
        return torch.norm(torch.relu(Ax - b))
    else:
        batch_index = _node_batch(graph, 'cons')
        batch_size = _graph_count(graph)
        violations = torch.relu(Ax - b).square()
        per_graph = violations.new_zeros((*violations.shape[:-1], batch_size))
        return per_graph.scatter_add(-1, batch_index.expand_as(violations), violations).sqrt()
    
def cal_objective_batch(graph, Ivars, Cvars, reduction = True, *, include_constant=True):
    only_discrete = _only_discrete(graph)
    if not only_discrete:
        obj_C = torch.sum((Cvars * graph['Cvars'].x[:, 2]), dim=-1)
    obj_I = torch.sum((Ivars * graph['Ivars'].x[:, 2]), dim=-1)
    values = obj_I if only_discrete else obj_I + obj_C
    if include_constant:
        values = values + _objective_constants(graph, values).sum()
    return values.sum() if reduction else values

def cal_feasibility_graph(graph, reduction = True):
    flag = _only_discrete(graph)
    if not flag:
        I2cons_index, I2cons_attr = graph['Ivars', 'to', 'cons'].edge_index, graph['Ivars', 'to', 'cons'].edge_attr
        C2cons_index, C2cons_attr = graph['Cvars', 'to', 'cons'].edge_index, graph['Cvars', 'to', 'cons'].edge_attr
        A_I = from_edge_to_matrix(I2cons_index, I2cons_attr, (graph['Ivars'].x.shape[0], graph['cons'].x.shape[0])).t()
        A_C = from_edge_to_matrix(C2cons_index, C2cons_attr, (graph['Cvars'].x.shape[0], graph['cons'].x.shape[0])).t()
        Ax = constraint_matmul(A_I, graph['Ivars'].x[:, -1]) + constraint_matmul(A_C, graph['Cvars'].x[:, -1])
    else:
        I2cons_index, I2cons_attr = graph['Ivars', 'to', 'cons'].edge_index, graph['Ivars', 'to', 'cons'].edge_attr
        A_I = from_edge_to_matrix(I2cons_index, I2cons_attr, (graph['Ivars'].x.shape[0], graph['cons'].x.shape[0])).t()
        Ax = constraint_matmul(A_I, graph['Ivars'].x[:, -1])
    b = graph['cons'].x[:, -1]
    if reduction:
        return torch.norm(torch.relu(Ax - b))
    else:
        batch_index = _node_batch(graph, 'cons')
        batch_size = _graph_count(graph)
        return (torch.scatter_add(Ax.new_zeros(batch_size), dim=0, index=batch_index, src=torch.relu(Ax - b)**2))**0.5

def cal_objective_graph(graph, reduction = True, *, include_constant=True):
    obj_I = graph['Ivars'].x[:, 2] * graph['Ivars'].x[:, -1]
    flag = _only_discrete(graph)
    constants = _objective_constants(graph, obj_I) if include_constant else obj_I.new_zeros(1)
    obj_C = None if flag else graph['Cvars'].x[:, 2] * graph['Cvars'].x[:, -1]
    if reduction:
        return obj_I.sum() + (0 if flag else obj_C.sum()) + constants.sum()
    result = obj_I.new_zeros(_graph_count(graph)).scatter_add_(0, _node_batch(graph, 'Ivars'), obj_I)
    if not flag:
        result = result.scatter_add(0, _node_batch(graph, 'Cvars'), obj_C)
    return result + constants

# def guidance(graph, weight):
#     '''
#     The guidance function for the MILP graph
#     target_function = objective + weight * feasibility
#     return guidance value (gradient of the target function)
#     '''
def target_function_batch(graph, Ivars, Cvars, weight = 10, reduction = True):
    """Return total or per-graph objectives plus constraint penalties.

    Without reduction, the final axis indexes graphs and any leading axis
    indexes samples, matching ``cal_feasibility_batch``.
    """
    if reduction:
        coef = cal_objective_batch(graph, Ivars, Cvars, reduction=True)
    else:
        def per_graph(values, node_type):
            contributions = values * graph[node_type].x[:, 2]
            result = contributions.new_zeros((*contributions.shape[:-1], _graph_count(graph)))
            return result.scatter_add(-1, _node_batch(graph, node_type).expand_as(contributions),
                                      contributions)

        coef = per_graph(Ivars, 'Ivars')
        if not _only_discrete(graph):
            coef = coef + per_graph(Cvars, 'Cvars')
        coef = coef + _objective_constants(graph, coef)
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
    """Differentiate variable values without mutating features or their gradients."""
    with torch.enable_grad(), torch.autocast(device_type=graph['Ivars'].x.device.type, enabled=False):
        integers = graph['Ivars'].x[:, -1].detach().clone().requires_grad_(True)
        continuous = graph['Cvars'].x[:, -1].detach().clone().requires_grad_(True)
        target = target_function_batch(graph, integers, continuous, weight=weight)
        gradients = torch.autograd.grad(target, (integers, continuous), allow_unused=True)
    return tuple(torch.zeros_like(values) if gradient is None else gradient
                 for values, gradient in zip((integers, continuous), gradients))

def target_function_graph(graph, weight = 10, reduction = True):
    return cal_objective_graph(graph, reduction) + weight * cal_feasibility_graph(graph, reduction)

def duplicate_hetero_data(heterodata, times):
    """Repeat complete samples while preserving each graph's partition.

    A Batch must first be unbatched; batching Batch objects nests graph-level
    metadata and changes the graph IDs used by normalization and evaluation.
    """
    try:
        repetitions = operator.index(times)
    except TypeError as error:
        raise ValueError('times must be a positive integer') from error
    if repetitions < 1:
        raise ValueError('times must be a positive integer')
    if isinstance(heterodata, Batch):
        source = heterodata.clone()
        mode = getattr(source, 'only_discrete', False)
        # Inference may replace a collated per-graph flag with a scalar. PyG's
        # stored slice metadata still expects one entry per original graph.
        if isinstance(mode, bool):
            source.only_discrete = [mode] * source.num_graphs
        elif isinstance(mode, torch.Tensor) and mode.numel() == 1:
            source.only_discrete = mode.reshape(1).expand(source.num_graphs).clone()
        graphs = source.to_data_list()
    else:
        graphs = [heterodata]
    return Batch.from_data_list([graph.clone() for _ in range(repetitions) for graph in graphs])
    

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
            AIx = constraint_matmul(A_I.t(), Ivars.t()).t() if self.if_Ivars_batch else constraint_matmul(A_I.t(), Ivars)
            
            if not _only_discrete(graph):
                C2cons_index, C2cons_attr = graph['Cvars', 'to', 'cons'].edge_index, graph['Cvars', 'to', 'cons'].edge_attr
                A_C = from_edge_to_matrix(C2cons_index, C2cons_attr, (graph['Cvars'].x.shape[0], graph['cons'].x.shape[0]))
                ACx = constraint_matmul(A_C.t(), Cvars.t()).t() if self.if_Cvars_batch else constraint_matmul(A_C.t(), Cvars)
                Ax = AIx + ACx
            else:
                Ax = AIx
            b = graph['cons'].x[:, -1]
            if self.if_reduction:
                return torch.norm(torch.relu(Ax - b))
            else:
                return torch.norm(torch.relu(Ax - b), dim=-1)
            
class objective:
    def __init__(self, if_only_graph, if_reduction):
        self.if_only_graph = if_only_graph
        self.if_reduction = if_reduction
    def __call__(self, graph, Ivars, Cvars, *, include_constant=True):
        if self.if_only_graph:
            return cal_objective_graph(graph, self.if_reduction, include_constant=include_constant)
        else:
            return cal_objective_batch(graph, Ivars, Cvars, self.if_reduction, include_constant=include_constant)
        
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

    scale = (grad ** 2).mean(dim=-1, keepdim=True)
    clipped_scale = torch.clamp(scale, max=clip_scale)
    co_ef = clipped_scale / (scale + 1e-9)  # [B]
    grad = grad * co_ef

    return grad        
        
        
if __name__ == '__main__':
    import os
    file_root = '/data/GM4MILP/datasets/load_balancing_tiny/problem/'
    sol_root = '/data/GM4MILP/datasets/load_balancing_tiny/solution/'
    sol_path = os.path.join(sol_root, os.listdir(sol_root)[0])
    file_path = os.path.join(file_root, os.listdir(sol_root)[0].replace('_sol.pkl', '.mps.gz'))
    from fmip.utils.milp_reader import MIPmodel
    from fmip.dataset import toHeteroData
    graph_com = MIPmodel(file_path).generGraphCom(sol_path)
    graph = toHeteroData(graph_com['x_cons'], graph_com['x_Cvars'], graph_com['x_Ivars'], graph_com['C2cons_edge_attr'], graph_com['I2cons_edge_attr'], graph_com['C2cons_edge_index'], graph_com['I2cons_edge_index'])
    duplicate_Ivars = torch.stack([graph['Ivars'].x[:, -1] for _ in range(10)])
    duplicate_Cvars = torch.stack([graph['Cvars'].x[:, -1] for _ in range(10)])
    feasibility_cal = feasibility(if_only_graph=False, if_reduction=False, if_Ivars_batch=True, if_Cvars_batch=False)
    objective_cal = objective(if_only_graph=False, if_reduction=False)
    print(feasibility_cal(graph, duplicate_Ivars, graph['Cvars'].x[:, -1]))
    print(objective_cal(graph, duplicate_Ivars, graph['Cvars'].x[:, -1]))
