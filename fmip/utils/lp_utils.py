import torch 
import torch.nn as nn
import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB
from utils.milp_reader import MIPmodel
from dataset import GraphMILP
from torch_geometric.data import HeteroData
import torch.nn.functional as F
'''
This file implemented basic function to calculate LP feasibility and objective value
'''
def dual_objective_function(predictions:torch.Tensor, std_matrix, right_high_side:torch.Tensor, coef:torch.Tensor):

    Aty = calculate_Aty(predictions, std_matrix)
    dual_objective = right_high_side @ predictions - torch.sum(F.relu(- coef + Aty))
    return dual_objective

def dual_objective_gradient(predictions:torch.Tensor, std_matrix, right_high_side:torch.Tensor, coef:torch.Tensor):
    
     # Compute A^T y
    std_matrix = std_matrix.t()
    Aty = torch.matmul(std_matrix, predictions)

    # Compute the mask for ReLU
    mask = (-coef + Aty > 0).float()  # Mask: 1 if ReLU is active, else 0

    # Compute the gradient
    gradient = right_high_side - torch.matmul(std_matrix.T, mask)

    return gradient

def reorder_split(predictions, var_ptr, cons_ptr, return_numpy=True):
    predict_var = predictions[:var_ptr[-1]]
    predict_cons = predictions[var_ptr[-1]:]
    
    point_indicator_var = (var_ptr[1:] - var_ptr[:-1]).cpu().tolist()
    point_indicator_cons = (cons_ptr[1:] - cons_ptr[:-1]).cpu().tolist()
    
    split_var = torch.split(predict_var, point_indicator_var)
    split_cons = torch.split(predict_cons, point_indicator_cons)
    
    return [torch.cat([split_var[i], split_cons[i]]) for i in range(len(split_var))] \
        if not return_numpy else [torch.cat([split_var[i], split_cons[i]]).cpu().numpy() for i in range(len(split_var))]
    
def fix_non_basis(m, predictions, coef, lb, ub, confidence = 0.5):
    '''
    Fix the non-basis variable to the bound
    Args:
        m: gurobi model
        predictions: numpy array of the predictions
        confidence: the confidence threshold to fix the variable
    '''
    Vars = m.getVars()
    for i in range(m.NumVars):
        if predictions[i] < confidence:
            fixed_value = lb[i] if coef[i] >= 0 and lb[i] != -np.inf else ub[i]
            Vars[i].lb = fixed_value  # 下界
            Vars[i].ub = fixed_value  # 上界
    return m
    # for i in range(m.NumConstrs):
    #     if predictions[i + m.NumVars] > confidence:
    #         m.getConstr(i).setAttr('Lazy', 1)
        
def transform_to_equalities(model):
    """
    将 Gurobi 模型中的所有约束转换为 Ax + s = b 形式。
    """
    # 创建新模型
    new_model = Model("EqualityModel")
    
    # 复制变量并建立映射
    var_map = {}  # 用于映射原模型变量到新模型变量
    for var in model.getVars():
        new_var = new_model.addVar(lb=var.lb, ub=var.ub, obj=0,  # 暂时不设置目标函数
                                   vtype=var.vtype, name=var.varName)
        var_map[var] = new_var
    new_model.update()
    
    # 获取约束并转换
    for constr in model.getConstrs():
        sense = constr.Sense  # 约束类型 (<=, ==, >=)
        rhs = constr.getAttr("RHS")  # 右侧值
        row = model.getRow(constr)  # 获取约束的系数
        
        # 添加松弛变量
        slack_var = new_model.addVar(lb=0 if sense == '<' else -GRB.INFINITY,
                                     ub=GRB.INFINITY if sense == '<' else 0,
                                     name=f"slack_{constr.ConstrName}")
        
        # 构建新的等式约束
        expr = sum(row.getCoeff(i) * var_map[row.getVar(i)] for i in range(row.size()))
        expr += slack_var
        new_model.addConstr(expr == rhs, name=f"eq_{constr.ConstrName}")
    
    # 设置目标函数
    original_objective = model.getObjective()
    new_objective = sum(original_objective.getCoeff(i) * var_map[original_objective.getVar(i)] 
                        for i in range(original_objective.size()))
    new_model.setObjective(new_objective, model.ModelSense)  # 复制目标函数的方向

    # 更新模型
    new_model.update()
    new_model.setParam('Method', 0)
    return new_model

def recover_basis_np(predictions:np.ndarray, num_cons:int):
    solution = np.zeros_like(predictions.astype(int))
    top_idx = np.argpartition(-predictions, num_cons)[:num_cons]
    solution[top_idx] = 1
    return solution

def recover_basis(predictions, num_cons:int, logits=False):
    if not logits:
        return recover_basis_np(predictions, num_cons) if type(predictions) == np.ndarray else recover_basis_torch(predictions, num_cons)

def recover_basis_logits(predictions:torch.Tensor, num_cons:int):
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    assert predictions.shape[1] == 3
    top_idx = torch.topk(predictions[:, 1], num_cons).indices
    solution = torch.zeros(predictions.shape[0], dtype=torch.float)
    solution[top_idx] = 1.
    solution[predictions[:, 0] < predictions[:, 2]] = 2.
    return solution

def recover_basis_torch(predictions:torch.Tensor, num_cons:int):
    solution = torch.zeros_like(predictions, dtype=torch.int)
    top_idx = torch.topk(predictions, num_cons, dim=0).indices
    solution[top_idx] = 1
    return solution

def recover_basis_gurobi(basis:np.ndarray, coef:np.ndarray, num_vars:int):
    #TODO
    pass

def calculate_Aty(predictions:torch.Tensor, std_matrix:torch.Tensor):
    if std_matrix.is_sparse:
        return torch.sparse.mm(std_matrix.t(), predictions.view(-1, 1)).view(-1)
    else:
        return std_matrix.t() @ predictions


    
    
    
def dual_infer_primal_basis(predictions:torch.Tensor, std_matrix:torch.Tensor, coef:torch.Tensor):
    std_matrix_shape = std_matrix.shape
    Aty = calculate_Aty(predictions, std_matrix)
    
    error_cons = (Aty - coef)/ (torch.abs(coef) + 1e-6)
    error_dual = (predictions)/ (torch.max(torch.abs(predictions) + 1e-6))
    error = torch.cat([error_cons, error_dual])
    basis = torch.zeros_like(error, dtype=torch.int)
    topk_small = torch.topk(F.relu(error), std_matrix_shape[0], largest=False)
    basis[error > 0] = 2
    basis[error < 0] = 0
    basis[topk_small.indices] = 1
    return basis

def dual_infer_primal_solution(predictions:torch.Tensor, std_matrix:torch.Tensor, right_high_side:torch.Tensor, coef:torch.Tensor):
    Aty = calculate_Aty(predictions, std_matrix)
    solution = torch.zeros_like(coef, dtype=torch.int)
    error = Aty - coef
    solution[error >= 0] = 1
    return solution
    
    
    

def recover_solution(basis:torch.Tensor, std_matrix:torch.Tensor, right_high_side:torch.Tensor, coef:torch.Tensor, lb:torch.Tensor, ub:torch.Tensor) -> torch.Tensor:
    """
    Recovers the solution of a linear programming problem given the basis, standard matrix, right-hand side, coefficients, lower bounds, and upper bounds.
    Args:
        basis (torch.Tensor): The basis matrix.
        std_matrix (torch.Tensor): The standard matrix.
        right_high_side (torch.Tensor): The right-hand side vector.
        coef (torch.Tensor): The coefficient vector.
        lb (torch.Tensor): The lower bounds vector.
        ub (torch.Tensor): The upper bounds vector.
    Returns:
        torch.Tensor: The recovered solution vector. 
        Note that the solution contains both the variable values and the slack values.
    """
    basis = (basis == 1).bool()
    var_sol = torch.where(coef >= 0, lb, ub)
    #con_sol = torch.zeros(n_cons, device=device)
    sol = var_sol#torch.cat([var_sol, con_sol])
    sol[basis == 0] = lb[basis == 0]
    sol[basis == 2] = ub[basis == 2]
    #std_matrix = torch.cat([matrix, torch.eye(n_cons, device=device)], dim=1)
    sub_matrix = std_matrix[:, basis]
    sub_matrix_inv = torch.linalg.pinv(sub_matrix)
    
    z = sub_matrix_inv @ (right_high_side - std_matrix[:, ~basis] @ sol[~basis])
    sol[basis] = z
    
    # project back to the bound
    sol = torch.max(torch.min(sol, ub), lb)
    # sol[var_sol.shape[0]:] = torch.max(sol[var_sol.shape[0]:], 0)[0]
    
    return sol
    

def cat_sparse_identity(A):
    '''
    A: [n_cons, n_var] Sparse matrix
    Return:
    [n_cons, n_var + n_cons] Sparse matrix
    '''
    n_cons, n_vars = A.shape
    device = A.device

    identity_index = torch.stack([torch.arange(n_cons), torch.arange(n_cons)], dim=0)
    identity_attr = torch.ones(n_cons, device=device)

    identity_index[1, :] += n_vars

    new_index = torch.cat([A.indices(), identity_index], dim=1)
    new_attr = torch.cat([A.values(), identity_attr])

    return torch.sparse_coo_tensor(new_index, new_attr, (n_cons, n_vars + n_cons), device=device)

def select_columns(A, selected_cols):
    '''
    A: [n_cons, n_var] Sparse matrix
    selected_cols: List of column indices to select
    Return:
    Sparse submatrix with only the specified columns, with the same number of rows
    '''
    row_indices, col_indices = A.indices()
    values = A.values()

    mask = torch.isin(col_indices, torch.tensor(selected_cols, device=A.device))
    new_row_indices = row_indices[mask]
    new_col_indices = col_indices[mask]
    new_values = values[mask]
    
    col_map = {old: new for new, old in enumerate(selected_cols)}
    new_col_indices = torch.tensor([col_map[col.item()] for col in new_col_indices], device=A.device)

    submatrix = torch.sparse_coo_tensor(
        torch.stack([new_row_indices, new_col_indices]),
        new_values,
        (A.shape[0], len(selected_cols)), 
        device=A.device
    )
    return submatrix    
    
def feasibility(x, A, b, lb, ub):
    '''
    x: [n_var] or [n_samples, n_var]
    A: [n_cons, n_var] can be sparse
    b: [n_cons]
    lb: [n_var]
    ub: [n_var]
    Return:
    The norm of the violation
    '''
    violate_1 = torch
    violate_2 = torch.norm(torch.max(x - lb, torch.zeros_like(lb)), p=2)
    violate_3 = torch.norm(torch.max(ub - x, torch.zeros_like(ub)), p=2)
    return torch.max(violate_1, torch.max(violate_2, violate_3))

def objective(x, c):
    '''
    x: [n_var] or [n_samples, n_var]
    c: [n_var] or [n_samples, n_var]
    Return:
    The objective value
    '''
    return c @ x

def create_graph_data(file, to_eq=True):
    model = MIPmodel(file)
    model = model.generBipar(to_eq=to_eq)
    graph_data = GraphMILP()
    graph_data['vars'].x = model['x_var']
    graph_data['cons'].x = model['x_cons']
    graph_data['vars', 'to', 'cons'].edge_index = model['edge_index']
    graph_data['vars', 'to', 'cons'].edge_attr = model['edge_attr']
    return graph_data

def recover_approx_solution(graph_data, basis):
    basis = torch.tensor(basis)
    ori_problem = graph_data.restore_problem()
    #std_matrix = torch.cat([ori_problem['A'].to_dense(), torch.eye(ori_problem['A'].shape[0])], dim=1)
    std_matrix = ori_problem['A'].to_dense()
    solution = recover_solution(basis, std_matrix, ori_problem['b'], ori_problem['c'], ori_problem['lb'], ori_problem['ub'])
    solution = solution.cpu().tolist()
    return solution


def mask_with_knowledge_(probs, vars_feat, item = - torch.inf):
    """
    IN-PLACE FUNCTION
    Mask the probabilities with the knowledge of the variables.
    - If the lower bound is -inf, then the probability of 0 is set to 0.
    - If the upper bound is inf, then the probability of 2 is set to 0.

    """
    # If lb = -inf then, set the probability to 0
    probs[:, 0] = torch.where(vars_feat[:, 0] == 0, item, probs[:, 0])
    # If ub = inf then, set the probability to 0
    probs[:, 2] = torch.where(vars_feat[:, 1] == 0, item, probs[:, 2])
    
    return None


def mask_with_knowledge(probs, vars_feat):
    """
    Mask the probabilities with the knowledge of the variables.
    - If the lower bound is -inf, then the probability of 0 is set to 0.
    - If the upper bound is inf, then the probability of 2 is set to 0.
    
    returns:
    probs: masked probabilities
    """
    # Make a copy of the input tensor to avoid modifying it
    masked_probs = probs.clone()
    
    mask_with_knowledge_(masked_probs, vars_feat)
    
    return masked_probs

def mask_and_divide_dominance(probs, var_feat):
    masked_probs = probs.clone()
    mask_with_knowledge_(masked_probs, var_feat, 0)
    masked_probs = masked_probs / torch.sum(masked_probs, dim=1, keepdim=True)
    return masked_probs

def normalize_problem(graph_data):
    pass