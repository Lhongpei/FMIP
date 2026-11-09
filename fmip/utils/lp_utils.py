import torch 
import torch.nn as nn
import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB
from fmip.utils.milp_reader import MIPmodel
from fmip.dataset import GraphMILP
from fmip.utils.milp_utils import constraint_matmul
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
    Aty = constraint_matmul(std_matrix, predictions)

    # Compute the mask for ReLU
    mask = (-coef + Aty > 0).float()  # Mask: 1 if ReLU is active, else 0

    # Compute the gradient
    gradient = right_high_side - constraint_matmul(std_matrix.T, mask)

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
            Vars[i].lb = fixed_value  # Lower bound
            Vars[i].ub = fixed_value  # Upper bound
    return m
    # for i in range(m.NumConstrs):
    #     if predictions[i + m.NumVars] > confidence:
    #         m.getConstr(i).setAttr('Lazy', 1)
        
def transform_to_equalities(model):
    """
    Convert all constraints in a Gurobi model to the form Ax + s = b.
    """
    # Create a new model.
    new_model = Model("EqualityModel")
    
    # Copy variables and build the mapping.
    var_map = {}  # Map original variables to their copies.
    for var in model.getVars():
        new_var = new_model.addVar(lb=var.lb, ub=var.ub, obj=0,  # Set objective coefficients below.
                                   vtype=var.vtype, name=var.varName)
        var_map[var] = new_var
    new_model.update()
    
    # Convert the constraints.
    for constr in model.getConstrs():
        sense = constr.Sense  # Constraint sense (<=, ==, >=)
        rhs = constr.getAttr("RHS")  # Right-hand side
        row = model.getRow(constr)  # Retrieve the constraint coefficients.
        
        # Add slack variables.
        # Build the equivalent equality constraint.
        expr = sum(row.getCoeff(i) * var_map[row.getVar(i)] for i in range(row.size()))
        if sense != GRB.EQUAL:
            slack_var = new_model.addVar(lb=0 if sense == '<' else -GRB.INFINITY,
                                        ub=GRB.INFINITY if sense == '<' else 0,
                                        name=f"slack_{constr.ConstrName}")
            expr += slack_var
        new_model.addConstr(expr == rhs, name=f"eq_{constr.ConstrName}")
    
    # Set the objective function.
    original_objective = model.getObjective()
    new_objective = sum(original_objective.getCoeff(i) * var_map[original_objective.getVar(i)] 
                        for i in range(original_objective.size())) + original_objective.getConstant()
    new_model.setObjective(new_objective, model.ModelSense)  # Preserve the objective sense.

    # Update the model.
    new_model.update()
    new_model.setParam('Method', 0)
    return new_model

def recover_basis_np(predictions:np.ndarray, num_cons:int):
    solution = np.zeros_like(predictions.astype(int))
    if not 0 <= num_cons <= predictions.size:
        raise ValueError("num_cons must be between zero and the number of variables")
    top_idx = np.argpartition(-predictions, num_cons - 1)[:num_cons] if num_cons else []
    solution[top_idx] = 1
    return solution

def recover_basis(predictions, num_cons:int, logits=False):
    if logits:
        return recover_basis_logits(predictions, num_cons)
    return recover_basis_np(predictions, num_cons) if isinstance(predictions, np.ndarray) else recover_basis_torch(predictions, num_cons)

def recover_basis_logits(predictions:torch.Tensor, num_cons:int):
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    assert predictions.shape[1] == 3
    top_idx = torch.topk(predictions[:, 1], num_cons).indices
    solution = torch.zeros(predictions.shape[0], dtype=torch.float, device=predictions.device)
    solution[predictions[:, 0] < predictions[:, 2]] = 2.
    solution[top_idx] = 1.
    return solution

def recover_basis_torch(predictions:torch.Tensor, num_cons:int):
    solution = torch.zeros_like(predictions, dtype=torch.int)
    top_idx = torch.topk(predictions, num_cons, dim=0).indices
    solution[top_idx] = 1
    return solution

def recover_basis_gurobi(basis:np.ndarray, coef:np.ndarray, num_vars:int):
    raise NotImplementedError('Use recover_basis followed by gp_utils.input_basis with an explicit Gurobi model')

def calculate_Aty(predictions:torch.Tensor, std_matrix:torch.Tensor):
    return constraint_matmul(std_matrix.t(), predictions)


    
    
    
def dual_infer_primal_basis(predictions:torch.Tensor, std_matrix:torch.Tensor, coef:torch.Tensor):
    std_matrix_shape = std_matrix.shape
    Aty = calculate_Aty(predictions, std_matrix)
    
    error_cons = (Aty - coef)/ (torch.abs(coef) + 1e-6)
    error_dual = (predictions)/ (torch.max(torch.abs(predictions) + 1e-6))
    error = torch.cat([error_cons, error_dual])
    basis = torch.zeros_like(error, dtype=torch.int)
    topk_small = torch.topk(torch.abs(error), std_matrix_shape[0], largest=False)
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
    basic = basis == 1
    sol = torch.where(basis == 2, ub, lb).clone()
    if not torch.isfinite(sol[~basic]).all():
        raise ValueError('A nonbasic variable must have a finite bound for its selected basis status')
    #std_matrix = torch.cat([matrix, torch.eye(n_cons, device=device)], dim=1)
    sub_matrix = std_matrix[:, basic]
    sub_matrix_inv = torch.linalg.pinv(sub_matrix)
    
    z = sub_matrix_inv @ (right_high_side - std_matrix[:, ~basic] @ sol[~basic])
    sol[basic] = z
    
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
    A = A.coalesce()
    n_cons, n_vars = A.shape
    device = A.device

    identity_index = torch.stack([torch.arange(n_cons, device=device), torch.arange(n_cons, device=device)], dim=0)
    identity_attr = torch.ones(n_cons, device=device, dtype=A.dtype)

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
    A = A.coalesce()
    selected_cols = torch.as_tensor(selected_cols, device=A.device, dtype=torch.long).reshape(-1)
    if ((selected_cols < 0) | (selected_cols >= A.shape[1])).any():
        raise IndexError('Column index outside sparse matrix dimensions')
    row_indices, col_indices = A.indices()
    values = A.values()
    rows, columns, selected_values = [], [], []
    for new_column, original_column in enumerate(selected_cols):
        mask = col_indices == original_column
        rows.append(row_indices[mask])
        columns.append(torch.full_like(row_indices[mask], new_column))
        selected_values.append(values[mask])
    new_row_indices = torch.cat(rows) if rows else row_indices[:0]
    new_col_indices = torch.cat(columns) if columns else col_indices[:0]
    new_values = torch.cat(selected_values) if selected_values else values[:0]
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
    The maximum of constraint and bound violation norms for Ax <= b.
    Batched inputs return one violation value per sample.
    '''
    Ax = constraint_matmul(A, x) if x.ndim == 1 else constraint_matmul(A, x.t()).t()
    violate_1 = torch.linalg.vector_norm(torch.relu(Ax - b), dim=-1)
    violate_2 = torch.linalg.vector_norm(torch.relu(lb - x), dim=-1)
    violate_3 = torch.linalg.vector_norm(torch.relu(x - ub), dim=-1)
    return torch.maximum(violate_1, torch.maximum(violate_2, violate_3))

def objective(x, c):
    '''
    x: [n_var] or [n_samples, n_var]
    c: [n_var] or [n_samples, n_var]
    Return:
    The objective value
    '''
    return (c * x).sum(dim=-1)

def create_graph_data(file, to_eq=True):
    """Build the legacy LP-basis graph in Gurobi variable order.

    The default adds slack variables so basis recovery operates on equalities.
    Neural FMIP inference should use MIPmodel.generGraphCom/toHeteroData instead.
    """
    reader = MIPmodel(file)
    model = transform_to_equalities(reader.model) if to_eq else reader.model
    matrix = model.getA().tocoo()
    variables = model.getVars()
    graph_data = GraphMILP()
    graph_data['vars'].x = torch.tensor([
        [var.LB if var.LB > -GRB.INFINITY else 0.,
         var.UB if var.UB < GRB.INFINITY else 0., var.Obj,
         var.LB > -GRB.INFINITY, var.UB < GRB.INFINITY, 0.]
        for var in variables], dtype=torch.float32).reshape(-1, 6)
    graph_data['cons'].x = torch.tensor([con.RHS for con in model.getConstrs()], dtype=torch.float32).reshape(-1, 1)
    graph_data['vars', 'to', 'cons'].edge_index = torch.tensor(np.stack([matrix.col, matrix.row]), dtype=torch.long)
    graph_data['vars', 'to', 'cons'].edge_attr = torch.tensor(matrix.data, dtype=torch.float32).reshape(-1, 1)
    graph_data.constraint_senses = [con.Sense for con in model.getConstrs()]
    graph_data.model_sense = model.ModelSense
    graph_data.objective_constant = model.ObjCon
    return graph_data

def recover_approx_solution(graph_data, basis):
    if 'vars' not in graph_data.node_types:
        raise ValueError('Basis recovery requires create_graph_data(..., to_eq=True), not an FMIP tripartite graph')
    if any(sense != GRB.EQUAL for sense in graph_data.constraint_senses):
        raise ValueError('Basis recovery requires equality form; use create_graph_data(..., to_eq=True)')
    features = graph_data['vars'].x
    edges = graph_data['vars', 'to', 'cons']
    std_matrix = torch.sparse_coo_tensor(edges.edge_index[[1, 0]], edges.edge_attr.reshape(-1),
                                         (graph_data['cons'].x.shape[0], features.shape[0]),
                                         device=features.device).to_dense()
    basis = torch.as_tensor(basis, device=features.device)
    lower = torch.where(features[:, 3].bool(), features[:, 0], -torch.inf)
    upper = torch.where(features[:, 4].bool(), features[:, 1], torch.inf)
    solution = recover_solution(basis, std_matrix, graph_data['cons'].x.reshape(-1), features[:, 2], lower, upper)
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
    probs[:, 0] = torch.where(~vars_feat[:, 3].bool(), item, probs[:, 0])
    # If ub = inf then, set the probability to 0
    probs[:, 2] = torch.where(~vars_feat[:, 4].bool(), item, probs[:, 2])
    
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
    total = masked_probs.sum(dim=1, keepdim=True)
    fallback = torch.zeros_like(masked_probs)
    fallback[:, 1] = 1  # A free variable must be basic.
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    return torch.where(total > 0, masked_probs / denominator, fallback)

def normalize_problem(graph_data):
    raise NotImplementedError('Problem normalization requires an explicit reversible scaling convention')
