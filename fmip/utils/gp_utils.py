import gurobipy as gp
import torch
import numpy as np
from pl_gmip_model import GMIPModel
from utils.milp_reader import MIPmodel
from dataset import toHeteroData
from utils.lp_utils import recover_basis_np, transform_to_equalities, recover_basis_logits
def gurobi_basis_warm_start(file_path, var_basis, con_basis):
    m = gp.read(file_path)
    m.setParam('LPWarmStart', 2)
    m.setParam('Method', 0)
    m.vbasis = var_basis
    m.cbasis = con_basis
    m.optimize()
    metrics = {}
    metrics['status'] = m.status
    metrics['obj'] = m.ObjVal
    metrics['solving_time'] = m.Runtime
    metrics['iterations'] = m.IterCount
    return metrics

def to_basis_we_need(basis):
    return 2 if basis == -2 else (0 if basis == -3 else basis + 1)

def to_gurobi_basis(basis):
    return -2 if basis == 2 else basis - 1
    
def extract_primal_solution(model):
    num_vars = model.NumVars
    solution = []
    for i in range(num_vars):
        var = model.getVars()[i]
        solution.append(var.x)
    return solution

def pred_by_model(mip_model: MIPmodel, model, guidance = False, device = 'cuda', parallel_size = 1):
    data = mip_model.generGraphCom()
    graph = toHeteroData(data['x_cons'], data['x_Cvars'], data['x_Ivars'], 
                                        data['C2cons_edge_attr'], data['I2cons_edge_attr'],
                                        data['C2cons_edge_index'], data['I2cons_edge_index']).to(device)
    if isinstance(model, GMIPModel):
        result = model.inference(graph, guidance = guidance, batchsize = parallel_size)


    return result

def pred_FMIPump(name, model:GMIPModel, guidance = False, device = 'cuda'):
    data = MIPmodel(name).generGraphCom()
    graph = toHeteroData(data['x_cons'], data['x_Cvars'], data['x_Ivars'], 
                                        data['C2cons_edge_attr'], data['I2cons_edge_attr'],
                                        data['C2cons_edge_index'], data['I2cons_edge_index']).to(device)
    return model.inference_FMIPump(graph, name = name, solution_pool_size=100)
def extract_dual_solution(model):
    num_cons = model.NumConstrs
    solution = []
    for i in range(num_cons):
        constr = model.getConstrs()[i]
        solution.append(constr.pi)
    return solution

def input_primal_basis(model, basis):
    approx_basis_gurobi = [to_gurobi_basis(i) for i in basis]
    for var in model.getVars():
        var.setAttr("VBasis", int(approx_basis_gurobi[var.index]))
    return model

def input_basis(model, basis, equality = False):
    if equality:
        return input_basis_eq(model, basis)
    else:
        return input_basis_ori(model, basis)

def input_basis_eq(model, basis):
    approx_basis_gurobi = [to_gurobi_basis(i) for i in basis]
    for var in model.getVars():
        var.setAttr("VBasis", int(approx_basis_gurobi[var.index]))
    for constr in model.getConstrs():
        constr.setAttr("CBasis", -1)
    return model

def input_basis_ori(model, basis):
    approx_basis_gurobi = [to_gurobi_basis(i) for i in basis]
    approx_vbasis = approx_basis_gurobi[:model.NumVars]
    approx_cbasis = approx_basis_gurobi[model.NumVars:]
    for var in model.getVars():
        var.setAttr("VBasis", int(approx_vbasis[var.index]))
    for constr in model.getConstrs():
        constr.setAttr("CBasis", int(approx_cbasis[constr.index]))
    return model

def input_solution(model, solution):
    approx_solution_primal = solution[:model.NumVars]
    approx_solution_dual = solution[model.NumVars:]
    for var in model.getVars():
        var.setAttr("PStart", approx_solution_primal[var.index])
    for constr in model.getConstrs():
        constr.setAttr("DStart", approx_solution_dual[constr.index])
    return model

def input_primal_solution(model, solution):
    approx_solution_primal = solution
    for var in model.getVars():
        var.setAttr("PStart", approx_solution_primal[var.index])
    return model

    
def gurobi_solve(file, model = 'cash', predict = None, equality = False, **kwargs):
    m = init_model(file, equality)
    m.setParam('Presolve', 0)
    if model == 'default':
        m.setParam('LPWarmStart', 1)
        init_basis = [0 for i in range(m.NumVars)] + [1 for i in range(m.NumConstrs)]
        m = input_basis(m, init_basis, equality)
    elif model == 'warm_start_1':
        m.setParam('LPWarmStart', 1)
        init_basis = recover_basis_logits(predict, m.NumConstrs)
        m = input_basis(m, init_basis, equality)
    elif model == 'warm_start_2':
        m.setParam('LPWarmStart', 2)
        init_basis = recover_basis_logits(predict, m.NumConstrs)
        m = input_basis(m, init_basis, equality)
    elif model == 'warm_start_3':
        m.setParam('LPWarmStart', 2)
        init_basis = recover_basis_logits(predict, m.NumConstrs)
        m = input_basis(m, init_basis, equality)
    m.optimize()
    return m

def extract_constraint_matrix(model):
    num_vars = model.NumVars
    num_cons = model.NumConstrs
    A = torch.zeros((num_cons, num_vars))

    for i in range(num_cons):
        constr = model.getConstrs()[i]
        for j in range(num_vars):
            var = model.getVars()[j]
            coeff = model.getCoeff(constr, var)
            if coeff != 0:
                A[i, j] = coeff

    return A
def init_model(file, equalities=False, simplex=True):
    m = gp.read(file)
    if equalities:
        m = transform_to_equalities(m)
    m.setParam('Method', 0) if simplex else None
    m.setParam('LPWarmStart', 2)
    m.setParam('Presolve', 0)
    return m
