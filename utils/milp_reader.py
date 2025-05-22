import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import torch
import pickle
class MIPmodel:
    """
    Class for reading MILP by Gurobi
    MILP Formulation:
    
    """
    def __init__(self, file_path, name='default', log=False):
        self.model = gp.read(file_path)
        self.hasSolved = False
        self.log = log
        self.file_path = file_path
        if self.log:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)

    def presolve_(self, presolve=2):
        """
        Presolve the MIP problem
        """
        self.model.setParam('Presolve', presolve)
        self.model = self.model.presolve()
        
        
        
    def solve(self):
        """Solve the MIP problem"""
        self.model.optimize()
        self.hasSolved = True


    def showResult(self):
        """Displays result attributes

        Returns:
            dict: A dictionary with result attribute names and their values
        """
        self.solve()
        result_dict = {}
        for attr in self.resultLis:
            result_dict[attr] = self.model.getAttr(attr)
        return result_dict

    def setTimeLimit(self, time):
        """Sets the time limit for solving

        Args:
            time (int): Time limit for solving
        """
        self.model.setParam('TimeLimit', time)

    def getObjective(self):
        return self.model.getObjective()

    def getVars(self):
        return self.model.getVars()

    def getConstrs(self):
        return self.model.getConstrs()

    def getA(self):
        return self.model.getA()
    
    def setModel(self, model:gp.Model):
        """
        Set the model to a new Gurobi model.

        Args:
            model (gp.Model): The new Gurobi model to set.
        """
        self.model = model
        self.hasSolved = False

    def varsDict(self):
        vars_dict = {}
        vars_ = self.getVars()
        for i in range(len(vars_)):
            vars_dict[i] = vars_[i].VarName
        return vars_dict

    def generGraphCom(self, solution_info_path=None):
        """
        Generates the graph components. 

        Args:
            file_path (str): File path (default is None)
            
        Returns:
            Bipartite graph components
            The input LP format is as follows:
            min     c^T x
            s.t.    Ax >= b
                    x = [x_I, x_C] 
                    l <= x_C <= u
                    x_I \in {0, 1, ..., K}
        """
    
        # try:
        solution = False if solution_info_path is None else True
        sol_info = pickle.load(open(solution_info_path, 'rb')) if solution else None
        # Generate the constraint matrix as a sparse matrix
        A = sp.csr_matrix(self.model.getA())
        label_info = None if not solution else sol_info['label_info']
        # Extract matrix data for bipartite graph representation
        data = A.data
        indices = A.indices
        indptr = A.indptr
        vars_ = self.getVars()
        constrs = self.getConstrs()
        m, n = A.shape
        x_cons = []
        x_Cvars = []
        x_Ivars = []
        I2cons_edge_attr = []
        C2cons_edge_attr = []
        I2cons_edge_index = [[], []]
        C2cons_edge_index = [[], []]
        
        Ivars_indices = {}
        Cvars_indices = {}
        num_Ivars = 0
        num_Cvars = 0
        for i in range(n):
            varLB = vars_[i].LB
            varUB = vars_[i].UB
            hasLB = 1
            hasUB = 1
            if varLB == float('-inf'):
                varLB = 0
                hasLB = 0
            if varUB == float('inf'):
                varUB = 0
                hasUB = 0
            obj = vars_[i].Obj
            label = 0 if not solution else label_info[vars_[i].VarName]
            assert float('-inf') < varLB < float('inf')
            assert float('-inf') < varUB < float('inf')
            assert float('-inf') < obj < float('inf')
            
            if vars_[i].VType == GRB.CONTINUOUS:
                x_Cvars.append([varLB, varUB, obj, hasLB, hasUB, label])
                Cvars_indices[i] = num_Cvars
                num_Cvars += 1
            else:
                x_Ivars.append([varLB, varUB, obj, hasLB, hasUB, label])
                Ivars_indices[num_Ivars] = i
                Ivars_indices[i] = num_Ivars
                num_Ivars += 1
                

        for i in range(m):
            start_idx = indptr[i]
            end_idx = indptr[i + 1]

            UB = constrs[i].RHS
            x_cons.append([UB])

            
            for k in range(start_idx, end_idx):
                if data[k] == 0:
                    continue 
                
                var_idx = indices[k]
                if vars_[var_idx].VType == GRB.CONTINUOUS:
                    C2cons_edge_attr.append([data[k]])
                    C2cons_edge_index[1].append(i)
                    C2cons_edge_index[0].append(Cvars_indices[var_idx])
                else:
                    I2cons_edge_attr.append([data[k]])
                    I2cons_edge_index[1].append(i)
                    I2cons_edge_index[0].append(Ivars_indices[var_idx])
        # Convert to tensors for PyTorch
        x_cons = torch.tensor(x_cons, dtype=torch.float)
        x_Cvars = torch.tensor(x_Cvars, dtype=torch.float)
        x_Ivars = torch.tensor(x_Ivars, dtype=torch.float)
        C2cons_edge_attr = torch.tensor(C2cons_edge_attr, dtype=torch.float)
        I2cons_edge_attr = torch.tensor(I2cons_edge_attr, dtype=torch.float)
        C2cons_edge_index = torch.tensor(C2cons_edge_index, dtype=torch.long)
        I2cons_edge_index = torch.tensor(I2cons_edge_index, dtype=torch.long)
        

        readed_data = {
            'x_cons': x_cons,
            'x_Cvars': x_Cvars,
            'x_Ivars': x_Ivars,
            'C2cons_edge_attr': C2cons_edge_attr,
            'I2cons_edge_attr': I2cons_edge_attr,
            'C2cons_edge_index': C2cons_edge_index,
            'I2cons_edge_index': I2cons_edge_index,
            'if_labeled': solution
        }

        return readed_data
        # except Exception as e:
        #     print(self.file_path)
        #     raise e
        
    def are_all_integer_vars_binary(self):
        model = self.model
        for v in model.getVars():
            if v.VType in [GRB.INTEGER, GRB.BINARY]:
                if v.VType == GRB.INTEGER:
                    if not (v.LB == 0 and v.UB == 1):
                        return False
        return True
if __name__ == '__main__':
    import os 

    load_balancing_path = "/data/GM4MILP/datasets/load_balancing"
    file_path = os.path.join(load_balancing_path, os.listdir(load_balancing_path)[0])
    # Read the MILP file
    model = MIPmodel(file_path, log=True)
    # model.presolve_()
    print(model.are_all_integer_vars_binary())
    print(model.generGraphCom())
    model.solve()