import gurobipy as gp
from gurobipy import GRB
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pl_gmip_model import GMIPModel
from utils.milp_reader import MIPmodel
from utils.gp_utils import pred_by_model
import copy
import pickle
import time
from arg import default_args

time_points = []
iter_points = []
obj_values = []

def save_history(model, where):
    if where == GRB.Callback.MIP:
        # 获取当前运行时间（秒）
        current_time = model.cbGet(GRB.Callback.RUNTIME)

        # Only record every 10 seconds
        if len(time_points) == 0 or current_time - time_points[-1] >= 1:
            # 获取当前最优解的目标值
            current_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            
            current_iteration = model.cbGet(GRB.Callback.MIP_ITRCNT)
            
                # 记录时间和 Gap
            time_points.append(current_time)
            obj_values.append(current_obj)
            iter_points.append(current_iteration)

class ApolloOptimizer:
    def __init__(self, root_dir, problem_set, policy_path, set_threads = None, timelimit = None, guidance = False, sense = "min"):
        self.root_dir = root_dir
        self.problem_root = os.path.join(root_dir, problem_set, 'problem')
        self.sense = sense
        self.problem_set = problem_set
        self.load_policy(policy_path)
        # self.solution_root = os.path.join(root_dir, 'datasets', problem_set, 'solution')
        self.result_dict = {}
        self.set_threads = set_threads
        self.timelimit = timelimit
        self.total_timelimit = timelimit
        self.guidance = guidance
        self.solving_history_dict = {}


    def load_policy(self, policy_path):
        args = default_args()
        args.sense = self.sense
        args.max_integer_num = 2
        if self.problem_set in ["load_balance", "item_placement"] and "gnn" not in policy_path:
            args.only_discrete = False
        else:
            args.only_discrete = True
        self.policy = GMIPModel.load_from_checkpoint(policy_path, param_args=args, load_datasets=False)  
        self.policy.eval()
        self.policy.to('cuda')
    
    def _get_variable_info(self, model: gp.Model):
        """Extract variable information from the model"""
        vars_list = model.getVars()
        
        Ivars_indice = [i for i, var in enumerate(vars_list) 
                        if var.VType in [gp.GRB.INTEGER, gp.GRB.BINARY]]
        # Ivars_origin_lbub = [(var.LB, var.UB) for var in vars_list 
        #                     if var.VType in [gp.GRB.INTEGER, gp.GRB.BINARY]]
        Cvars_indice = [i for i, var in enumerate(vars_list) 
                        if var.VType == gp.GRB.CONTINUOUS]
        
        return vars_list, Ivars_indice, Cvars_indice
    
    def apply_predsearch_strategy(self,
                                  model: gp.Model,
                                  pred_Ivars_prob,
                                  k_0,
                                  k_1,
                                  delta_rate):
        """Apply predsearch strategy for warm start"""

        vars_list, Ivars_indice, _ = self._get_variable_info(model)

        max_prob_Ivars = pred_Ivars_prob[:,0] - pred_Ivars_prob[:,1]
        pred_Ivars = np.argmax(pred_Ivars_prob, axis=1)
        sorted_idx = np.argsort(max_prob_Ivars)[::-1]

        if k_1 != 0:
            fixed_1_ind = sorted_idx[-int(len(sorted_idx) * k_1):]
            pred_Ivars[fixed_1_ind] = 1
        else:
            fixed_1_ind = []
        
        if k_0 != 0:
            fixed_0_ind = sorted_idx[:int(len(sorted_idx) * k_0)]
            pred_Ivars[fixed_0_ind] = 0
        else:
            fixed_0_ind = []        

        trust_bound_ind = [*fixed_0_ind, *fixed_1_ind]
        delta = int(len(trust_bound_ind) * delta_rate)
        alpha = model.addVars(len(trust_bound_ind), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1)
        Ivars_list = [vars_list[i] for i in Ivars_indice]
        for i, idx in enumerate(trust_bound_ind):
            pred_x_i = pred_Ivars[idx]
            if pred_x_i == 0:
                model.addConstr(Ivars_list[idx] <= alpha[i])
            elif pred_x_i == 1:
                model.addConstr(1 - Ivars_list[idx] <= alpha[i])
        model.addConstr(gp.quicksum(alpha) <= delta)
        model.update()

        if self.set_threads is not None:
            model.setParam('Threads', self.set_threads)
        if self.timelimit is not None:
            model.setParam('Timelimit', max(1, min(self.timelimit, 400)))

        time0 = time.time()
        model.optimize(save_history)

        if model.Status in [gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED]:
            best_model = None
            bin_solution = None
        else:
            best_model = model
            bin_solution = [var.X for var in best_model.getVars() if var.VType == gp.GRB.BINARY]
        solving_time = time.time() - time0

        if self.timelimit is not None:
            self.timelimit -= solving_time
        return best_model, bin_solution

    def process_instance_apollo(self, prob_name, params: list):
        for param in params:
            result = {}
            self.timelimit = self.total_timelimit
            best_model, solving_time = self.apply_apollo_strategy(prob_name, param)
            
            if best_model is not None:
                result['apollo_solving_time'] = solving_time
                result['apollo_obj'] = best_model.ObjVal
            self.result_dict[str(param)][prob_name] = result
            self.solving_history_dict[str(param)][prob_name] = {}
            self.solving_history_dict[str(param)][prob_name]['time'] = time_points
            self.solving_history_dict[str(param)][prob_name]['iter'] = iter_points
            self.solving_history_dict[str(param)][prob_name]['obj'] = obj_values
            time_points.clear()
            iter_points.clear()
            obj_values.clear()

    def get_prediction(self, model):
        self.MIPModel.setModel(model)
        result = pred_by_model(self.MIPModel, self.policy, self.guidance)
        return result

    def apply_apollo_strategy(self, prob_name, param:list):
        problem_path = os.path.join(self.problem_root, prob_name)
        model = gp.read(problem_path)
        self.MIPModel = MIPmodel(problem_path)
        param = np.array(param)
        if_fixed = np.zeros(len(model.getVars()), dtype = np.bool)

        if len(param.shape) == 1:
            k_0, k_1, delta_rate, epoch = param
            epoch = int(epoch)
            k_0s = np.array([k_0 for _ in range(epoch)])
            k_1s = np.array([k_1 for _ in range(epoch)])
            delta_rates = np.array([delta_rate for _ in range(epoch)])
        else:
            epoch = param.shape[0]
            k_0s = param[:, 0]
            k_1s = param[:, 1]
            delta_rates = param[:, 2]
        
        best_model = None
        best_obj = np.inf if model.ModelSense == gp.GRB.MINIMIZE else -np.inf

        time0 = time.time()

        for i in range(epoch):
            model_pred = model.copy()
            pred_solution = self.get_prediction(model_pred)
            pred_Ivars_prob = pred_solution['Ivars']
            model_solved, bin_solution = self.apply_predsearch_strategy(model_pred,
                                                                        pred_Ivars_prob,
                                                                        k_0s[i],
                                                                        k_1s[i],
                                                                        delta_rates[i])
            if model_solved and \
                (model.ModelSense == gp.GRB.MINIMIZE and model_solved.ObjVal < best_obj) or \
                (model.ModelSense == gp.GRB.MAXIMIZE and model_solved.ObjVal > best_obj):
                best_obj = model_solved.ObjVal
                best_model = model_solved
            if i < epoch - 1:
                intersect_ind = np.where(np.argmax(pred_Ivars_prob, axis = 1) == bin_solution)[0]
                if intersect_ind.size > 0:
                    self.fix_by_ind(model, intersect_ind, np.argmax(pred_Ivars_prob, axis = 1))
                    if_fixed[intersect_ind] = True
        
        solving_time = time.time() - time0
        return best_model, solving_time 

    def fix_by_ind(self, model: gp.Model, index, pred_solution):
        vars_list = model.getVars()
        for i in index:
            vars_list[i].LB = pred_solution[i]
            vars_list[i].UB = pred_solution[i]
        model.update()

    def run_all(self, params: list):
        """Process all prediction files"""
        with open(f'/data/GM4MILP/dataset_split/{self.problem_set}/split.pkl', 'rb') as f:
            split = pickle.load(f)['test']
        prob_name_list = np.sort(split)[:10]
        self.result_dict = {str(para): {} for para in params}
        self.solving_history_dict = {str(para): {} for para in params}
        for prob_name in tqdm(prob_name_list):
            self.process_instance_apollo(prob_name, params)
                
        return self.result_dict
    
    def save_all_results(self, prefix='warmstart_result'):
        """Save all results to CSV"""
        for param, result in self.result_dict.items():
            df = pd.DataFrame(result).T
            df.to_csv(f'{prefix}_{str(param)}.csv')
        for param, history in self.solving_history_dict.items():
            with open(f'{prefix}_{str(param)}_history.pkl', 'wb') as f:
                pickle.dump(history, f)
        
        return self.result_dict

if __name__ == "__main__":
    root_dir = '/data/GM4MILP/instances_final'
    params_dict = {
        'cauctions': [
            [0.3, 0.06, 0.3, 2]
        ],
        'fcmnf': [
            [0.3, 0.03, 0.2, 2]
        ],
        'gisp': [
            [0.2, 0.02, 0.2, 2]
        ],
        'indset': [
            [0.3, 0.2, 0.3, 2]
        ],
        'item_placement': [
            [0.3, 0.08, 0.4, 2]
        ],
        'load_balance': [
            [0.2, 0.2, 0.2, 2]
        ],
        'setcover': [
            [0.3, 0.04, 0.2, 2]
        ],
    }
    senses = {
        'cauctions': 'max',
        'gisp': 'max',
        'indset': 'max',
        'fcmnf': 'min',
        'item_placement': 'min',
        'load_balance': 'min',
        'setcover': 'min',
    }
    source = "gnn"
    guidance = False
    threads = 32
    # for problem_set in ['gisp', 'indset', 'setcover', 'fcmnf', 'cauctions', 'load_balance', 'item_placement']:
    for problem_set in ['item_placement']:
        params = params_dict[problem_set]
        sense = senses[problem_set]
        policy_root = f'/data/GM4MILP/result_cache/models/{"" if source == "diff" else "gnn_"}{problem_set}/'
        save_folders = np.sort(os.listdir(policy_root))[-1]
        policy_folder = os.path.join(policy_root, save_folders, "checkpoints")
        policy_path = os.path.join(policy_folder, np.sort(os.listdir(policy_folder))[2])
        
        optimizer = ApolloOptimizer(root_dir, problem_set, policy_path, set_threads = threads, timelimit = 600, guidance = guidance, sense = sense)
        optimizer.run_all(params)
        optimizer.save_all_results(prefix = f'downstream_result/apollo/{source}_apollo_{problem_set}{"_guidance" if guidance and source == "diff" else ""}')