import gurobipy as gp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from fmip.utils.gp_utils import pred_by_model
import copy
from gurobipy import GRB
import time
import pickle
from fmip.pl_gmip_model import GMIPModel
from fmip.utils.milp_reader import MIPmodel
from fmip.arg import default_args

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

class PredsearchOptimizer:
    def __init__(self, root_dir, problem_set, policy_path, set_threads = None, timelimit = 600, guidance = False, sense = "min"):
        self.root_dir = root_dir
        self.problem_root = os.path.join(root_dir, problem_set, 'problem')
        self.sense = sense
        self.load_policy(policy_path)
        self.result_dict = {}
        self.set_threads = set_threads
        self.timelimit = timelimit
        self.guidance = guidance
        self.problem_set = problem_set
        self.solving_history_dict = {}

    def load_policy(self, policy_path):
        args = default_args()
        args.sense = self.sense
        args.max_integer_num = 2
        if problem_set in ["load_balance", "item_placement"] and "gnn" not in policy_path:
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
                                  prob_name,
                                  param):
        """Apply predsearch strategy for warm start"""
        k_0, k_1, delta_rate = param

        self.MIPModel = MIPmodel(os.path.join(self.problem_root, prob_name))
        model = gp.read(os.path.join(self.problem_root, prob_name))
        pred_solution = self.get_prediction()
        pred_Ivars_prob = pred_solution['Ivars']
        
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
            model.setParam('Timelimit', self.timelimit)

        time0 = time.time()
        model.optimize()

        if model.Status in [gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED]:
            best_model = None
        else:
            best_model = model
        solving_time = time.time() - time0

        return best_model, solving_time

    def process_instance_predsearch(self, prob_name, params: list):
        for param in params:

            result = {}
            best_model, solving_time = self.apply_predsearch_strategy(prob_name, param)
            
            result['predsearch_solving_time'] = solving_time
            if best_model is not None:
                result['predsearch_obj'] = best_model.ObjVal
                result['predsearch_iter'] = best_model.IterCount
            self.result_dict[str(param)][prob_name] = result
            self.solving_history_dict[str(param)][prob_name] = {}
            self.solving_history_dict[str(param)][prob_name]['time'] = time_points
            self.solving_history_dict[str(param)][prob_name]['iter'] = iter_points
            self.solving_history_dict[str(param)][prob_name]['obj'] = obj_values
            time_points.clear()
            iter_points.clear()
            obj_values.clear()
       
    def get_prediction(self):
        result = pred_by_model(self.MIPModel, self.policy, self.guidance)
        return result
    
    def run_all(self, params: list):
        """Process all prediction files"""
        self.result_dict = {str(para): {} for para in params}
        self.solving_history_dict = {str(para): {} for para in params}
        with open(f'/data/GM4MILP/dataset_split/{self.problem_set}/split.pkl', 'rb') as f:
            split = pickle.load(f)['test']
        prob_name_list = np.sort(split)[0:10]
        for prob_name in tqdm(prob_name_list):
            self.process_instance_predsearch(prob_name, params)
                
        return self.result_dict
    
    def save_all_results(self, prefix='warmstart_result'):
        """Save all results to CSV"""
        # for param, result in self.result_dict.items():
        #     df = pd.DataFrame(result).T
        #     df.to_csv(f'{prefix}_{str(param)}.csv')
        for param, history in self.solving_history_dict.items():
            with open(f'{prefix}_{str(param)}_history.pkl', 'wb') as f:
                pickle.dump(history, f)
        
        return self.result_dict

if __name__ == "__main__":
    root_dir = '/data/GM4MILP/instances_final'
    params_dict = {
        'cauctions': [
            [0.3, 0.06, 0.3]
        ],
        'fcmnf': [
            [0.3, 0.03, 0.2]
        ],
        'gisp': [
            [0.2, 0.02, 0.2]
        ],
        'indset': [
            [0.3, 0.2, 0.3]
        ],
        'item_placement': [
            [0.3, 0.08, 0.4]
        ],
        'load_balance': [
            [0.2, 0.2, 0.2]
        ],
        'setcover': [
            [0.3, 0.04, 0.2]
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
    source = "diff"
    guidance = False
    Timelimit = 600
    threads = 32
    # for problem_set in ['gisp', 'indset', 'setcover', 'fcmnf', 'cauctions', 'load_balance', 'item_placement']:
    for problem_set in ['item_placement']:
        params = params_dict[problem_set]
        sense = senses[problem_set]
        policy_root = f'/data/GM4MILP/result_cache/models/{"" if source == "diff" else "gnn_"}{problem_set}/'
        save_folders = np.sort(os.listdir(policy_root))[-1]
        policy_folder = os.path.join(policy_root, save_folders, "checkpoints")
        policy_path = os.path.join(policy_folder, np.sort(os.listdir(policy_folder))[2])
        
        optimizer = PredsearchOptimizer(root_dir, problem_set, policy_path, set_threads = threads, timelimit = Timelimit, guidance = guidance, sense= sense)
        optimizer.run_all(params)
        optimizer.save_all_results(prefix = f'downstream_result/predsearch/{source}_predsearch_solve_{problem_set}{"_guidance" if guidance and source == "diff" else ""}')