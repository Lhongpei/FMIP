import gurobipy as gp
import os
import pickle
import numpy as np
import pandas as pd
from gurobipy import GRB
from tqdm import tqdm
import copy
import time
import concurrent.futures
import torch
from fmip.pl_gmip_model import GMIPModel
from fmip.utils.milp_reader import MIPmodel
from fmip.arg import default_args
from fmip.utils.gp_utils import pred_by_model

time_points_dict = {}
iter_points_dict = {}
obj_values_dict = {}

def save_history(model, where, ind):
    if where == GRB.Callback.MIP:
        # 获取当前运行时间（秒）
        current_time = model.cbGet(GRB.Callback.RUNTIME)

        # Only record every 10 seconds
        if len(time_points_dict[ind]) == 0 or current_time - time_points_dict[ind][-1] >= 1:
            # 获取当前最优解的目标值
            current_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            
            current_iteration = model.cbGet(GRB.Callback.MIP_ITRCNT)
            
                # 记录时间和 Gap
            time_points_dict[ind].append(current_time)
            obj_values_dict[ind].append(current_obj)
            iter_points_dict[ind].append(current_iteration)
    
class NeuralDivingOptimizer:
    def __init__(self, root_dir, problem_set, policy_path,
                 solve_original=False, fixed_strategy='random', set_threads = None, timelimit = None,
                 sample_times = None, fixed_rate = None, guidance = None, sense = "min"):
        self.root_dir = root_dir
        self.sense = sense
        self.load_policy(policy_path)
        self.problem_root = os.path.join(root_dir, problem_set, 'problem')
        self.problem_set = problem_set
        self.result_dict = {}
        self.set_threads = set_threads
        self.timelimit = timelimit
        self.guidance = guidance
        self.solve_original = solve_original
        self.sample_times = sample_times if sample_times else 10
        self.sample_ratio = fixed_rate if fixed_rate else 0.8
        assert fixed_strategy in ['random', 'probs']
        self.solving_history = {}
        
    def load_policy(self, policy_path):
        args = default_args()
        args.max_integer_num = 2
        args.sense = self.sense
        if problem_set in ["load_balance", "item_placement"] and "gnn" not in policy_path:
            args.only_discrete = False
        else:
            args.only_discrete = True

        self.policy = GMIPModel.load_from_checkpoint(policy_path, param_args=args, load_datasets=False)  
        self.policy.eval()
        self.policy.to('cuda')

    def _get_variable_info(self, model):
        """Extract variable information from the model"""
        vars_list = model.getVars()
        
        Ivars_indice = [i for i, var in enumerate(vars_list) 
                        if var.VType in [gp.GRB.INTEGER, gp.GRB.BINARY]]
        Cvars_indice = [i for i, var in enumerate(vars_list) 
                        if var.VType == gp.GRB.CONTINUOUS]
        
        return vars_list, Ivars_indice, Cvars_indice
    
    # def apply_threshould_fixed_strategy(self, path, pred_Ivars_prob, pred_Cvars, Ivars_indice, Cvars_indice, threshold=0.8):
    #     model = gp.read(path)
    #     vars_list = model.getVars()
    #     return self._apply_threshould_fixed_strategy(
    #         model, vars_list, pred_Ivars_prob, pred_Cvars, 
    #         Ivars_indice, Cvars_indice, threshold
    #     )
        
    # def fallback_strategy(self, path, pred_Ivars, pred_Cvars, Ivars_indice, Cvars_indice):
    #     model = gp.read(path)
    #     vars_list = model.getVars()
    #     return self._fallback_strategy(
    #         model, vars_list, pred_Ivars, pred_Cvars, Ivars_indice, Cvars_indice
    #     )
    
    # def solve_original_problem(self, path):
    #     model = gp.read(path)
    #     return self._solve_original_problem(model)
    
    # def _apply_threshould_fixed_strategy(self, model, vars_list, pred_Ivars_prob, pred_Cvars, 
    #                      Ivars_indice, Cvars_indice, threshold):
    #     """Apply warm start with given threshold"""
    #     extreme_row = np.where(np.max(pred_Ivars_prob, axis=1) > threshold)[0]
    #     pred_Ivars = np.argmax(pred_Ivars_prob, axis=1)
        
    #     for i in extreme_row:
    #         vars_list[i].LB = pred_Ivars[i]
    #         vars_list[i].UB = pred_Ivars[i]
            
    #     for ind, i in enumerate(Cvars_indice):
    #         vars_list[i].Start = pred_Cvars[ind]
            
    #     model.update()
    #     model.optimize()
    #     return {
    #         'status': model.Status,
    #         'time': model.Runtime,
    #         'obj': model.ObjVal if model.Status == gp.GRB.OPTIMAL else None
    #     }
        
    # def _fallback_strategy(self, model, vars_list, pred_Ivars, pred_Cvars, Ivars_indice, Cvars_indice):
    #     """Fallback strategy when initial warm start fails"""
    #     for ind, i in enumerate(Ivars_indice):
    #         vars_list[i].Start = pred_Ivars[ind]
            
    #     for ind, i in enumerate(Cvars_indice):
    #         vars_list[i].Start = pred_Cvars[ind]
            
    #     model.update()
    #     model.optimize()
    #     return {
    #             'status': model.Status,
    #             'time': model.Runtime,
    #             'obj': model.ObjVal if model.Status == gp.GRB.OPTIMAL else None
    #         }
    
    # def _solve_original_problem(self, model):
    #     """Solve original problem without warm start"""
    #     if self.set_threads is not None:
    #         model.setParam('Threads', self.set_threads)
    #     if self.timelimit is not None:
    #         model.setParam('Timelimit', self.timelimit)
    #     model.optimize()
    #     return {
    #         'status': model.Status,
    #         'time': model.Runtime,
    #         'obj': model.ObjVal if model.Status == gp.GRB.OPTIMAL else None
    #     }
    
    def apply_sampled_strategy(self, path, pred_Ivars_prob,
                          Ivars_indice, sample_times=10, fixed_ratio=1.):
        """Apply warm start with sampled strategy using parallel solving
        
        Args:
            model: The original Gurobi model
            vars_list: List of all variables in the model
            pred_Ivars_prob: Predicted probabilities for integer variables
            pred_Cvars: Predicted values for continuous variables
            Ivars_indice: Indices of integer variables
            Cvars_indice: Indices of continuous variables
            sample_times: Number of samples to try
            sample_ratio: Ratio of integer variables to fix in each sample
            
        Returns:
            tuple: (best_model, max_solving_time) where:
                best_model: The model with the best solution found (or None if all failed)
                max_solving_time: The maximum optimization time across all samples
        """
        # Generate sample batches
        if fixed_ratio < 1.:
            fixed_ind_set = [np.random.choice(Ivars_indice, 
                                            int(len(Ivars_indice) * fixed_ratio),
                                            replace=False) 
                            for _ in range(sample_times)]
        else:
            fixed_ind_set = [Ivars_indice for _ in range(sample_times)]
            
        pred_Ivars = torch.multinomial(torch.Tensor(pred_Ivars_prob), sample_times, replacement=True).T
        

        # Prepare arguments for parallel processing
        args_list = []
        for i in range(sample_times):
            args_list.append((
                path,
                fixed_ind_set[i],
                pred_Ivars[i],
                i
                # pred_Cvars,
                # Cvars_indice
            ))
        
        # Function to process a single sample
        def _process_sample(args):
            path, sample_batch, pred_Ivars, ind = args
            model = gp.read(path)
            vars_list = model.getVars()

            time_points_dict[ind] = []
            iter_points_dict[ind] = []
            obj_values_dict[ind] = []
            

            for i in sample_batch:
                vars_list[i].LB = pred_Ivars[i]
                vars_list[i].UB = pred_Ivars[i]
            
            # # Set warm start for continuous variables
            # for ind, i in enumerate(Cvars_indice):
            #     vars_list[i].Start = pred_Cvars[ind]
            
            model.update()
            # model.setParam('OutputFlag', 0)  # Suppress output
            if self.set_threads is not None:
                model.setParam('Threads', self.set_threads)
            if self.timelimit is not None:
                model.setParam('Timelimit', self.timelimit)
                
            def tmp_save_history(model, where, ind = ind):
                return save_history(model, where, ind)

            start_time = time.time()
            model.optimize(tmp_save_history)
            solve_time = time.time() - start_time
            
            return model, solve_time
        
        # Track parallel execution
        best_obj = float('inf') if self.sense == "min" else float('-inf')
        best_model = None
        max_solving_time = 0
        best_ind = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_sample, args) for args in args_list]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                sample_model, solve_time = future.result()
                if solve_time > max_solving_time:
                    max_solving_time = solve_time
                if self.sense == "min":
                    if sample_model.Status not in [gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED] and sample_model.ObjVal < best_obj:
                        best_obj = sample_model.ObjVal
                        best_model = sample_model
                        best_ind = i
                elif self.sense == "max":
                    if sample_model.Status not in [gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED] and sample_model.ObjVal > best_obj:
                        best_obj = sample_model.ObjVal
                        best_model = sample_model
                        best_ind = i
                else:
                    raise ValueError("Invalid sense. Use 'min' or 'max'.")
        
        return best_model, max_solving_time, best_ind
    
    def get_prediction(self):
        result = pred_by_model(self.MIPModel, self.policy, self.guidance)
        return result
    
    def process_instance_sampled(self, prob_name):
        result = {}
        
        self.MIPModel = MIPmodel(os.path.join(self.problem_root, prob_name))
        problem_path = os.path.join(self.problem_root, prob_name)
        m_warm = gp.read(problem_path)
        pred_solution = self.get_prediction()
        pred_Ivars_prob = pred_solution['Ivars']
        _, Ivars_indice, _ = self._get_variable_info(m_warm)
        #sampled strategy
        best_model, max_solving_time, best_ind = self.apply_sampled_strategy(
            problem_path, pred_Ivars_prob,
            Ivars_indice, sample_times=self.sample_times, fixed_ratio=self.sample_ratio
        )
        self.solving_history[prob_name] = {}
        self.solving_history[prob_name]['time'] = time_points_dict[best_ind]
        self.solving_history[prob_name]['iter'] = iter_points_dict[best_ind]
        self.solving_history[prob_name]['obj'] = obj_values_dict[best_ind]

        if self.solve_original is not None:
            result['warm_solving_time'] = max_solving_time
            result['warm_obj'] = best_model.ObjVal if best_model is not None else None
            result['warm_iter'] = best_model.IterCount if best_model is not None else None
        # else:
        #     best_model, max_solving_time2 = self.apply_sampled_strategy(
        #         problem_path, pred_Ivars_prob, pred_Cvars,
        #         Ivars_indice, Cvars_indice, sample_times=10, fixed_ratio=0.5
        #     )
        #     result['warm_solving_time'] = max_solving_time + max_solving_time2
        #     result['warm_obj'] = best_model.ObjVal if best_model is not None else None
        return result
       
    def run_all(self):
        """Process all prediction files"""
        with open(f'/data/GM4MILP/dataset_split/{self.problem_set}/split.pkl', 'rb') as f:
            split = pickle.load(f)['test']
        prob_name_list = np.sort(split)[:10]
        for prob_name in tqdm(prob_name_list):
            try:
                self.result_dict[prob_name] = self.process_instance_sampled(prob_name)
            except Exception as e:
                raise e
                print(f"Error processing {pred_name}: {str(e)}")
                self.result_dict[pred_name] = {'error': str(e)}
                
        return pd.DataFrame(self.result_dict).T

    
    # def run_all_pred_origin(self):
    #     """Process all prediction files"""
    #     origin_result_dict = {}
    #     pred_name_list = os.listdir(self.prediction_dir)
    #     for pred_name in tqdm(pred_name_list):
    #         prob_name = pred_name.replace('_pred.pkl', '.mps.gz')
    #         problem_path = os.path.join(self.problem_root, prob_name)
    #         try:
    #             origin_result_dict[prob_name] = self.solve_original_problem(problem_path)
    #         except Exception as e:
    #             print(f"Error processing {prob_name}: {str(e)}")
    #             origin_result_dict[prob_name] = {'error': str(e)}
                
    #     return pd.DataFrame(origin_result_dict).T
    
    def save_results(self, output_file='warmstart_result.csv'):
        """Save results to CSV"""
        df = pd.DataFrame(self.result_dict).T
        df.to_csv(output_file)
        with open(output_file.replace('.csv', '_history.pkl'), 'wb') as f:
            pickle.dump(self.solving_history, f)
        return df

if __name__ == "__main__":
    root_dir = '/data/GM4MILP/instances_final'
    source = "gnn"
    guidance = True
    params = [50, 0.1]
    senses = {
        'cauctions': 'max',
        'gisp': 'max',
        'indset': 'max',
        'fcmnf': 'min',
        'item_placement': 'min',
        'load_balance': 'min',
        'setcover': 'min',
    }
    # for problem_set in ['gisp', 'indset', 'setcover', 'fcmnf', 'cauctions', 'load_balance', 'item_placement']:
    for problem_set in ['item_placement']:
        sense = senses[problem_set]
        policy_root = f'/data/GM4MILP/result_cache/models/{"" if source == "diff" else "gnn_"}{problem_set}/'
        save_folders = np.sort(os.listdir(policy_root))[-1]
        policy_folder = os.path.join(policy_root, save_folders, "checkpoints")
        policy_path = os.path.join(policy_folder, np.sort(os.listdir(policy_folder))[2])
        optimizer = NeuralDivingOptimizer(root_dir, problem_set, policy_path, set_threads=32, timelimit = 200,
                                       sample_times = params[0],
                                       fixed_rate = params[1],
                                       guidance = guidance,
                                       sense = sense)
        df = optimizer.run_all()
        optimizer.save_results(f'./downstream_result/neural_diving/{source}_sample_neural_diving_solve_{problem_set}_{str(params)}{"_guidance" if guidance and source == "diff" else ""}.csv')
        # df = optimizer.run_all_pred_origin()
        # df.to_csv(f'{problem_set}_origin.csv')
        