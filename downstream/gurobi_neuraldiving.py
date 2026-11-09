import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from fmip.utils.arg import default_args
from fmip.utils.gp_utils import pred_by_model, load_policy_checkpoint
from fmip.utils.solver_validation import load_test_instances, history_path

time_points_dict = {}
iter_points_dict = {}
obj_values_dict = {}

def save_history(model, where, ind):
    if where == GRB.Callback.MIP:
        current_time = model.cbGet(GRB.Callback.RUNTIME)
        if len(time_points_dict[ind]) == 0 or current_time - time_points_dict[ind][-1] >= 1:

            current_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            current_iteration = model.cbGet(GRB.Callback.MIP_ITRCNT)
            
            time_points_dict[ind].append(current_time)
            obj_values_dict[ind].append(current_obj)
            iter_points_dict[ind].append(current_iteration)
    
class NeuralDivingOptimizer:
    def __init__(self, root_dir, problem_set, policy_path,
                 solve_original=False, fixed_strategy='random', set_threads = None, timelimit = None,
                 sample_times = None, fixed_rate = None, guidance = None, sense = "min",
                 split_path=None, instance_limit=None):
        self.root_dir = root_dir
        self.sense = sense
        self.problem_set = problem_set
        self.load_policy(policy_path)
        self.problem_root = os.path.join(root_dir, problem_set, 'problem')
        self.split_path = split_path
        self.instance_limit = instance_limit
        self.problem_set = problem_set
        self.result_dict = {}
        self.set_threads = set_threads
        self.timelimit = timelimit
        self.guidance = guidance
        self.solve_original = solve_original
        self.sample_times = sample_times if sample_times is not None else 10
        self.sample_ratio = fixed_rate if fixed_rate is not None else 0.8
        assert fixed_strategy in ['random', 'probs']
        self.fixed_strategy = fixed_strategy
        self.solving_history = {}
        
    def load_policy(self, policy_path):
        self.policy = load_policy_checkpoint(policy_path, sense=self.sense)

    def _get_variable_info(self, model):
        """Extract variable information from the model"""
        vars_list = model.getVars()
        
        Ivars_indice = [i for i, var in enumerate(vars_list) 
                        if var.VType in [gp.GRB.INTEGER, gp.GRB.BINARY]]
        Cvars_indice = [i for i, var in enumerate(vars_list) 
                        if var.VType == gp.GRB.CONTINUOUS]
        
        return vars_list, Ivars_indice, Cvars_indice
    

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
            tuple: (best_model, solving_time, best_index) where:
                best_model: The model with the best solution found (or None if all failed)
                solving_time: Wall time for the complete parallel search
                best_index: Index of the sample with the best solution
        """
        # Generate sample batches
        if not 0 <= fixed_ratio <= 1 or sample_times < 1:
            raise ValueError("fixed_ratio must be in [0, 1] and sample_times must be positive")
        pred_Ivars_prob = np.asarray(pred_Ivars_prob, dtype=float)
        if (pred_Ivars_prob.ndim != 2 or pred_Ivars_prob.shape[0] != len(Ivars_indice)
                or pred_Ivars_prob.shape[1] == 0):
            raise ValueError("Expected one categorical probability row per integer variable")
        if not np.isfinite(pred_Ivars_prob).all() or (pred_Ivars_prob < 0).any():
            raise ValueError("Predicted probabilities must be finite and nonnegative")
        row_mass = pred_Ivars_prob.sum(axis=1, keepdims=True)
        if not np.isfinite(row_mass).all() or (row_mass <= 0).any():
            raise ValueError("Each integer variable needs positive finite probability mass")
        pred_Ivars_prob = pred_Ivars_prob / row_mass
        if getattr(self, 'fixed_strategy', 'random') == 'probs':
            fixed = np.argsort(-pred_Ivars_prob.max(axis=1), kind='stable')[:int(len(Ivars_indice) * fixed_ratio)]
            fixed_ind_set = [fixed for _ in range(sample_times)]
        elif fixed_ratio < 1.:
            fixed_ind_set = [np.random.choice(len(Ivars_indice),
                                            int(len(Ivars_indice) * fixed_ratio),
                                            replace=False) 
                            for _ in range(sample_times)]
        else:
            fixed_ind_set = [range(len(Ivars_indice)) for _ in range(sample_times)]
            
        pred_Ivars = torch.multinomial(torch.Tensor(pred_Ivars_prob), sample_times, replacement=True).T
        

        # Prepare arguments for parallel processing
        args_list = []
        for i in range(sample_times):
            args_list.append((
                path,
                fixed_ind_set[i],
                pred_Ivars[i],
                i
            ))
        
        # Function to process a single sample
        def _process_sample(args, environment):
            path, sample_batch, pred_Ivars, ind = args
            model = gp.read(path, env=environment)
            try:
                vars_list = model.getVars()
                time_points_dict[ind] = []
                iter_points_dict[ind] = []
                obj_values_dict[ind] = []

                for i in sample_batch:
                    variable = vars_list[Ivars_indice[i]]
                    value = pred_Ivars[i].item()
                    if variable.VType not in (gp.GRB.INTEGER, gp.GRB.BINARY) or not variable.LB <= value <= variable.UB:
                        raise ValueError("Sampled integer value lies outside the original variable domain")
                    variable.LB = value
                    variable.UB = value

                model.update()
                model.setParam('Threads', solver_threads)
                if self.timelimit is not None:
                    model.setParam('Timelimit', self.timelimit)

                def tmp_save_history(model, where, ind = ind):
                    return save_history(model, where, ind)

                model.optimize(tmp_save_history)
            except Exception:
                model.dispose()
                raise
            return model, ind
        
        # Track parallel execution
        if self.sense not in ('min', 'max'):
            raise ValueError("Invalid sense. Use 'min' or 'max'.")
        best_model = None
        best_ind = 0
        def minimizes(model):
            model_sense = getattr(model, 'ModelSense', None)
            if model_sense == gp.GRB.MINIMIZE:
                return True
            if model_sense == gp.GRB.MAXIMIZE:
                return False
            return self.sense == 'min'
        # Bound worker concurrency by the allocation, including solver threads.
        allocated_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
        solver_threads = self.set_threads if self.set_threads is not None else 1
        workers = min(sample_times, max(1, allocated_cpus // max(1, solver_threads)))
        # Gurobi environments must be created on the main thread and exclusively
        # owned by one worker. Keep samples sequential within each environment.
        def _process_group(task):
            samples, environment = task
            group_best, group_index = None, 0
            candidate = None
            try:
                for sample in samples:
                    candidate, index = _process_sample(sample, environment)
                    better = candidate.SolCount > 0 and (group_best is None or
                        (candidate.ObjVal < group_best.ObjVal if minimizes(candidate)
                         else candidate.ObjVal > group_best.ObjVal))
                    if better:
                        if group_best is not None:
                            group_best.dispose()
                        group_best, group_index = candidate, index
                    else:
                        candidate.dispose()
                    candidate = None
            except Exception:
                if candidate is not None and candidate is not group_best:
                    candidate.dispose()
                if group_best is not None:
                    group_best.dispose()
                raise
            return group_best, group_index, environment

        environments = []
        kept_environment = None
        search_start = time.time()
        try:
            for _ in range(workers):
                environments.append(gp.Env(params={'Threads': solver_threads}))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                failure = None
                for i, environment in enumerate(environments):
                    try:
                        futures.append(executor.submit(_process_group, (args_list[i::workers], environment)))
                    except Exception as error:
                        failure = error
                        break
                for future in concurrent.futures.as_completed(futures):
                    try:
                        sample_model, sample_ind, environment = future.result()
                    except Exception as error:
                        if failure is None:
                            failure = error
                        continue
                    # Collect every worker before disposing its environment,
                    # including successful workers after another worker fails.
                    if failure is not None:
                        if sample_model is not None:
                            sample_model.dispose()
                        continue
                    better = sample_model is not None and (best_model is None or
                        (sample_model.ObjVal < best_model.ObjVal if minimizes(sample_model)
                         else sample_model.ObjVal > best_model.ObjVal))
                    if better:
                        if best_model is not None:
                            best_model.dispose()
                        best_model, best_ind = sample_model, sample_ind
                        kept_environment = environment
                    elif sample_model is not None:
                        sample_model.dispose()
                if failure is not None:
                    raise failure
            if best_model is not None:
                # Preserve the winning model's environment for its remaining lifetime.
                best_model._fmip_environment = kept_environment
        except Exception:
            if best_model is not None:
                best_model.dispose()
            kept_environment = None
            raise
        finally:
            for environment in environments:
                if environment is not kept_environment:
                    environment.dispose()
        
        return best_model, time.time() - search_start, best_ind
    
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

        result['warm_solving_time'] = max_solving_time
        result['warm_obj'] = best_model.ObjVal if best_model is not None else None
        result['warm_iter'] = best_model.IterCount if best_model is not None else None
        if self.solve_original:
            if self.set_threads is not None:
                m_warm.setParam('Threads', self.set_threads)
            if self.timelimit is not None:
                m_warm.setParam('TimeLimit', self.timelimit)
            start = time.time()
            m_warm.optimize()
            result['original_solving_time'] = time.time() - start
            result['original_obj'] = m_warm.ObjVal if m_warm.SolCount else None
        m_warm.dispose()
        if best_model is not None:
            environment = getattr(best_model, '_fmip_environment', None)
            best_model.dispose()
            if environment is not None:
                environment.dispose()
        return result
       
    def run_all(self):
        """Process all prediction files"""
        prob_name_list = load_test_instances(
            self.problem_root, self.split_path,
            getattr(self.policy.args, 'dataset_cache', None), self.instance_limit)
        for prob_name in tqdm(prob_name_list):
            self.result_dict[prob_name] = self.process_instance_sampled(prob_name)
                
        return pd.DataFrame(self.result_dict).T

    
    def save_results(self, output_file='warmstart_result.csv'):
        """Save results to CSV"""
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        df = pd.DataFrame(self.result_dict).T
        df.to_csv(output_file)
        with open(history_path(output_file), 'wb') as f:
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
                                       sense = sense,
                                       split_path=f'/data/GM4MILP/dataset_split/{problem_set}/split.pkl', instance_limit=10)
        df = optimizer.run_all()
        optimizer.save_results(f'./downstream_result/neural_diving/{source}_sample_neural_diving_solve_{problem_set}_{str(params)}{"_guidance" if guidance and source == "diff" else ""}.csv')
        # df = optimizer.run_all_pred_origin()
        # df.to_csv(f'{problem_set}_origin.csv')
