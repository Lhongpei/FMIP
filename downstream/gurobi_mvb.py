# from MIPGNN.global_vars import *
# from main_utils import *
from fmip.utils.mvb import *
from gurobipy import *
import pickle
# import coptpy as cp
# from coptpy import COPT
import numpy as np
import time
import argparse
import pandas as pd
import os
import gurobipy as gp
from tqdm import tqdm
from fmip.pl_gmip_model import GMIPModel
from fmip.utils.milp_reader import MIPmodel
from fmip.arg import default_args
from fmip.utils.gp_utils import pred_by_model

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

class MVBOptimizer:
    def __init__(self, root_dir, problem_set, policy_path, args, source, set_threads = None, timelimit = None, guidance = False, sense = "min"):
        self.root_dir = root_dir
        self.sense = sense
        self.load_policy(policy_path)
        self.source = source
        self.problem_root = os.path.join(root_dir, problem_set, 'problem')
        self.result_dict = {}
        self.set_threads = set_threads
        self.timelimit = timelimit
        self.guidance = guidance
        self.args = args
        self.problem_set = problem_set
        self.solving_history = {}

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

    def _get_variable_info(self, model):
        """Extract variable information from the model"""
        vars_list = model.getVars()
        
        Ivars_indice = [i for i, var in enumerate(vars_list) 
                        if var.VType in [gp.GRB.INTEGER, gp.GRB.BINARY]]
        Cvars_indice = [i for i, var in enumerate(vars_list) 
                        if var.VType == gp.GRB.CONTINUOUS]
        
        return vars_list, Ivars_indice, Cvars_indice

    def mvb_experiment(self, initgrbmodel, probs, args):
        m=initgrbmodel.getAttr("NumConstrs")
        n=initgrbmodel.getAttr("NumVars")

        mvbsolver = MVB(m, n)
        mvbsolver.registerModel(initgrbmodel, solver = "gurobi")
        _, Ivars_indice, _ = self._get_variable_info(mvbsolver._model)
        mvbsolver.registerVars(Ivars_indice)
        mvbsolver.setParam(fixratio=args.fixratio, threshold=args.fixthresh,tmvb=[args.fixthresh, args.tmvb],pSuccessLow = [args.psucceed_low],pSuccessUp = [args.psucceed_up])
        mvb_model = mvbsolver.getMultiVarBranch(Xpred=probs,upCut=args.upCut,lowCut=args.lowCut,ratio_involve=args.ratio_involve,ratio=[args.ratio_low,args.ratio_up])
        # mvb_model.setParam("MIPGap", args.gap/2)
        if args.robust:
            mvb_model.setParam("Cuts", 1)
            mvb_model.setParam("MIPFocus", 2)
        if self.set_threads:
            mvb_model.setParam("Threads", self.set_threads)
        if self.timelimit:
            mvb_model.setParam("Timelimit", self.timelimit)

        start_time = time.time()

        mvb_model.optimize(save_history)
        if mvb_model.Status in [gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED]:
            mvbObjVal = None
            mvbIter = None
        else:
            mvbObjVal = mvb_model.getAttr("ObjVal")
            mvbIter = mvb_model.getAttr("IterCount")
        
        results = {}
        results['mvb_solving_time'] = time.time() - start_time
        results['mvb_obj'] = mvbObjVal
        results['mvb_iter'] = mvbIter

        return results

    def get_prediction(self):
        result = pred_by_model(self.MIPModel, self.policy, self.guidance)
        return result
    
    def save_prediction(self, file_name, prediction):
        with open(file_name, 'wb') as f:
            pickle.dump(prediction, f)
        
    
    def process_instance_mvb(self, prob_name):

        self.MIPModel = MIPmodel(os.path.join(self.problem_root, prob_name))
        m_init = gp.read(os.path.join(self.problem_root, prob_name))
        pred_solution = self.get_prediction()
        pred_save_root = f'downstream_result/{self.source}_pred/{self.problem_set}'
        if not os.path.exists(pred_save_root):
            os.makedirs(pred_save_root)
            
        base, ext = os.path.splitext(prob_name)
        print('base', base)
        
        self.save_prediction(os.path.join(pred_save_root, base + '_pred.pkl'), pred_solution)
        pred_Ivars_prob = pred_solution['Ivars']
        
        results = self.mvb_experiment(m_init, pred_Ivars_prob, self.args)

        return results
           
    def run_all(self):
        """Process all prediction files"""
        prob_name_list = os.listdir(self.problem_root)
        with open(f'/data/GM4MILP/dataset_split/{self.problem_set}/split.pkl', 'rb') as f:
            split = pickle.load(f)['test']
        prob_name_list = np.sort(split)[:10]
        for prob_name in tqdm(prob_name_list):
            self.result_dict[prob_name] = self.process_instance_mvb(prob_name)
            self.solving_history[prob_name] = {}
            self.solving_history[prob_name]['time'] = time_points
            self.solving_history[prob_name]['iter'] = iter_points
            self.solving_history[prob_name]['obj'] = obj_values
            time_points.clear()
            iter_points.clear()
            obj_values.clear()
                
        return pd.DataFrame(self.result_dict).T

    def save_results(self, output_file='warmstart_result.csv'):
        df = pd.DataFrame(self.result_dict).T
        df.to_csv(output_file)
        with open(output_file.replace('.csv', '_history.pkl'), 'wb') as f:
            pickle.dump(self.solving_history, f)
        return df

def get_args(robust = 0, fixratio = 0.0, upCut = 1, lowCut = 1, psucceed = 0.9, tmvb = 0.9, ratio_involve = 0,
             ratio_low = 0.8, ratio_up = 0.0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_free", type=int, default=0, help="0: use GNN prediction, 1: use LP relaxation")
    parser.add_argument("--solver", type=str, default='gurobi', choices=['gurobi', 'copt'])
    parser.add_argument("--robust", type=int, default=robust, help="0: primal heuristic experiments, 1: braching rule experiments")
    parser.add_argument("--maxtime", type=float, default=3600.0)
    parser.add_argument("--gap", type=float, default=0.001)
    parser.add_argument("--heuristics", type=float, default=0.05)
    parser.add_argument("--fixthresh", type=float, default=1.1)
    parser.add_argument("--fixratio", type=float, default=fixratio)
    parser.add_argument("--tmvb", type=float, default=tmvb)
    parser.add_argument("--psucceed_low", type=float, default=psucceed)
    parser.add_argument("--psucceed_up", type=float, default=psucceed)
    parser.add_argument("--upCut", type=int, default=upCut)
    parser.add_argument("--lowCut", type=int, default=lowCut)
    parser.add_argument("--ratio_involve", type=int, default=ratio_involve)
    parser.add_argument("--ratio_low", type=float, default=ratio_low)
    parser.add_argument("--ratio_up", type=float, default=ratio_up)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    param = [1, 1, 0.0, 0.7, 0.9, 0, 0.9, 0]
    args = get_args(upCut = param[0],
                    lowCut = param[1],
                    fixratio = param[2],
                    psucceed = param[3],
                    tmvb = param[4],
                    ratio_involve= param[5],
                    ratio_low = param[6],
                    ratio_up = param[7]
                    )
    source = "diff"
    guidance = False
    root_dir = '/data/GM4MILP/instances_final'
    threads = 32
    senses = {
        'cauctions': 'max',
        'gisp': 'max',
        'indset': 'max',
        'fcmnf': 'min',
        'item_placement': 'min',
        'load_balance': 'min',
        'setcover': 'min',
    }
    # for problem_set in os.listdir(root_dir):
    # for problem_set in ['gisp', 'indset', 'setcover', 'fcmnf', 'cauctions', 'load_balance', 'item_placement']:
    for problem_set in ['item_placement']:
        # if os.path.exists(f'downstream_result/mvb/{source}_mvb_solve_{problem_set}_{str(param)}.csv'):
        #     continue
        sense = senses[problem_set]
        policy_root = f'/data/GM4MILP/result_cache/models/{"" if source == "diff" else "gnn_"}{problem_set}/'
        save_folders = np.sort(os.listdir(policy_root))[-1]
        policy_folder = os.path.join(policy_root, save_folders, "checkpoints")
        policy_path = os.path.join(policy_folder, np.sort(os.listdir(policy_folder))[2 if source == "diff" else 3])

        optimizer = MVBOptimizer(root_dir, problem_set, policy_path, args, source, set_threads = threads, timelimit = 600, guidance = guidance, sense = sense)
        df = optimizer.run_all()
        optimizer.save_results(f'downstream_result/mvb/{source}_mvb_solve_{problem_set}_{str(param)}{"_guidance" if guidance and source == "diff" else ""}.csv')