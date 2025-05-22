"""Lightning module for training the DIFUSCO PDG model."""

import os
import gurobipy as gp
from pl_meta_model import COMetaModel
from dataset import GraphMILP
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_meta_model import COMetaModel
from utils.lp_utils import *
import pickle
from models.gnn import MILPGCN
from utils.milp_utils import *
import bisect
from utils.focalloss import FocalLoss
from torch.multiprocessing import Pool
import bisect
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,  # PR AUC
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc as sk_auc
)
class GMIPModel(COMetaModel):
    def __init__(self,
               param_args=None, load_datasets=True):

        super(GMIPModel, self).__init__(param_args=param_args, load_datasets=load_datasets)
        self.R = 32
        self.N_iter = 2
        self.init_tau = 1
        self.min_tau = 0.1
        self.model = self.model = MILPGCN(
            input_dim_x_Cvars=self.Cvars_dim,
            input_dim_x_Ivars=self.Ivars_dim,
            input_dim_x_cons=self.cons_dim,
            hidden_dim=self.args.hidden_dim,
            num_layers=self.args.n_layers,
            max_integer_num=self.max_integer_num,
            learn_norm=True,
            only_discrete=self.only_discrete,
            )
        self.Ivars_batch_target = target_func(weight_c=0.002, weight_f=1, if_reduction=False, if_only_graph=False, if_Ivars_batch=True, if_Cvars_batch=False)
        self.Cvars_batch_target = target_func(weight_c=0.002, weight_f=1, if_reduction=True, if_only_graph=False, if_Ivars_batch=False, if_Cvars_batch=False)
        self.target = target_func(weight_c=1, weight_f=1, if_reduction=True, if_only_graph=False, if_Ivars_batch=False, if_Cvars_batch=False)
        self.feasibility_scale = None
        if self.args.ce_weight is not None:
            ce_weight = torch.tensor(self.args.ce_weight).float().to(self.device)
        else:
            ce_weight = None
        self.discrete_loss = nn.CrossEntropyLoss(weight=ce_weight) if not self.args.use_focal_loss else FocalLoss(gamma=1, alpha=0.5)
        if self.args.use_focal_loss:
            print('Using Focal Loss with gamma=2 and alpha=0.5')
            
            
    def forward(self, graph_data:GraphMILP, t):
        """Run the forward pass of the model.

        Args:
            graph_data (GraphMILP): The input graph data.
            t (torch.Tensor): The time step.

        Returns:
            (Ivars_pred, Cvars_v_pred):
                Ivars_pred (torch.Tensor): The predicted integer variables with expected shape: [num_Ivars_nodes, max_integer_num]
                Cvars_v_pred (torch.Tensor): The predicted v for CFM with expected shape: [num_Cvars_nodes]
        """
        return self.model.forward(graph_data, t)
    def get_tau(self):
        current_epoch = self.current_epoch
        total_epochs = self.args.num_epochs
        return self.init_tau * (1 - current_epoch / total_epochs) + self.min_tau * (current_epoch / total_epochs)

    def discrete_training_noise(self, node_labels, point_indicator, t):
        # Sample from diffusion
        t = torch.from_numpy(t).long().to(self.device)
        t = t.repeat_interleave(point_indicator.reshape(-1), dim=0)
        xt = self.DFM.sample(node_labels, t)
        t = t.reshape(-1)
        xt = xt.reshape(-1)
        return xt, t
    
    def continuous_training_noise(self, node_labels, point_indicator, t):
        t = torch.from_numpy(t).long().to(self.device)
        # t = torch.cat([t.repeat_interleave(point_indicator_var.reshape(-1).cpu(), dim=0),
        #                t.repeat_interleave(point_indicator_cons.reshape(-1).cpu(), dim=0)])
        t = t.repeat_interleave(point_indicator.reshape(-1), dim=0)

        xt, dxt = self.CFM.sample(node_labels, t)
        xt = xt.reshape(-1)
        return xt, t, dxt

    def multimodule_training_step(self, batch:GraphMILP, batch_idx):
        graph_data = batch
        
        point_indicator_Cvars = graph_data['Cvars'].ptr[1:] - graph_data['Cvars'].ptr[:-1]
        point_indicator_Ivars = graph_data['Ivars'].ptr[1:] - graph_data['Ivars'].ptr[:-1]

        t = np.random.rand(point_indicator_Cvars.shape[0])
        Cvars_label = graph_data.extract_Cvars_label()
        Ivars_label = graph_data.extract_Ivars_label()
        device = Cvars_label.device
        
        Ivars_t, t_d = self.discrete_training_noise(Ivars_label, point_indicator_Ivars, t)
        Cvars_t, t_c, v = self.continuous_training_noise(Cvars_label, point_indicator_Cvars, t)
        
        graph_data.input_Ivars_label_(Ivars_t)
        graph_data.input_Cvars_label_(Cvars_t)
        
        t_d = t_d.reshape(-1)
        t_c = t_c.reshape(-1)
        v = v.reshape(-1)
        
        x_Ivars_nn, x_Cvars_nn = self.pred_clean_sample(graph_data, torch.cat([t_d, t_c], dim = 0))
        x_Ivars_sample_onehot = torch.nn.functional.gumbel_softmax(x_Ivars_nn, tau=self.get_tau(), hard=True, dim=-1)
        sampled_values_x_Ivars = torch.sum(x_Ivars_sample_onehot * torch.arange(self.max_integer_num, device=device), dim=1)
        graph_data['Cvars'].x = torch.cat([graph_data['Cvars'].x[:, :-1], x_Cvars_nn.view(-1,1)], dim=1)
        graph_data['Ivars'].x = torch.cat([graph_data['Ivars'].x[:, :-1], sampled_values_x_Ivars.view(-1,1)], dim=1)
        # feasibility = cal_feasibility_graph(graph_data, reduction=True)
        # if self.feasibility_scale is None:
        #     self.feasibility_scale = feasibility.item()
        # feasibility = feasibility / (self.feasibility_scale + 1e-6)
        # x_Ivars_nn, x_Cvars_nn = graph_data.projection(x_Ivars_nn, x_Cvars_nn)
        x_Cvars_nn = graph_data.projection_Cvars(x_Cvars_nn)
        loss_c = F.mse_loss(x_Cvars_nn, Cvars_label.float())
        loss_d = self.discrete_loss(x_Ivars_nn, Ivars_label.long())
        loss = loss_c * self.c_d_weight  + loss_d  #+ feasibility *5
 
        self.log("train/loss_c", loss_c)
        self.log("train/loss_d", loss_d)
        # self.log("train/feasibility", feasibility)
        self.log("train/loss", loss)
        return loss
    
    # def sampleNeighbor(self, graph_data:GraphMILP):
        
    
    def pure_discrete_training_step(self, graph_data:GraphMILP, batch_idx):
        point_indicator_Ivars = graph_data['Ivars'].ptr[1:] - graph_data['Ivars'].ptr[:-1]

        t = np.random.rand(point_indicator_Ivars.shape[0])
        Ivars_label = graph_data.extract_Ivars_label()
        device = Ivars_label.device
        
        Ivars_t, t_d = self.discrete_training_noise(Ivars_label, point_indicator_Ivars, t)
        
        graph_data.input_Ivars_label_(Ivars_t)
        
        t_d = t_d.reshape(-1)
        
        x_Ivars_nn, _ = self.pred_clean_sample(graph_data, torch.cat([t_d], dim = 0))
        x_Ivars_sample_onehot = torch.nn.functional.gumbel_softmax(x_Ivars_nn, tau=self.get_tau(), hard=True, dim=-1)
        sampled_values_x_Ivars = torch.sum(x_Ivars_sample_onehot * torch.arange(self.max_integer_num, device=device), dim=1)
        graph_data['Ivars'].x = torch.cat([graph_data['Ivars'].x[:, :-1], sampled_values_x_Ivars.view(-1,1)], dim=1)
        # feasibility = cal_feasibility_graph(graph_data, reduction=True)
        # if self.feasibility_scale is None:
        #     self.feasibility_scale = feasibility.item()
        # feasibility = feasibility / (self.feasibility_scale + 1e-6)
        # x_Ivars_nn, x_Cvars_nn = graph_data.projection(x_Ivars_nn, x_Cvars_nn)

        loss_d = self.discrete_loss(x_Ivars_nn, Ivars_label.long())
        loss = loss_d #+ feasibility *5
 
        self.log("train/loss_c", 0)
        self.log("train/loss_d", loss_d)
        # self.log("train/feasibility", feasibility)
        self.log("train/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.multimodule_training_step(batch, batch_idx)
        
    def discrete_denoise(self, x0, xt, t_start, t_end):
        return self.DFM.next_state(x0, xt, t_start, t_end)
        
    def continuous_denoise(self, v, xt, t_start, t_end):
        return self.CFM.denoise(v, xt, t_start, t_end)
        
    def pred_clean_sample(self, graph_data:GraphMILP, t_start):
        pred_Ivars_0, pred_Cvars_0 = self.forward(
        graph_data, t_start
        )
        return pred_Ivars_0, pred_Cvars_0
        
    def mixed_denoise_step(self, batch: GraphMILP, t_start, t_end):
        with torch.no_grad():
            x_Cvars_t = batch.extract_Cvars_label()
            x_Ivars_t = batch.extract_Ivars_label().long()
            pred_Ivars_0, pred_Cvars_0 = self.pred_clean_sample(batch, t_start)
            pred_Ivars_0 = torch.nn.functional.softmax(pred_Ivars_0, dim=-1)
            pred_Ivars_0 = batch.projection_Ivars(pred_Ivars_0)
            pred_Cvars_0 = batch.projection_Cvars(pred_Cvars_0)
            Ivars_t = self.discrete_denoise(pred_Ivars_0, x_Ivars_t, t_start, t_end)
            Cvars_t = self.continuous_denoise(pred_Cvars_0, x_Cvars_t, t_start, t_end)
            return Ivars_t, Cvars_t
        
    def discrete_denoise_step(self, batch: GraphMILP, t_start, t_end):
        with torch.no_grad():
            x_Ivars_t = batch.extract_Ivars_label().long()
            pred_Ivars_0, _ = self.pred_clean_sample(batch, t_start)
            pred_Ivars_0 = torch.nn.functional.softmax(pred_Ivars_0, dim=-1)
            pred_Ivars_0 = batch.projection_Ivars(pred_Ivars_0)
            Ivars_t = self.discrete_denoise(pred_Ivars_0, x_Ivars_t, t_start, t_end)
            return Ivars_t
    

    @torch.no_grad()#TODO Add Batch support
    def multimodal_guided_denoise_step_TFG(self, single_graph: GraphMILP, t_start, t_end):
        self.temperature = 1
        self.step_size = 0.1
        Ixt = single_graph.extract_Ivars_label().long()
        
        only_discrete = self.only_discrete
        if not only_discrete:
            Cxt = single_graph.extract_Cvars_label()   
        # -----------------Discrete Guidance-----------------
        # (1) No Requirement for gradient
        Ivars_pred, Cvars_pred = self.forward(single_graph, t_start)
        Ivars_pred = torch.nn.functional.softmax(Ivars_pred, dim=-1)
        Ivars_pred_list = torch.multinomial(Ivars_pred, self.R, replacement=True).T
        if not only_discrete:
            Clb = single_graph['Cvars'].x[:, 0]
            Cub = single_graph['Cvars'].x[:, 1]
        
        
        target_value = self.Ivars_batch_target(single_graph, Ivars_pred_list.float(), Cvars_pred)
        normed_target_value = (target_value - target_value.mean()) / (1e-6 + target_value.std())
        if self.args.sense == 'min':
            energy = torch.softmax(-normed_target_value/self.temperature, dim=0)
        elif self.args.sense == 'max':
            energy = torch.softmax(normed_target_value/self.temperature, dim=0)
        else:
            raise ValueError("Sense should be either 'min' or 'max'")
        rate_m_list = self.DFM.rate_matrix(Ivars_pred_list.long(), Ixt, t_start, t_end)
        
        #TODO: check the if the following code is correct
        weighted_rate_m_dt = torch.sum(rate_m_list * energy[:, None, None], dim=0) * (t_end - t_start)
        next_state_probs = torch.scatter(weighted_rate_m_dt, -1, Ixt[:, None], 0.0)
        next_state_probs = torch.scatter(next_state_probs, -1, Ixt[:, None], 1.0 - torch.sum(next_state_probs, dim=-1, keepdim=True)).clamp(min=0.0, max=1.0)
        next_Ivars = torch.multinomial(next_state_probs, 1).reshape(-1)
        
        sample_discrete_idx = torch.multinomial(energy, 1).reshape(-1)
        sample_discrete = Ivars_pred_list[sample_discrete_idx]

        
        if not only_discrete:
            #-----------------Continuous Guidance-----------------
            single_graph.input_Ivars_label_(Ixt)
            with torch.enable_grad():
                for _ in range(self.N_iter):
                    Cxt_copy = Cxt.clone().detach().requires_grad_(True)
                    single_graph['Cvars'].x = torch.cat([single_graph['Cvars'].x[:, :-1], Cxt_copy.view(-1,1)], dim=1)
                    _, Cxt_1 = self.forward(single_graph, t_start)
                    target_value = self.Cvars_batch_target(single_graph, sample_discrete.squeeze(0).float(), Cxt_1)
                    grad = torch.autograd.grad(target_value.sum(), Cxt_copy)[0]
                    grad_ = rescale_grad(grad, clip_scale=1.0)
                    Cxt = torch.clamp(Cxt - self.step_size * grad_, Clb, Cub)
            
            single_graph.input_Ivars_label_(sample_discrete)
            single_graph.input_Cvars_label_(Cxt)
            Cvars_new_pred = self.forward(single_graph, t_start)[1]
            v_pred = (Cvars_new_pred - Cxt) / (1 - t_start)
            next_Cvars = torch.clamp(Cxt + v_pred * (t_end - t_start), Clb, Cub).squeeze()
        else:
            next_Cvars = None
            
        if t_end < 1.0:
            return next_Ivars, next_Cvars
        else:
            return next_state_probs, next_Cvars

    def training_step(self, batch, batch_idx):
        return self.multimodule_training_step(batch, batch_idx) if self.only_discrete == False \
            else self.pure_discrete_training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, draw=False, split='test'):
        return self.inference_parallel(batch, batch_idx, draw=draw, split=split) if self.args.inference_type =='parallel' \
            else self.inference_batch(batch, batch_idx, draw=draw, split=split)

    def validation_step(self, batch:GraphMILP, batch_idx):
        assert self.device != 'cpu'
        return self.test_step(batch, batch_idx, split='val')
    
    def inference_batch(self, batch:GraphMILP, batch_idx, draw=False, split='test'):
        return self.inference_batch_only_discrete(batch, batch_idx, draw=draw, split=split) if self.only_discrete == True \
            else self.inference_batch_multimodal(batch, batch_idx, draw=draw, split=split)
    
    def cal_metrics(self, graph_data:GraphMILP, pred_Ivars, Ivars_solution):
        ce_loss = F.cross_entropy(pred_Ivars, Ivars_solution.long()).item()
        # Calculate AUC (only for binary classification)
        if pred_Ivars.shape[1] == 2:  # binary case
    # Get probabilities for positive class
            pos_probs = pred_Ivars[:, 1].detach().cpu().numpy()
            true_np = Ivars_solution.cpu().numpy()
            
            # Initialize metrics (default to safe values for extreme cases)
            metrics = {
                'roc_auc': 0.5,
                'pr_auc': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
            
            try:
                # ROC AUC 
                metrics['roc_auc'] = roc_auc_score(true_np, pos_probs)
                
                # PR AUC 
                metrics['pr_auc'] = average_precision_score(true_np, pos_probs)
                

                if len(np.unique(true_np)) >= 2:  
                    precision, recall, thresholds = precision_recall_curve(true_np, pos_probs)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                    best_idx = np.nanargmax(f1_scores) 
                    
                    metrics['f1'] = f1_scores[best_idx]
                    metrics['precision'] = precision[best_idx]
                    metrics['recall'] = recall[best_idx]
                    
            except ValueError as e:
                print(f"Metrics calculation failed: {e}")  
            
        else:
            metrics = {'roc_auc': -1, 'pr_auc': -1, 'f1': -1, 'precision': -1, 'recall': -1}
        # Sample from predictions
        
        pred_Ivars_sample = torch.multinomial(pred_Ivars, 1).reshape(-1)
        pred_Ivars_sample = pred_Ivars_sample
        graph_data.input_Ivars_label_(pred_Ivars_sample)
        graph_data.only_discrete = self.only_discrete
        feasibility = cal_feasibility_graph(graph_data, reduction=False).mean()
        metrics = metrics | {'feasibility': feasibility.item(), 'ce_loss': ce_loss}
        pr_auc = metrics['pr_auc']
        metrics = metrics | {'metric': pr_auc}
            
        return metrics
    
    def inference_batch_only_discrete(self, batch: GraphMILP, batch_idx, draw=False, split='test'):
        graph_data = batch
        Ivars_solution = graph_data.extract_Ivars_label()
        shape_Ivars = Ivars_solution.shape
        graph_data.input_Ivars_label_(self.DFM.x_0_sample(shape_Ivars))

        pred_Ivars, _ = self.inference_core(graph_data)
        
        # Convert predictions and solutions to proper shapes
        pred_probs = pred_Ivars  # assuming shape [batch_size, 2] for binary classification
        true_labels = Ivars_solution.long()  # shape [batch_size]
        
        metrics = self.cal_metrics(graph_data, pred_probs, true_labels)
        batchsize = len(graph_data['Ivars'].ptr[1:])
        
        if self.args.save_predictions:
            self.save_labels(pred_probs, [], graph_data, self.save_pred_root)
            
        for key, value in metrics.items():
            self.log(f"{split}/{key}", value, on_epoch=True, sync_dist=True, batch_size=batchsize)

        return metrics

    
    def inference_batch_multimodal(self, batch:GraphMILP, batch_idx, draw=False, split='test'):
        # start_time = time.time()
        graph_data = batch
        # node_labels = torch.cat([graph_data['vars'].x[:, -1], graph_data['cons'].x[:, -1]])
        Ivars_solution = graph_data.extract_Ivars_label()
        Cvars_solution = graph_data.extract_Cvars_label()
        shape_Cvars = Cvars_solution.shape
        shape_Ivars = Ivars_solution.shape
        graph_data.input_Ivars_label_(self.DFM.x_0_sample(shape_Ivars))
        graph_data.input_Cvars_label_(self.CFM.x_0_sample(shape_Cvars))

        pred_Ivars, pred_Cvars = self.inference_core(graph_data)
        # predict_labels_mask = predict_labels_mask
        
        
        metrics = self.cal_metrics(graph_data, pred_Ivars, Ivars_solution.long())
        metrics = metrics | {'Cvar_mse': F.mse_loss(pred_Cvars, Cvars_solution.float()).item()}
        batchsize = len(graph_data['Ivars'].ptr[1:])
        # metrics = metrics | {'cost_time_per_graph': 0}
        if self.args.save_predictions:
            self.save_labels(pred_Ivars, pred_Cvars, graph_data, self.save_pred_root)
            
        for key, value in metrics.items():
            self.log(f"{split}/{key}", value, on_epoch=True, sync_dist=True, batch_size=batchsize)


        return metrics
        
    def save_labels(self, Ivars_solution, Cvars_solution, graph_data, save_root):
        
        if not self.only_discrete:
            Ivars_solution_list = torch.split(Ivars_solution, (graph_data['Ivars'].ptr[1:] - graph_data['Ivars'].ptr[:-1]).cpu().tolist())
            Cvars_solution_list = torch.split(Cvars_solution, (graph_data['Cvars'].ptr[1:] - graph_data['Cvars'].ptr[:-1]).cpu().tolist())
            Ivars_solution_list = [x.cpu().numpy() for x in Ivars_solution_list]
            Cvars_solution_list = [x.cpu().numpy() for x in Cvars_solution_list]
            
            for i, name in enumerate(graph_data['name']):
                dealed_name = name.replace('.mps.gz', '_pred.pkl')
                with open(os.path.join(save_root, dealed_name), 'wb') as f:
                    pickle.dump({'Ivars': Ivars_solution_list[i], 'Cvars': Cvars_solution_list[i]}, f)
        else:
            Ivars_solution_list = torch.split(Ivars_solution, (graph_data['Ivars'].ptr[1:] - graph_data['Ivars'].ptr[:-1]).cpu().tolist())
            Ivars_solution_list = [x.cpu().numpy() for x in Ivars_solution_list]
            
            for i, name in enumerate(graph_data['name']):
                dealed_name = name.replace('.mps.gz', '_pred.pkl')
                with open(os.path.join(save_root, dealed_name), 'wb') as f:
                    pickle.dump({'Ivars': Ivars_solution_list[i]}, f)
        
    def pure_inference_parallel(self, batch, batch_idx, draw=False, split='test'):
        # start_time = time.time()
        graph_data = duplicate_hetero_data(batch, 32)
        # node_labels = torch.cat([graph_data['vars'].x[:, -1], graph_data['cons'].x[:, -1]])
        Ivars_solution = graph_data.extract_Ivars_label()
        Cvars_solution = graph_data.extract_Cvars_label()
        shape_Cvars = Cvars_solution.shape
        shape_Ivars = Ivars_solution.shape
        graph_data.input_Ivars_label_(self.DFM.x_0_sample(shape_Ivars))
        graph_data.input_Cvars_label_(self.CFM.x_0_sample(shape_Cvars))

        pred_Ivars, pred_Cvars = self.inference_core(graph_data)
        # predict_labels_mask = predict_labels_mask
        pred_Ivars_sample = torch.multinomial(pred_Ivars, 1).reshape(-1)
        graph_data.input_Ivars_label_(pred_Ivars_sample)
        graph_data.input_Cvars_label_(pred_Cvars)
        feasibility = cal_feasibility_graph(graph_data, reduction=False)
        print('The feasibility is ', feasibility)
        cost = cal_objective_graph(graph_data, reduction=False)
        print('The cost is ', cost)
        batchsize = len(graph_data['name'])
        
 
        Cvar_mse = F.mse_loss(pred_Cvars, Cvars_solution.float()).item()
        Ivar_ce = F.cross_entropy(pred_Ivars, Ivars_solution.long()).item()
        metrics = {'metric': Cvar_mse + Ivar_ce, 'Cvar_mse': Cvar_mse, 'Ivar_ce': Ivar_ce}
        # metrics = metrics | {'cost_time_per_graph': 0}

        for key, value in metrics.items():
            self.log(f"{split}/{key}", value, on_epoch=True, sync_dist=True, batch_size=batchsize)

        if self.args.save_predictions:
            self.save_labels(pred_Ivars, pred_Cvars, graph_data, self.save_pred_root)

        return metrics

    def inference_parallel(self, batch, batch_idx, draw=False, split='test'):
        with torch.no_grad():
            batchsize = len(batch['Ivars'].ptr[1:])
            assert batchsize == 1
            Ivars_solution = batch.extract_Ivars_label()
            Cvars_solution = batch.extract_Cvars_label()
            shape_Cvars = Cvars_solution.shape
            shape_Ivars = Ivars_solution.shape
            batch.input_Ivars_label_(self.DFM.x_0_sample(shape_Ivars))
            batch.input_Cvars_label_(self.CFM.x_0_sample(shape_Cvars))
            pred_Ivars, pred_Cvars = self.guided_inference_core(batch)
            # predict_labels_mask = predict_labels_mask
            pred_Ivars_sample = torch.multinomial(pred_Ivars, 1).reshape(-1)
            batch.input_Ivars_label_(pred_Ivars_sample)
            batch.input_Cvars_label_(pred_Cvars)
            feasibility = cal_feasibility_graph(batch, reduction=False)
            # print('The feasibility is ', feasibility)
            cost = cal_objective_graph(batch, reduction=False)
            # print('The cost is ', cost)
            batchsize = len(batch['name'])
            
    
            Cvar_mse = F.mse_loss(pred_Cvars, Cvars_solution.float()).item()
            Ivar_ce = F.cross_entropy(pred_Ivars, Ivars_solution.long()).item()
            metrics = {'metric': Cvar_mse + Ivar_ce, 'Cvar_mse': Cvar_mse, 'Ivar_ce': Ivar_ce}
            metrics = metrics | {'feasibility': feasibility[0].item(), 'cost': cost[0].item()}
            # metrics = metrics | {'cost_time_per_graph': 0}
            
            for key, value in metrics.items():
                self.log(f"{split}/{key}", value, on_epoch=True, sync_dist=True, batch_size=batchsize)
                
            if self.args.save_predictions:
                self.save_labels(pred_Ivars, pred_Cvars, batch, self.save_pred_root)

        return metrics
    
    def inference(self, batch:GraphMILP, guidance=False, batchsize=1):
        with torch.no_grad():
            batch.only_discrete = self.only_discrete
            
            if batchsize > 1:
                graph = duplicate_hetero_data(batch, batchsize)
                Ivars_solution = graph.extract_Ivars_label()
                if not self.only_discrete:
                    Cvars_solution = graph.extract_Cvars_label()
                    shape_Cvars = Cvars_solution.shape
                shape_Ivars = Ivars_solution.shape
                graph.input_Ivars_label_(self.DFM.x_0_sample(shape_Ivars))
                if not self.only_discrete:
                    graph.input_Cvars_label_(self.CFM.x_0_sample(shape_Cvars))
                pred_Ivars, pred_Cvars = self.inference_core(graph)
                return {
                    'Ivars': pred_Ivars.reshape(batchsize, -1, 2).cpu().numpy(),
                    'Cvars': pred_Cvars.reshape(batchsize, -1).cpu().numpy() if not self.only_discrete else None,
                }
            
            Ivars_solution = batch.extract_Ivars_label()
            if not self.only_discrete:
                Cvars_solution = batch.extract_Cvars_label()
                shape_Cvars = Cvars_solution.shape
            shape_Ivars = Ivars_solution.shape
            batch.input_Ivars_label_(self.DFM.x_0_sample(shape_Ivars))
            if not self.only_discrete:
                batch.input_Cvars_label_(self.CFM.x_0_sample(shape_Cvars))
            
            if not guidance:
                pred_Ivars, pred_Cvars = self.inference_core(batch)
            else:
                pred_Ivars, pred_Cvars = self.guided_inference_core(batch)
            return {
                'Ivars': pred_Ivars.cpu().numpy(),
                'Cvars': pred_Cvars.cpu().numpy() if not self.only_discrete else None,
            }
            
    
    def inference_core(self, graph_data:GraphMILP):
        steps = self.inference_steps
        # Perform diffusion steps
        if self.only_discrete:
            for i in range(steps):
                t_start, t_end = self.inference_scheduler(i)
                t_start = torch.tensor([t_start], device=self.device)
                t_end = torch.tensor([t_end], device=self.device)
                pred_Ivars = self.discrete_denoise_step(graph_data, t_start, t_end)
                if i != steps - 1:
                    graph_data.input_Ivars_label_(pred_Ivars)
                pred_Cvars = None
        else:    
            for i in range(steps):
                t_start, t_end = self.inference_scheduler(i)
                t_start = torch.tensor([t_start], device=self.device)
                t_end = torch.tensor([t_end], device=self.device)
                pred_Ivars, pred_Cvars = self.mixed_denoise_step(graph_data, t_start, t_end)
                if i != steps - 1:
                    graph_data.input_Ivars_label_(pred_Ivars)
                    graph_data.input_Cvars_label_(pred_Cvars)

        return pred_Ivars, pred_Cvars
    
    @torch.no_grad()
    def guided_inference_core(self, graph_data:GraphMILP):
        steps = self.inference_steps
        # Perform diffusion steps
        for i in range(steps):
            t_start, t_end = self.inference_scheduler(i)
            t_start = torch.tensor([t_start], device=self.device)
            t_end = torch.tensor([t_end], device=self.device)
            pred_Ivars, pred_Cvars = self.multimodal_guided_denoise_step_TFG(graph_data, t_start, t_end)
            if i != steps - 1:
                graph_data.input_Ivars_label_(pred_Ivars)
                if not self.only_discrete:
                    graph_data.input_Cvars_label_(pred_Cvars)

        return pred_Ivars, pred_Cvars