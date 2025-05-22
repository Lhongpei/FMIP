"""A meta PyTorch Lightning model for training and evaluating DIFUSCO models."""

import copy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.data import Batch
from pytorch_lightning.utilities import rank_zero_info
from utils.lr_schedulers import get_schedule_fn
from utils.flow_scheduler import DiscreteFlow, ContinuousFlow, InferenceScheduler
from dataset import SplitSchedule, GraphMILPDataset
import os
from utils.lp_utils import *

class COMetaModel(pl.LightningModule):
    def __init__(self, param_args, load_datasets=True):
        super(COMetaModel, self).__init__()
        self.Cvars_dim = 6
        self.Ivars_dim = 6
        self.cons_dim = 1
        self.args = param_args
        self.c_d_weight = self.args.c_d_weight
        self.only_discrete = self.args.only_discrete
        if load_datasets:
            self.load_datasets()
            self.max_integer_num = self.test_dataset.max_integer_num
            print("Number of classes:", self.max_integer_num)
        if hasattr(self.args, "max_integer_num"):
            self.max_integer_num = self.args.max_integer_num
        self.DFM = DiscreteFlow(
            max_integer_num=self.max_integer_num,
            mode=self.args.discrete_flow_mode,
           # dt=self.args.d_dt,
        )
        self.CFM = ContinuousFlow(
            scheduler=self.args.flow_schedule, #dt=self.args.c_dt
        )
        
        if self.args.save_predictions:
            self.save_pred_root = os.path.join(self.args.storage_path, "predictions", self.args.dataset_name)
            if not os.path.exists(self.save_pred_root):
                os.makedirs(self.save_pred_root)
        
        self.num_training_steps_cached = None
        self.inference_steps = self.args.inference_steps
        
        self.inference_scheduler = InferenceScheduler(
            self.inference_steps, self.args.inference_schedule
        )

    def load_datasets(self):
        if not os.path.exists(self.args.dataset_cache):
            os.makedirs(self.args.dataset_cache)
        file_root = os.path.join(
            self.args.dataset_root, self.args.dataset_name
        )  #'/data/GM4MILP/datasets/load_balancing_tiny'
        saved_path = os.path.join(
            self.args.dataset_cache, self.args.dataset_name
        )  #'dataset_cache/load_balancing_tiny'
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        split = SplitSchedule(file_root, saved_path)
        split_dict = split.get_split()
        self.train_dataset = GraphMILPDataset(
            file_root, saved_path, file_name_list=split_dict["train"], prefix="train",
        )
        self.problem_root = self.train_dataset.problem_root
        self.solution_root = self.train_dataset.solution_root
        self.validation_dataset = GraphMILPDataset(
            file_root, saved_path, file_name_list=split_dict["valid"], prefix="valid",
        )
        self.test_dataset = GraphMILPDataset(
            file_root, saved_path, file_name_list=split_dict["test"], prefix="test",
        )

    # def on_test_epoch_end(self):
    #       # 将收集的批次结果合并
    #       unmerged_metrics = {}
    #       for metrics in self.test_outputs:
    #           for k, v in metrics.items():
    #               if k not in unmerged_metrics:
    #                   unmerged_metrics[k] = []
    #               unmerged_metrics[k].append(v)

    #       # 计算每个指标的平均值
    #       merged_metrics = {}
    #       for k, v in unmerged_metrics.items():
    #           merged_metrics[k] = float(np.mean(v))

    #       # 记录到日志
    #       self.logger.log_metrics(merged_metrics, step=self.global_step)
    #       print(f"Test Metrics: {merged_metrics}")

    #       # 清空 `test_outputs` 以便复用
    #       self.test_outputs = []

    def get_total_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps
      
        dataset_size = (
            self.trainer.limit_train_batches * len(dataset)
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        self.num_training_steps_cached = (
            dataset_size // effective_batch_size
        ) * self.trainer.max_epochs
        return self.num_training_steps_cached

    def configure_optimizers(self):
        rank_zero_info(
            "Parameters: %d" % sum([p.numel() for p in self.model.parameters()])
        )
        rank_zero_info("Training steps: %d" % self.get_total_num_training_steps())

        if self.args.lr_scheduler == "constant":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            scheduler = get_schedule_fn(
                self.args.lr_scheduler, self.get_total_num_training_steps()
            )(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

    def duplicate_hetero_data(self, heterodata, times, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""

        return Batch.from_data_list(
            [copy.deepcopy(heterodata) for _ in range(times)]
        )

    def train_dataloader(self):
        batch_size = self.args.batch_size
        train_dataloader = GraphDataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return train_dataloader

    def test_dataloader(self):
        batch_size = (
            1 if self.args.inference_type == "parallel" else self.args.batch_size
        )
        print("Test dataset size:", len(self.test_dataset))
        test_dataloader = GraphDataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )
        return test_dataloader

    def val_dataloader(self):
        batch_size = (
            1 if self.args.inference_type == "parallel" else self.args.batch_size
        )
        if self.args.validation_examples > 0:
            val_dataset = torch.utils.data.Subset(
                self.validation_dataset, range(self.args.validation_examples)
            )
        else:
            val_dataset = self.validation_dataset
        print("Validation dataset size:", len(val_dataset))
        val_dataloader = GraphDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        return val_dataloader
