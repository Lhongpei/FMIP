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
from fmip.utils.lr_schedulers import get_schedule_fn
from fmip.utils.flow_scheduler import DiscreteFlow, ContinuousFlow, InferenceScheduler
from fmip.dataset import SplitSchedule, GraphMILPDataset
from fmip.utils.milp_utils import duplicate_hetero_data
import os
from argparse import Namespace
from fmip.utils.lp_utils import *

class COMetaModel(pl.LightningModule):
    def __init__(self, param_args=None, load_datasets=True):
        super(COMetaModel, self).__init__()
        self.Cvars_dim = 6
        self.Ivars_dim = 6
        self.cons_dim = 1
        if param_args is None:
            raise ValueError("param_args must be supplied, or restored from checkpoint hyperparameters")
        self.args = Namespace(**param_args) if isinstance(param_args, dict) else copy.deepcopy(param_args)
        self.c_d_weight = self.args.c_d_weight
        self.only_discrete = self.args.only_discrete
        if load_datasets:
            self.load_datasets()
            self.max_integer_num = max(
                dataset.max_integer_num for dataset in
                (self.train_dataset, self.validation_dataset, self.test_dataset)
            )
            if self.args.discrete_flow_mode == "mask":
                self.max_integer_num += 1  # Reserve a class that is never a clean label.
        if getattr(self.args, "max_integer_num", None) is not None:
            if load_datasets and self.args.max_integer_num < self.max_integer_num:
                raise ValueError("max_integer_num does not cover the dataset's integer domain and reserved mask")
            self.max_integer_num = self.args.max_integer_num
        if not hasattr(self, "max_integer_num"):
            raise ValueError("max_integer_num is required when load_datasets=False")
        if self.max_integer_num < (2 if self.args.discrete_flow_mode == "mask" else 1):
            raise ValueError("The classifier must contain real classes and, in mask mode, a reserved mask")
        self.args.max_integer_num = self.max_integer_num
        # Restoring a checkpoint for inference must not need the training data.
        self.save_hyperparameters({"param_args": self.args, "load_datasets": False})
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
            os.makedirs(self.save_pred_root, exist_ok=True)
        
        self.num_training_steps_cached = None
        self.inference_steps = self.args.inference_steps
        
        self.inference_scheduler = InferenceScheduler(
            self.inference_steps, self.args.inference_schedule
        )

    def load_datasets(self):
        os.makedirs(self.args.dataset_cache, exist_ok=True)
        file_root = os.path.join(
            self.args.dataset_root, self.args.dataset_name
        )  #'/data/GM4MILP/datasets/load_balancing_tiny'
        saved_path = os.path.join(
            self.args.dataset_cache, self.args.dataset_name
        )  #'dataset_cache/load_balancing_tiny'
        os.makedirs(saved_path, exist_ok=True)
        split = SplitSchedule(file_root, saved_path)
        split_dict = split.get_split()
        train_split = getattr(self.args, "training_split", "train")
        valid_split = getattr(self.args, "validation_split", "valid")
        test_split = getattr(self.args, "test_split", "test")
        for name in (train_split, valid_split, test_split):
            if name not in split_dict:
                raise ValueError(f"Unknown dataset split {name!r}; available splits: {list(split_dict)}")
        train_files = split_dict[train_split]
        limit = getattr(self.args, "select_num_train_instance", None)
        if limit is not None:
            if limit <= 0:
                raise ValueError("select_num_train_instance must be positive")
            train_files = train_files[:limit]
        self.train_dataset = GraphMILPDataset(
            file_root, saved_path, file_name_list=train_files, prefix="train",
        )
        self.problem_root = self.train_dataset.problem_root
        self.solution_root = self.train_dataset.solution_root
        self.validation_dataset = GraphMILPDataset(
            file_root, saved_path, file_name_list=split_dict[valid_split], prefix="valid",
        )
        self.test_dataset = GraphMILPDataset(
            file_root, saved_path, file_name_list=split_dict[test_split], prefix="test",
        )



    def get_total_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        # Lightning accounts for distributed sampling, batch limits, accumulation,
        # and the final partial accumulation step.
        self.num_training_steps_cached = int(self.trainer.estimated_stepping_batches)
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

        return duplicate_hetero_data(heterodata, times).to(device)

    def train_dataloader(self):
        batch_size = self.args.batch_size
        train_dataloader = GraphDataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=self.args.num_workers > 0,
            drop_last=False,
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
                self.validation_dataset,
                range(min(self.args.validation_examples, len(self.validation_dataset)))
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
