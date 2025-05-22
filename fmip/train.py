import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from argparse import ArgumentParser
from dataset import GraphMILP
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info
import time
from pl_gmip_model import GMIPModel
from pl_gmip_model import GMIPModel

def arg_parser():
    parser = ArgumentParser(
        description="Train a Pytorch-Lightning diffusion model on a TSP dataset."
    )
    parser.add_argument("--storage_path", type=str, default="./result_cache")
    parser.add_argument("--dataset_root", type=str, default="/data/GM4MILP/datasets")
    parser.add_argument("--dataset_name", type=str, default="load_balancing_tiny")
    parser.add_argument("--dataset_cache", type=str, default="dataset_cache")

    parser.add_argument("--training_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="valid")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--validation_examples", type=int, default=-1) #Note -1 means all examples
    parser.add_argument("--inference_type", type=str, default="batch")

    parser.add_argument(
        "--c_d_weight", type=float, default=1.0, help="Weight for the discrete loss."
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine-decay")

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_activation_checkpoint", action="store_true")

    # About Guidance
    parser.add_argument("--use_guidance", action="store_true")
    parser.add_argument("--num_discrete_sample", type=int, default=10)
    parser.add_argument("--num_continuous_sample", type=int, default=10)
    parser.add_argument("--num_discrete_guide_step", type=int, default=10)
    parser.add_argument("--num_continuous_guide_step", type=int, default=10) 
    
    parser.add_argument("--discrete_flow_mode", type=str, default="uniform")
    parser.add_argument("--flow_schedule", type=str, default="linear")
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--inference_schedule", type=str, default="cosine")
    parser.add_argument("--inference_trick", type=str, default="ddim")
    parser.add_argument("--sequential_sampling", type=int, default=3)
    parser.add_argument("--parallel_sampling", type=int, default=64)

    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--sparse_factor", type=int, default=-1)
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--two_opt_iterations", type=int, default=1000)
    parser.add_argument("--save_numpy_heatmap", action="store_true")

    parser.add_argument("--project_name", type=str, default="GMIP")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_logger_name", type=str, default=None)
    parser.add_argument(
        "--resume_id", type=str, default=time.strftime("%Y-%m-%d_%H-%M-%S"), help="Resume training on wandb."
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume_weight_only", action="store_true")

    parser.add_argument("--select_num_train_instance", type=int, default=None)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_valid_only", action="store_true")

    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--mode", type=str, default="primal")
    parser.add_argument("--only_discrete", action="store_true")
    
    parser.add_argument("--extra_name_wandb", type=str, default="")
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--ce_weight", type=float, nargs='+', default=None, help="List of weights")
    

    args = parser.parse_args()
    return args


def main(args):
    args.do_train = True
    epochs = args.num_epochs
    project_name = args.project_name

    model_class = GMIPModel

    saving_mode = "max"

    model = model_class(param_args=args)
    args.wandb_logger_name = args.wandb_logger_name if args.wandb_logger_name is not None else args.dataset_name
    args.wandb_logger_name = args.wandb_logger_name + args.extra_name_wandb
    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path, f"models"),
        id=args.resume_id or wandb_id,
    )
    rank_zero_info(
        f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/metric",
        mode=saving_mode,  # TODO: change to 'subopt' in the future
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(
            wandb_logger.save_dir,
            args.wandb_logger_name,
            wandb_logger._id,
            "checkpoints",
        ),
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        max_epochs=epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=5,
        strategy=DDPStrategy(static_graph=True),
        precision=16 if args.fp16 else 32
    )
    print(trainer.accelerator)
    # rank_zero_info(f"{'-' * 100}\n" f"{str(model.model)}\n" f"{'-' * 100}\n")

    ckpt_path = args.ckpt_path

    if args.do_train:
        if args.resume_weight_only:
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        if args.do_test:
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    elif args.do_test:
        trainer.validate(model, ckpt_path=ckpt_path)
        if not args.do_valid_only:
            trainer.test(model, ckpt_path=ckpt_path)
        trainer.logger.finalize("success")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
