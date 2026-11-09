"""Launch with two Slurm tasks to check changing modality usage under DDP."""
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.loader import DataLoader

from test_data_network import make_graph
from test_training_inference import make_model


def main():
    seed_everything(31)
    model = make_model()
    del model.log
    # DistributedSampler gives each rank an empty-C graph, then a mixed graph.
    dataset = [make_graph(continuous=False), make_graph(continuous=False),
               make_graph(), make_graph()]
    model.train_dataset = dataset
    model.train_dataloader = lambda: DataLoader(dataset, batch_size=1, shuffle=False)
    trainer = Trainer(accelerator='cpu', devices=2,
                      strategy=DDPStrategy(find_unused_parameters=True, process_group_backend='gloo'),
                      max_epochs=1, max_steps=2, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0,
                      limit_val_batches=0)
    trainer.fit(model)
    assert trainer.global_step == 2
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    if trainer.is_global_zero:
        print('DISTRIBUTED_SMOKE_OK: two ranks changed from empty continuous nodes to mixed graphs')


if __name__ == '__main__':
    main()
