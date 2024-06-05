import glob
import os
import sys

import joblib
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

sys.path.append("../")

from mol_diffusion.model import e3_diffusion

# from mol_diffusion.preprocessing import dbs_split, dbs_to_torch, xyzs_to_dbs
from mol_diffusion.training import train_diffu
from mol_diffusion.utils import dict_from_yaml

wandb_logger = WandbLogger(project="receptor-diffusion", log_model="all")

time_step = 200
scheduler = "quadratic_beta_schedule"
time_emb_dim = 32
node_attr_emb_dim = 20
# radius_decay = time_step * 0.8


save_path = "rec_diffusion_qm9"
input_dict = f"{save_path}/input.yml"
input_dict = dict_from_yaml(input_dict)

full_voca_size = input_dict["full_voca_size"]

for db_name in ["train", "val", "test"]:
    exec(f"{db_name} = torch.load('{input_dict[db_name]}')")

model_kwargs = {
    "irreps_in": f"{time_emb_dim}x0e",  # no input features
    "irreps_hidden": "64x0e + 64x0o + 32x1e + 32x1o + 8x2e + 8x2o",  # hyperparameter
    "irreps_out": "1e",  # 12 vectors out, but only 1 vector out per input
    "irreps_node_attr": f"{node_attr_emb_dim}x0e",
    "irreps_edge_attr": 3,
    "layers": 6,  # hyperparameter
    "max_radius": 3.5,
    "number_of_basis": 10,
    "radial_layers": 1,
    "radial_neurons": 128,
    "num_neighbors": 11,  # average number of neighbors w/in max_radius
    "num_nodes": 12,  # not important unless reduce_output is True
    "node_attr_n_kind": full_voca_size,
    "node_attr_emb_dim": node_attr_emb_dim,
    "time_emb_dim": time_emb_dim,
    # "radius_decay": radius_decay,
    "reduce_output": False,  # setting this to true would give us one scalar as an output.
}

model = e3_diffusion(time_step, scheduler, **model_kwargs)

model, result = train_diffu(
    model,
    save_path,
    train,
    val,
    test,
    n_gpus=6,
    max_epochs=500,
    every_n_epochs=10,
    logger=wandb_logger,
)
