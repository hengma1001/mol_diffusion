import glob
import os
import sys

import joblib
import torch

sys.path.append("../")

from mol_diffusion.preprocessing import dbs_split, dbs_to_torch, xyzs_to_dbs
from mol_diffusion.utils import dict_to_yaml

save_path = "rec_diffusion_qm9"
os.makedirs(save_path, exist_ok=True)
save_path = os.path.abspath(save_path)

lig_xyzs = glob.glob(
    "/lambda_stor/homes/heng.ma/Research/md_pkgs/dataset-qm9/xyz/*.xyz"
)
dbs = xyzs_to_dbs(lig_xyzs, node_attr=True)
dbs, full_voca_size, labelencoder = dbs_to_torch(dbs, scale_factor=1)

input_dict = {}
input_dict["full_voca_size"] = full_voca_size

print(full_voca_size)
le_save = f"{save_path}/labelEncoder.joblib"
joblib.dump(labelencoder, le_save, compress=9)
input_dict["labelencoder"] = le_save


train, val, test = dbs_split(dbs, split_ratio=[0.7, 0.2, 0.1], shuffle=True)

for dataset, name in zip([train, val, test], ["train", "val", "test"]):
    data_save = f"{save_path}/{name}.pth"
    torch.save(dataset, data_save)
    input_dict[name] = data_save

dict_to_yaml(input_dict, f"{save_path}/input.yml")
