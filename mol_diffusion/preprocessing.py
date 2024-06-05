import os
import random

import MDAnalysis as mda
import numpy as np
import torch
import torch_geometric
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def comp_to_dict(
    prot_pdb,
    lig_mol2,
    prot_sel="protein and not name H*",
    # input_feature=None,
    node_attr=None,
) -> dict:
    data = {}

    data["sys_name"] = os.path.basename(os.path.dirname(prot_pdb))
    prot_u = mda.Universe(prot_pdb)
    prot_u.atoms.names = [name.lstrip("0123456789") for name in prot_u.atoms.names]
    protein_noH = prot_u.select_atoms(prot_sel)
    # print(prot_pdb, protein_noH.n_atoms, protein_noH.segments)

    lig_u = mda.Universe(lig_mol2)
    lig_noH = lig_u.select_atoms("not name H*")

    positions = np.concatenate([protein_noH.positions, lig_noH.positions], axis=0)
    positions = positions - np.mean(positions, axis=0)
    data["pos"] = torch.Tensor(positions)
    # if input_feature:
    #     assert len(input_feature) == protein.atoms.n_atoms
    #     data['x'] = input_feature
    if node_attr:
        data["res_atom_name"] = [
            atom.resname + atom.name for atom in protein_noH.atoms
        ] + ["LIG" + elem for elem in lig_noH.atoms.elements]
    return data


def pdbs_to_dbs(comp_paths, **kwargs):
    prot_pdbs, lig_mols = paths_to_pdbs(comp_paths)
    dbs = [
        comp_to_dict(prot, lig, **kwargs)
        for prot, lig in tqdm(zip(prot_pdbs, lig_mols), total=len(prot_pdbs))
    ]
    return dbs


def lig_to_dict(
    lig_mol2,
    node_attr=None,
) -> dict:
    data = {}

    data["sys_name"] = os.path.basename(lig_mol2)[:-4]

    lig_u = mda.Universe(lig_mol2)
    lig_noH = lig_u.atoms

    positions = lig_noH.positions
    positions = positions - np.mean(positions, axis=0)
    data["pos"] = torch.Tensor(positions)
    # if input_feature:
    #     assert len(input_feature) == protein.atoms.n_atoms
    #     data['x'] = input_feature
    if node_attr:
        data["res_atom_name"] = ["LIG" + elem for elem in lig_noH.atoms.elements]
    return data


def ligs_to_dbs(comp_paths, **kwargs):
    prot_pdbs, lig_mols = paths_to_pdbs(comp_paths)
    dbs = [lig_to_dict(lig, **kwargs) for lig in tqdm(lig_mols, total=len(prot_pdbs))]
    return dbs


def xyzs_to_dbs(lig_xyzs, **kwargs):
    dbs = [lig_to_dict(lig, **kwargs) for lig in tqdm(lig_xyzs)]
    return dbs


def position_to_pdb(
    prot_pdb,
    lig_mol2,
    pdb_output,
    positions,
    prot_sel="protein and not name H*",
):
    prot_u = mda.Universe(prot_pdb)
    prot_u.atoms.names = [name.lstrip("0123456789") for name in prot_u.atoms.names]
    protein_noH = prot_u.select_atoms(prot_sel)

    lig_u = mda.Universe(lig_mol2)
    lig_noH = lig_u.select_atoms("not name H*")

    full_u = mda.Merge(protein_noH.atoms, lig_noH.atoms)
    full_u.atoms.positions = positions
    full_u.atoms.write(pdb_output)


def lig_to_pdb(
    lig_mol2,
    pdb_output,
    positions=None,
    select_string=None,
):
    lig_u = mda.Universe(lig_mol2)
    if select_string:
        atoms = lig_u.select_atoms(select_string)
    else:
        atoms = lig_u.atoms
    if positions is not None:
        atoms.positions = positions
    atoms.write(pdb_output)


def dbs_to_torch(dbs, scale_factor=1.0):
    full_voca = np.concatenate([data["res_atom_name"] for data in dbs])
    full_voca_size = len(set(full_voca))
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(full_voca)

    dbs_refined = []
    for data in dbs:
        data = torch_geometric.data.Data(
            pos=data["pos"] * scale_factor,
            z=torch.from_numpy(
                label_encoder.transform(data["res_atom_name"]).reshape(-1, 1)
            ),
            sys_name=data["sys_name"],
        )
        dbs_refined.append(data)
    return dbs_refined, full_voca_size, label_encoder


def path_to_pdb(comp_path):
    label = os.path.basename(comp_path)
    prot_pdb = f"{comp_path}/{label}_protein.pdb"
    lig_mol = f"{comp_path}/{label}_ligand.mol2"

    assert os.path.exists(prot_pdb), f"Missing protein {prot_pdb}"
    assert os.path.exists(lig_mol), f"Missing ligand {lig_mol}"

    return prot_pdb, lig_mol


def paths_to_pdbs(comp_paths):
    prot_pdbs = []
    lig_mols = []
    for comp_path in comp_paths:
        prot_pdb, lig_mol = path_to_pdb(comp_path)

        prot_pdbs.append(prot_pdb)
        lig_mols.append(lig_mol)
    return prot_pdbs, lig_mols


def dbs_split(dbs, split_ratio=[0.7, 0.2, 0.1], shuffle=True, random_seed=0):
    if shuffle:
        random.seed(random_seed)
        random.shuffle(dbs)
    train, val_test = train_test_split(dbs, train_size=int(split_ratio[0] * len(dbs)))
    val, test = train_test_split(val_test, train_size=int(split_ratio[1] * len(dbs)))

    train = torch_geometric.loader.DataLoader(train, batch_size=1, shuffle=shuffle)
    val = torch_geometric.loader.DataLoader(val, batch_size=1, shuffle=shuffle)
    test = torch_geometric.loader.DataLoader(test, batch_size=1, shuffle=shuffle)

    return train, val, test


# def pdbs_to_datasets(comp_paths, split_ratio=[0.7, 0.2, 0.1], **kwargs):
#     dbs, full_voca_size = pdbs_to_dbs(comp_paths, **kwargs)
#     train, val_test = train_test_split(dbs, train_size=int(split_ratio[0] * len(dbs)))
#     val, test = train_test_split(val_test, train_size=int(split_ratio[1] * len(dbs)))

#     train = torch_geometric.loader.DataLoader(train, batch_size=1, shuffle=True)
#     val = torch_geometric.loader.DataLoader(val, batch_size=1, shuffle=True)
#     test = torch_geometric.loader.DataLoader(test, batch_size=1, shuffle=True)

#     return train, val, test, full_voca_size


# feat = torch.from_numpy(features)  # convert to pytorch tensors
# ys = torch.from_numpy(labels)  # convert to pytorch tensors
# traj_data = []
# distances = ys - feat  # compute distances to next frame


# # make torch_geometric dataset
# # we want this to be an iterable list
# # x = None because we have no input features
# for frame, label in zip(feat, distances):
#     traj_data += [
#         torch_geometric.data.Data(
#             x=None, pos=frame.to(torch.float32), y=label.to(torch.float32)
#         )
#     ]

# train_split = 1637
# train_loader = torch_geometric.loader.DataLoader(
#     traj_data[:train_split], batch_size=1, shuffle=False
# )

# test_loader = torch_geometric.loader.DataLoader(
#     traj_data[train_split:], batch_size=1, shuffle=False
# )


def attr_to_onehot(res_atom_name):
    pass
