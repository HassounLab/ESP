import torch
import pickle
import os
import pandas as pd
import numpy as np
import dgllife.utils as chemutils
from torch.utils.data import Dataset
from rdkit.Chem import AllChem
from tqdm import tqdm

data_dir_prefix = './'

hp = {
    # data restriction (not change)
    "pos_prec": ['[M+H]+', '[M+H-H2O]+', '[M+H-2H2O]+', '[M+H-NH3]+', '[M+Na]+', '[M+H+2i]+'],
    "neg_prec": ['[M-H]-', '[M-H-H2O]-', '[M-H-CO2]-'],
    "element_list": "chnopsh",
    "data_dir": 'final_5',
    "mode": 'positive',
    "atom_feature": 'medium',
    "bond_feature": 'light',
    "ms_transformation": 'log10over3',
    "max_mz": 1000,
    "instrument_on_node": True,
    "self_loop": True,
    "num_virtual_nodes": 0,
    "fp_size": 4096,
    "noise": False,

    # bin size (change)
    "resolution": 1}


class msgnnDataset(Dataset):
    def __init__(self, data_list, noise):
        self.data_list = data_list
        self.noise = noise

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        i, pmz, g, setting_tensor, ms, fp = self.data_list[idx]
        return g, setting_tensor, ms, pmz, fp


def get_atom_featurizer(feature_mode, element_list):
    atom_mass_fun = chemutils.ConcatFeaturizer([chemutils.atom_mass])
    def atom_type_one_hot(atom):
        return chemutils.atom_type_one_hot(atom, allowable_set=element_list, encode_unknown=True)
    if feature_mode == 'medium':
        atom_featurizer_funs = chemutils.ConcatFeaturizer([
            chemutils.atom_mass,
            atom_type_one_hot,
            chemutils.atom_total_degree_one_hot,
            chemutils.atom_total_num_H_one_hot,
            chemutils.atom_is_aromatic_one_hot,
            chemutils.atom_is_in_ring_one_hot])
    return chemutils.BaseAtomFeaturizer({"h": atom_featurizer_funs, "m": atom_mass_fun})


def get_bond_featurizer(feature_mode, self_loop):
    if feature_mode == 'light':
        return chemutils.BaseBondFeaturizer(featurizer_funcs={'e': chemutils.ConcatFeaturizer([chemutils.bond_type_one_hot])}, self_loop=self_loop)


def get_ms_setting_all_nodes(precursor_type, ce, n_nodes, prec_pool):
    out = torch.zeros((n_nodes, len(prec_pool) + 1))
    out[:, prec_pool.index(precursor_type)] = 1.0
    out[:, -1] = ce
    return out


def get_ms_setting(precursor_type, ce, prec_pool):
    out = np.zeros(len(prec_pool) + 1)
    out[prec_pool.index(precursor_type)] = 1.0
    out[-1] = ce
    return out


def get_intensity_(x):
    return list(map(float, x.split(' ', 2)[0:2]))


def get_intensity(x):
    x_list = list(map(get_intensity_, x.split('\n')[:-1]))
    return np.array(x_list, dtype = float)


def get_ms_array(x, transformation, max_mz, resolution):
    mz_intensity = get_intensity(x)
    n_cells = int(max_mz / resolution)
    ms_array = np.zeros(n_cells, np.float32)
    mz_intensity = [p for p in mz_intensity if p[0] < max_mz + 1]
    for p in mz_intensity:
        bin_idx = int((p[0] - 1) / resolution)
        ms_array[bin_idx] += p[1]
    if transformation == "log10over3":
        out = np.log10(ms_array + 1) / 3
    return out


def mod_instrg_df2list(df, mol_dict, atom_feature, element_list, edge_feature, self_loop, num_virtual_nodes, prec_types):
    if element_list == "chnopsh":
        ele_list = ['H', 'C',  'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I']

    instru_g_tensor = {}
    for index, rw in df.iterrows():
        g = chemutils.mol_to_bigraph(
                mol_dict[rw['InChIKey']],
                node_featurizer = get_atom_featurizer(atom_feature, ele_list),
                edge_featurizer=get_bond_featurizer(edge_feature, self_loop),
                add_self_loop = self_loop,
                num_virtual_nodes = num_virtual_nodes
        )
        setting_tensor_on_nodes = get_ms_setting_all_nodes(rw['Precursor_type'], rw['NCE'], g.num_nodes(), prec_types)
        instru_g_tensor[rw['InChIKey']] = setting_tensor_on_nodes
    return setting_tensor_on_nodes


def df2list(df, trans, max_mz, resolution, mol_dict, atom_feature, element_list, edge_feature, self_loop, num_virtual_nodes, prec_types, instrument_on_node, fp_size):
    if element_list == "chnopsh":
        ele_list = ['H', 'C',  'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I']

    data_list = []
    for index, rw in df.iterrows():
        setting_tensor = get_ms_setting(rw['Precursor_type'], rw['NCE'], prec_types)
        fp = np.array([int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol_dict[rw['InChIKey']], radius = 2, nBits = fp_size).ToBitString()])

        g = chemutils.mol_to_bigraph(
                mol_dict[rw['InChIKey']],
                node_featurizer = get_atom_featurizer(atom_feature, ele_list),
                edge_featurizer=get_bond_featurizer(edge_feature, self_loop),
                add_self_loop = self_loop,
                num_virtual_nodes = num_virtual_nodes
        )
        if instrument_on_node:
            setting_tensor_on_nodes = get_ms_setting_all_nodes(rw['Precursor_type'], rw['NCE'], g.num_nodes(), prec_types)
            g.ndata['h'] = torch.cat((g.ndata['h'], setting_tensor_on_nodes), -1)
        data_list.append((rw['InChIKey'], int(rw['PrecursorMZ']/resolution), g, setting_tensor, get_ms_array(rw['ms'], trans, max_mz, resolution), fp))
    return data_list


def split_train_val(df, trans, max_mz, resolution, mol_dict, atom_feature, element_list, edge_feature, self_loop, num_virtual_nodes, prec_types, instrument_on_node, fp_size, ratio=0.9):
    all_ik = pd.unique(df.InChIKey)
    ik_msk = np.random.random(all_ik.shape)
    train_ik = [ik for ik, msk in zip(all_ik, ik_msk) if msk < ratio]
    val_ik = [ik for ik, msk in zip(all_ik, ik_msk) if msk >= ratio]
    train_df = df[df.InChIKey.isin(train_ik)].reset_index(drop=True)
    val_df = df[df.InChIKey.isin(val_ik)].reset_index(drop=True)

    train_list = df2list(train_df, trans, max_mz, resolution, mol_dict, atom_feature, element_list, edge_feature, self_loop, num_virtual_nodes, prec_types, instrument_on_node, fp_size)
    val_list = df2list(val_df, trans, max_mz, resolution, mol_dict, atom_feature, element_list, edge_feature, self_loop, num_virtual_nodes, prec_types,instrument_on_node, fp_size)
    return train_list, val_list


def create_train_val_dataset(mode, data, precs, atom_feature, edge_feature, ms_transformation, max_mz, resolution,instrument_on_node, self_loop, num_virtual_nodes, element_list, fp_size, noise):
    mol_dict, pos_train, neg_train = data
    pos_prec, neg_prec = precs
    if mode == 'positive':
        prec_types = pos_prec
        train_set, val_set = split_train_val(pos_train, ms_transformation, max_mz, resolution, mol_dict, atom_feature, element_list, edge_feature, self_loop, num_virtual_nodes,prec_types, instrument_on_node, fp_size, ratio=0.9)
    train_ds = msgnnDataset(train_set, noise)
    val_ds = msgnnDataset(val_set, noise)
    return train_ds, val_ds


def create_test_dataset(mode, data, precs, atom_feature, edge_feature, ms_transformation, max_mz, resolution, instrument_on_node, self_loop, num_virtual_nodes, element_list, fp_size, noise):
    mol_dict, pos_test, neg_test = data
    pos_prec, neg_prec = precs
    if mode == 'positive':
        prec_types = pos_prec
        test_set = df2list(pos_test, ms_transformation, max_mz, resolution, mol_dict, atom_feature, element_list, edge_feature, self_loop, num_virtual_nodes, prec_types, instrument_on_node, fp_size)
    test_ds = msgnnDataset(test_set, noise)
    return test_ds


def convert_graph_dgl_to_torchgeometric(ds):
    for i in tqdm(range(len(ds))):
        ds[i] = list(ds[i])
        graph = ds[i][2]
        g_nodes = graph.nodes().numpy()
        g_edges = [x.numpy() for x in graph.edges()]
        g_edges = np.vstack(g_edges)
        g_edges_f = graph.edata['e'].numpy()
        g_nodes_f = graph.ndata['h'].numpy()[g_nodes]
        ds[i][2] = [g_edges, g_nodes_f, g_edges_f]

    with open("./data/torch_trvate_"+str(int(1000/hp['resolution']))+"bin.pkl", 'wb') as fp:
        pickle.dump(ds, fp, protocol=4)


def load_trvate(hp):
    print('Loading Data...')
    pos_train = pd.read_csv(os.path.join(data_dir_prefix+'data', hp['data_dir'], "pos_train.csv"))
    neg_train = pd.read_csv(os.path.join(data_dir_prefix+'data', hp['data_dir'], "neg_train.csv"))
    with open(os.path.join(data_dir_prefix+'data', hp['data_dir'], "mol_dict.pkl"), 'rb') as f:
        mol_dict = pickle.load(f)

    print('Creating Dataset...')
    train_ds, val_ds = create_train_val_dataset(hp['mode'], (mol_dict, pos_train, neg_train), (hp['pos_prec'], hp['neg_prec']),
        hp['atom_feature'], hp['bond_feature'], hp['ms_transformation'], hp['max_mz'], hp['resolution'],
        hp['instrument_on_node'], hp['self_loop'], hp['num_virtual_nodes'], hp['element_list'], hp['fp_size'], hp['noise'])

    print('Loading Data...')
    pos_test = pd.read_csv(os.path.join(data_dir_prefix+'data', hp['data_dir'], "pos_test.csv"))
    neg_test = pd.read_csv(os.path.join(data_dir_prefix+'data', hp['data_dir'], "neg_test.csv"))

    print('Creating Dataset...')
    test_ds = create_test_dataset(hp['mode'], (mol_dict, pos_test, neg_test),  (hp['pos_prec'], hp['neg_prec']),
        hp['atom_feature'], hp['bond_feature'], hp['ms_transformation'], hp['max_mz'], hp['resolution'],
        hp['instrument_on_node'], hp['self_loop'], hp['num_virtual_nodes'], hp['element_list'], hp['fp_size'], hp['noise'])

    tr_l, va_l, te_l = len(train_ds.data_list), len(val_ds.data_list),  len(test_ds.data_list)
    print(tr_l, va_l, te_l)
    tr_idx, va_idx, te_idx = [i for i in range(tr_l)], [i for i in range(tr_l, tr_l+ va_l)],  [i for i in range(tr_l+ va_l, tr_l+ va_l+ te_l)]

    with open("./data/trvate_idx.pkl", 'wb') as f:
        pickle.dump((tr_idx, va_idx, te_idx), f)

    union_ds = list(train_ds.data_list) + list(val_ds.data_list) + list(test_ds.data_list)
    convert_graph_dgl_to_torchgeometric(union_ds)


if __name__ == "__main__":
    load_trvate(hp)
