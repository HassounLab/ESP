import argparse
import torch
import pickle
import os
import pandas as pd
import numpy as np

import dgllife.utils as chemutils
from torch.utils.data import Dataset
from rdkit.Chem import AllChem
from rdkit import Chem
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
    "resolution": 1,

    # candidate (change)
    "cand_size": 100,
    "cand_iterations": 1}


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
    return x.split(' ', 2)[0:2]


def get_intensity(x):
    x_list = list(map(get_intensity_, x.split('\n')[:-1]))
    return np.array(x_list, dtype = np.float)


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


class msgnnCandDataset(Dataset):
    def __init__(self, data_list, instrument_on_node, atom_feature, bond_feature,
                 self_loop, num_virtual_nodes, element_list, prec_types, fp_size, noise):
        self.data_list = data_list
        self.atom_feature = atom_feature
        self.bond_feature = bond_feature
        if element_list == "chnopsh":
            self.element_list = ['H', 'C', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I']
        self.instrument_on_node = instrument_on_node
        self.self_loop = self_loop
        self.num_virtual_nodes = num_virtual_nodes
        self.prec_types = prec_types
        self.fp_size = fp_size
        self.noise = noise

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        i, shift, prec, nce, ms = self.data_list[idx]
        m = Chem.MolFromInchi(i)
        fp = np.array([int(x) for x in AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=self.fp_size).ToBitString()])
        g = chemutils.mol_to_bigraph(
            m, node_featurizer=get_atom_featurizer(self.atom_feature, self.element_list),
            edge_featurizer=get_bond_featurizer(self.bond_feature, self.self_loop),
            add_self_loop=self.self_loop,
            num_virtual_nodes=self.num_virtual_nodes
        )

        setting_tensor = get_ms_setting(prec, nce, self.prec_types)
        if self.instrument_on_node == True:
            setting_tensor_on_nodes = get_ms_setting_all_nodes(
                prec, nce, g.num_nodes(), self.prec_types
            )
            g.ndata['h'] = torch.cat((g.ndata['h'], setting_tensor_on_nodes), -1)
        if self.noise:
            g.ndata['n'] = torch.zeros((g.num_nodes(), 5))
        return g, setting_tensor, ms, shift, fp


def load_transform_rank_data(data_g):
    for y in tqdm(data_g.keys()):
        for i, x in enumerate(data_g[y]):
            data_g[y][i] = list(data_g[y][i])
            graph = x[1]
            g_nodes = graph.nodes().numpy()
            g_edges = [y.numpy() for y in graph.edges()]
            g_edges = np.vstack(g_edges)
            g_edges_f = graph.edata['e'].numpy()
            g_nodes_f = graph.ndata['h'].numpy()[g_nodes]
            data_g[y][i][1] = [g_edges, g_nodes_f, g_edges_f]

    with open('./data/torch_tecand_'+str(int(1000/hp['resolution']))+"bin_te_cand"+str(hp['cand_size'])+'.pkl', 'wb') as fp:
        pickle.dump(data_g, fp, protocol=4)


def df2list_cand(mol_dict, df, trans, max_mz, resolution, cand, inchi_ik_dict, split='random'):
    data_list = list()
    cand_mask = list()
    inchi_list = list()
    target_inchi = list()
    # print(len(df))
    atom_feature = hp['atom_feature']
    element_list = hp['element_list']
    bond_feature = hp['bond_feature']
    self_loop = hp['self_loop']
    num_virtual_nodes = hp['num_virtual_nodes']
    instrument_on_node = hp['instrument_on_node']

    if element_list == "chnopsh":
        ele_list = ['H', 'C',  'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I']

    """print(len(set(cand.keys())))# 2537 distinct molecule in cand dict"""
    """# preprocess to save time"""
    dist_data_dict = {}
    dist_cand = list(set(cand.keys()))
    dist_cand.sort()

    for t_IK in tqdm(dist_cand):
        t_data_list = []

        for inchi in cand[t_IK]:
            #m = Chem.MolFromInchi(inchi)
            m = mol_dict[inchi]
            fp = np.array([int(x) for x in AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=hp['fp_size']).ToBitString()])
            g = chemutils.mol_to_bigraph(m,
                node_featurizer=get_atom_featurizer(atom_feature, ele_list),
                edge_featurizer=get_bond_featurizer(bond_feature,self_loop),
                add_self_loop=self_loop,
                num_virtual_nodes=num_virtual_nodes
            )

            ## not yet have instrument setting on g
            t_data_list.append((inchi, g, fp))
        dist_data_dict[t_IK] = t_data_list
    return dist_data_dict


def create_test_cand_dataset(mol_dict, mode, data, precs, atom_feat, bond_feat, ms_transformation, max_mz, resolution, instrument_on_node, self_loop, num_virtual_nodes, element_list, inchi_ik_dict, fp_size,
                             noise, split="random"):
    pos_prec, neg_prec = precs
    if mode == 'positive':
        dist_data_dict = df2list_cand(pos_test, ms_transformation, max_mz, resolution, cand, inchi_ik_dict, split)
        prec_types = pos_prec

    return dist_data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    pos_test = pd.read_csv(os.path.join(data_dir_prefix + 'data', hp['data_dir'], "pos_test.csv"))
    #neg_test = pd.read_csv(os.path.join(data_dir_prefix + 'data', hp['data_dir'], "neg_test.csv"))
    neg_test = []
    with open(os.path.join(data_dir_prefix+'data', hp['data_dir'], "mol_dict.pkl"), 'rb') as f:
        mol_dict = pickle.load(f)

    # inchi_ik_dict len 27,254. Not needed for Canopus
    #with open(data_dir_prefix + 'data/test_inchi_ik_dict.pkl', 'rb') as f:
        #inchi_ik_dict = pickle.load(f)

    ### 100, 250, 1000. For NIST
    #with open(data_dir_prefix + 'data/test_cand/' + str(hp['cand_size']) + '.pkl', 'rb') as f:
        #cand_list = pickle.load(f)

    #For Canopus
    with open(data_dir_prefix + 'data/test_cand/cand_dict_large.pkl', 'rb') as f: #AK
        cand_list = pickle.load(f)

    if str(hp['cand_size'])=='full':
        cand = cand_list
    else:
        cand = cand_list[0]
    print()

    cand, pos_test, neg_test = (cand, pos_test, neg_test)

    dist_data_dict = create_test_cand_dataset(mol_dict, hp['mode'], (cand, pos_test, neg_test), (hp['pos_prec'], hp['neg_prec']),
                                                                                hp['atom_feature'], hp['bond_feature'], hp['ms_transformation'], hp['max_mz'],
                                                                                hp['resolution'],hp['instrument_on_node'],
                                                                                hp['self_loop'],hp['num_virtual_nodes'],hp['element_list'],
                                                                                inchi_ik_dict, hp['fp_size'], hp['noise'])
    load_transform_rank_data(dist_data_dict)


