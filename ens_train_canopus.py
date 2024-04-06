import argparse
import copy
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from gnn_sp import GNN_graphpred
from mlp_mt import MLP_MT
from utils import compute_data_split, batch_data_list, convert_candidate_to_data_list, compute_lda_feature

from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from tqdm import tqdm
import time
np.random.seed(1)

import matplotlib.pyplot as plt

num_atom_type = 27


def train(epoch):
    mlp_model.eval()
    gnn_model.eval()


    for epoch_i in tqdm(range(epoch)):
        ensemble_model.train()

        train_loss = []

        for train_data_batch in train_fl_loader:
            rank_data_batch = train_data_batch.to(device)

            bins = rank_data_batch.x
            bins = torch.round(bins * 100.0) / 100.0
            bins = (bins - bins.mean()) / bins.std()

            ensemble_w = ensemble_model(bins)

            loss = F.binary_cross_entropy(ensemble_w, rank_data_batch.y, weight=rank_data_batch.weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        print("epoch %d, train loss %.2f" % (epoch_i + 1, np.mean(train_loss)))

        torch.save(ensemble_model.state_dict(), './results/fp_' + ensemble_model_identifier + '_epoch' + str(epoch_i) + '.pt')
        torch.save(ensemble_model.state_dict(), './results/fp_' + ensemble_model_identifier + '_best' + '.pt')

        # eval
        test_data_rank, _ = compute_rank()
        test_data_rank = np.array(test_data_rank)
        rank2file(epoch_i, test_data_rank)
def rank2file(i_epoch, test_data_rank):
    file2write.writelines("%d\n" % i_epoch)
    file2write.writelines("Average rank %.3f +- %.3f\n" % (test_data_rank.mean(), test_data_rank.std()))
    for i in range(1, 21):
        file2write.writelines("Rank at %d %.3f\n" % (i, (test_data_rank <= i).sum() / float(test_data_rank.shape[0])))
    file2write.writelines("\n\n\n")
    file2write.flush()


@torch.no_grad()
def compute_rank(model=-1, stop_i=-1):
    # ensemble_model.load_state_dict(torch.load('./results/'+args.ensemble_model_file_suffix, map_location='cpu'))  # TODO
    #AK
    ensemble_model.load_state_dict(torch.load(dir_path + 'best_model_ens_' + args.mode + '.pt', map_location='cpu'))  # TODO
    #ensemble_model.load_state_dict(torch.load('./results/fp_' + ensemble_model_identifier + '_best' + '.pt', map_location='cpu'))  # TODO

    mlp_model.eval()
    gnn_model.eval()
    ensemble_model.eval()

    # similarity = '_least'
    # print(similarity)
    # if similarity == '_original':
    #     with open('./data/torch_tecand_1000bin_te_cand100.pkl', 'rb') as fi:
    #         test_candidate_dict = pickle.load(fi)
    #     test_candidate_dict = test_candidate_dict
    # elif similarity == '_most':
    #     with open('./data/torch_tecand_1000bin_te_cand100_sim_least.pkl', 'rb') as fi:
    #         sim_least = pickle.load(fi)
    #     test_candidate_dict = sim_least
    # elif similarity == '_least':
    #     with open('./data/torch_tecand_1000bin_te_cand100_sim_most.pkl', 'rb') as fi:
    #         sim_most = pickle.load(fi)
    #     test_candidate_dict = sim_most
    rank = []
    loss_list = []
    mlp_loss, gnn_loss, mlp_rank, gnn_rank = [], [], [], []
    label = []
    loss_diff = []


    for i in tqdm(range(len(test_data_list))):
        # if i > 0 and i % 100 == 0:
        #     print("Average rank %.3f +- %.3f" % (np.mean(rank), np.std(rank)))
        # if stop_i != -1 and i >= stop_i:
        #     print("Average rank %.3f +- %.3f" % (np.mean(rank), np.std(rank)))
        #     return rank, None
        try:
            pred_out = torch.load("./results/cached_mlp_gnn_pred_" + args.mode + "_" + str(args.bins) + "/%d.pt" % i)
            mlp_out = pred_out[0].to(device)
            gnn_out = pred_out[1].to(device)
            by = pred_out[2].to(device)
        except:
            test_data = test_data_list[i]
            test_data[2][1] = test_data[2][1][:, :num_atom_type]
            #test_candidate_val = test_cand_list[i]
            #AK
            test_inchi_key = test_data[0]
            test_candidate_val = test_candidate_dict[test_inchi_key]


            test_candidate = convert_candidate_to_data_list(test_data, copy.deepcopy(test_candidate_val))
            #AK
            batch_list = [test_data] + test_candidate
            cand_loader = DataLoader(batch_list, batch_size=args.batch_size, shuffle=False, collate_fn=batch_data_list)
            mlp_out = torch.Tensor().to(device)
            gnn_out = torch.Tensor().to(device)
            by = torch.Tensor().to(device)

            #rank_data_batch = batch_data_list([test_data] + test_candidate)
            #rank_data_batch = rank_data_batch.to(device)

            # mlp_out, _, mlp_logits
            for rank_data_batch in cand_loader:
                rank_data_batch = rank_data_batch.to(device)
                mlp_out_batch, _, _ = mlp_model(rank_data_batch.x, rank_data_batch.edge_index, rank_data_batch.edge_attr,
                                       rank_data_batch.batch,
                                       rank_data_batch.instrument, rank_data_batch.fp, rank_data_batch.shift, return_logits=True)

                # gnn_out, _, gnn_logits
                gnn_out_batch, _, _ = gnn_model(rank_data_batch.x, rank_data_batch.edge_index, rank_data_batch.edge_attr,
                                       rank_data_batch.batch,
                                       rank_data_batch.instrument, rank_data_batch.fp, rank_data_batch.shift, return_logits=True)

                batch_y = rank_data_batch.y
                mlp_out = torch.cat([mlp_out, mlp_out_batch])
                gnn_out = torch.cat([gnn_out, gnn_out_batch])
                by = torch.cat([by, batch_y])

            pred_out = torch.vstack([mlp_out[None, :, :], gnn_out[None, :, :], by[None, :, :]])
            torch.save(pred_out, ("./results/cached_mlp_gnn_pred_" + args.mode + "_" + str(args.bins) + "/%d.pt" % i))

        # ensemble_w = ensemble_model(mlp_logits.detach(), gnn_logits.detach())
        # if ensemble_w[0] < 0.5:
        #     ensemble_w = torch.ones_like(ensemble_w)
        # else:
        #     ensemble_w = torch.zeros_like(ensemble_w)

        cosine_sim_mlp_all = F.cosine_similarity(mlp_out, by).cpu().detach().numpy()
        cosine_sim_gnn_all = F.cosine_similarity(gnn_out, by).cpu().detach().numpy()
        mlp_loss.append(-cosine_sim_mlp_all[0])
        gnn_loss.append(-cosine_sim_gnn_all[0])


        mlp_rank_ = (-cosine_sim_mlp_all[0] > -cosine_sim_mlp_all[2:]).sum() + 1  # skip itself
        gnn_rank_ = (-cosine_sim_gnn_all[0] > -cosine_sim_gnn_all[2:]).sum() + 1  # skip itself
        mlp_rank.append(mlp_rank_)
        gnn_rank.append(gnn_rank_)


        # if mlp_rank != gnn_rank:
        #     loss_diff.append(test_data[4]) # number of peaks[4], number of fp [5]
        #     label.append(mlp_rank - gnn_rank)
        # if i % 100 == 0 or i == len(test_data_list) - 1:
        #     loss_diff_ = np.array(loss_diff)
        #     label_ = np.array(label)
        #     np.savez('./results/spectra_label', spectra=loss_diff_, label=label_)



        # if test_data[1] >= 320:
        #     ensemble_w = torch.ones_like(mlp_out)
        # else:
        #     ensemble_w = torch.zeros_like(mlp_out)
        #
        # # compute feature
        # bin_mlp = torch.histc(-loss_mlp, bins=args.train_with_test_ratio_hist_size, min=0.0, max=1.0).detach()
        # bin_gnn = torch.histc(-loss_gnn, bins=args.train_with_test_ratio_hist_size, min=0.0, max=1.0).detach()
        # # bin_mlp = (bin_mlp - bin_mlp.mean()) / bin_mlp.std()
        # # bin_gnn = (bin_gnn - bin_gnn.mean()) / bin_gnn.std()
        # bin_mlp = bin_mlp / bin_mlp.sum()
        # bin_gnn = bin_gnn / bin_gnn.sum()
        # bins = torch.cat([bin_mlp, bin_gnn], dim=-1).unsqueeze(0)
        # ensemble_w = ensemble_model(bins)

        # if ensemble_w.item() > 0.5:
        #     ensemble_w = torch.ones_like(mlp_out)
        # else:
        #     ensemble_w = torch.zeros_like(mlp_out)

        bins = by[0].unsqueeze(0)
        bins = torch.round(bins * 100.0) / 100.0
        bins = (bins - bins.mean()) / bins.std()
        ensemble_w = ensemble_model(bins)

        # if ensemble_w.squeeze().item() > 0.5:
        #     ensemble_w = torch.ones_like(ensemble_w)
        # else:
        #     ensemble_w = torch.zeros_like(ensemble_w)

        out = ensemble_w * mlp_out + (1.0 - ensemble_w) * gnn_out
        # out = mlp_out
        # out = gnn_out

        loss = -F.cosine_similarity(out, by)
        loss = loss.cpu().numpy()
        test_rank = (loss[0] > loss[2:]).sum() + 1  # skip itself

        rank.append(test_rank)
        loss_list.append(loss[0])

    rank = np.array(rank)
    loss_list = np.array(loss_list)
    # with open('./results/ranks_sim'+similarity+'.csv', 'w') as f:
    #     print("Average rank %.3f +- %.3f" % (rank.mean(), rank.std()), file=fi)
    #     for i in range(1, 21):
    #         print("Rank at %d %.3f" % (i, (rank <= i).sum() / float(rank.shape[0])), file=fi)

    #
    #     for i in range(len(rank)):
    #         f.write(str(rank[i]) + ',')
    #
    # np.savez('./results/prediction_' + args.ensemble_model_file_suffix, ensemble_rank=rank, ensemble_loss=loss_list,
    #          mlp_rank=mlp_rank, gnn_rank=gnn_rank,
    #          mlp_loss=mlp_loss, gnn_loss=gnn_loss)

    return rank, loss_list


def construct_train_dataset():
    with open('./data/inchikey_formula_dict.pkl', 'rb') as fi:
        inchikey2formula = pickle.load(fi)

    mol_formula = [inchikey2formula[train_data[i][0]] for i in range(len(train_data))]

    train_dict_group_by_fl = {}

    n_train_data = len(mol_formula)

    assert n_train_data == len(train_data)

    for i in range(n_train_data):
        mol_formula_i = mol_formula[i]
        if mol_formula_i not in train_dict_group_by_fl:
            train_dict_group_by_fl[mol_formula_i] = []
        train_dict_group_by_fl[mol_formula_i].append(train_data[i])

    constructed_train_data = []

    for fl_values in tqdm(train_dict_group_by_fl.values()):
        if len(fl_values) == 1:
            continue

        fl_batch_data = batch_data_list(fl_values).to(device)

        mlp_out, _ = mlp_model(fl_batch_data.x, fl_batch_data.edge_index, fl_batch_data.edge_attr,
                               fl_batch_data.batch, fl_batch_data.instrument, fl_batch_data.fp, fl_batch_data.shift)

        gnn_out, _ = gnn_model(fl_batch_data.x, fl_batch_data.edge_index, fl_batch_data.edge_attr,
                               fl_batch_data.batch, fl_batch_data.instrument, fl_batch_data.fp, fl_batch_data.shift)

        mlp_out = mlp_out.detach()
        gnn_out = gnn_out.detach()

        for i in range(len(fl_values)):
            spectra = torch.from_numpy(fl_values[i][4]).to(device)
            spectra = spectra.unsqueeze(0).repeat(len(fl_values), 1)
            loss_mlp = -F.cosine_similarity(mlp_out, spectra).detach()
            loss_gnn = -F.cosine_similarity(gnn_out, spectra).detach()

            rank_mlp = (loss_mlp[i] > loss_mlp).sum() + 1  # skip itself
            rank_gnn = (loss_gnn[i] > loss_gnn).sum() + 1  # skip itself
            rank_mlp = rank_mlp.item()
            rank_gnn = rank_gnn.item()

            if rank_mlp == rank_gnn:
                continue

            label = 1 if rank_mlp < rank_gnn else 0
            # label = 1 if -loss_mlp[i] - (-loss_gnn[i]) >0 else 0 ## only using target loss
            # weight = 1.0
            weight = abs(rank_mlp - rank_gnn) / ((rank_mlp + rank_gnn) / 2.0)

            constructed_train_data.append([np.expand_dims(fl_values[i][4], 0),
                                           label * np.ones([1, 1]),
                                           weight * np.ones([1, 1],)])

    # with open('./data/1000bin_train_model_selector.pkl', 'wb') as fo:
    #     pickle.dump(constructed_train_data, fo, protocol=4)

    # with open('./data/1000bin_train_model_selector_loss_only.pkl', 'wb') as fo: ## only using target loss
    #     pickle.dump(constructed_train_data, fo, protocol=4)

    with open(dir_path + 'ens_train_' + args.mode + '_' + str(args.bins) + '.pkl', 'wb') as fo:
        pickle.dump(constructed_train_data, fo, protocol=4)

def construct_valid_dataset():
    mlp_model.eval()
    gnn_model.eval()
    with open('./data/inchikey_formula_dict.pkl', 'rb') as fi:
        inchikey2formula = pickle.load(fi)

    mol_formula = [inchikey2formula[valid_data[i][0]] for i in range(len(valid_data))]

    valid_dict_group_by_fl = {}

    n_valid_data = len(mol_formula)

    assert n_valid_data == len(valid_data)

    for i in range(n_valid_data):
        mol_formula_i = mol_formula[i]
        if mol_formula_i not in valid_dict_group_by_fl:
            valid_dict_group_by_fl[mol_formula_i] = []
        valid_dict_group_by_fl[mol_formula_i].append(valid_data[i])

    constructed_valid_data = []

    for fl_values in tqdm(valid_dict_group_by_fl.values()):
        if len(fl_values) == 1:
            continue

        fl_batch_data = batch_data_list(fl_values).to(device)

        mlp_out, _ = mlp_model(fl_batch_data.x, fl_batch_data.edge_index, fl_batch_data.edge_attr,
                               fl_batch_data.batch, fl_batch_data.instrument, fl_batch_data.fp, fl_batch_data.shift)

        gnn_out, _ = gnn_model(fl_batch_data.x, fl_batch_data.edge_index, fl_batch_data.edge_attr,
                               fl_batch_data.batch, fl_batch_data.instrument, fl_batch_data.fp, fl_batch_data.shift)

        mlp_out = mlp_out.detach()
        gnn_out = gnn_out.detach()

        for i in range(len(fl_values)):
            spectra = torch.from_numpy(fl_values[i][4]).to(device)
            spectra = spectra.unsqueeze(0).repeat(len(fl_values), 1)
            loss_mlp = -F.cosine_similarity(mlp_out, spectra).detach()
            loss_gnn = -F.cosine_similarity(gnn_out, spectra).detach()

            rank_mlp = (loss_mlp[i] > loss_mlp).sum() + 1  # skip itself
            rank_gnn = (loss_gnn[i] > loss_gnn).sum() + 1  # skip itself
            rank_mlp = rank_mlp.item()
            rank_gnn = rank_gnn.item()

            if rank_mlp == rank_gnn:
                continue

            label = 1 if rank_mlp < rank_gnn else 0
            # label = 1 if -loss_mlp[i] - (-loss_gnn[i]) >0 else 0 ## only using target loss
            # weight = 1.0
            weight = abs(rank_mlp - rank_gnn) / ((rank_mlp + rank_gnn) / 2.0)

            constructed_valid_data.append([np.expand_dims(fl_values[i][4], 0),
                                           label * np.ones([1, 1]),
                                           weight * np.ones([1, 1],)])

    # with open('./data/1000bin_train_model_selector.pkl', 'wb') as fo:
    #     pickle.dump(constructed_train_data, fo, protocol=4)

    # with open('./data/1000bin_train_model_selector_loss_only.pkl', 'wb') as fo: ## only using target loss
    #     pickle.dump(constructed_train_data, fo, protocol=4)

    with open(dir_path + 'ens_valid_' + args.mode + '_' + str(args.bins) + '.pkl', 'wb') as fo:
        pickle.dump(constructed_valid_data, fo, protocol=4)


def batch_data_list_fl(data_list, **kwargs):
    graph_list = []
    for i in range(len(data_list)):
        graph_list.append(Data(
            x=torch.from_numpy(data_list[i][0]).float(),
            y=torch.from_numpy(data_list[i][1]).float(),
            weight=torch.from_numpy(data_list[i][2]).float(),
        )
        )

    return Batch.from_data_list(graph_list)


# python ens_train_realistic.py --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100  --cuda 1 --l2norm 1e-6
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # cluster parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mlp_model_file_suffix', type=str, default='mlp_fp')
    parser.add_argument('--gnn_model_file_suffix', type=str, default='gnn_fp')
    #AK
    parser.add_argument('--ens_model_file_suffix', type=str, default='ens_fp')
    # training parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2norm', type=float, default=0.0)
    parser.add_argument('--drop_ratio', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bins', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='fp')
    # model parameters
    parser.add_argument('--hidden_dims', type=int, default=1024)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--JK', type=str, default="last")
    parser.add_argument('--graph_pooling', type=str, default="mean")
    parser.add_argument('--disable_mt_lda', action='store_true')
    parser.add_argument('--correlation_mat_rank', type=int, default=100)
    parser.add_argument('--ensemble_hidden_dim', type=int, default=256)

    parser.add_argument('--mt_lda_weight', type=float, default=0.01)
    parser.add_argument('--mlp_correlation_mix_residual_weight', type=float, default=0.8)
    parser.add_argument('--gnn_correlation_mix_residual_weight', type=float, default=0.7)

    parser.add_argument('--disable_two_step_pred', action='store_true')
    parser.add_argument('--disable_reverse', action='store_true')
    parser.add_argument('--disable_fingerprint', action='store_true')
    parser.add_argument('--disable_mt_fingerprint', action='store_true')
    parser.add_argument('--disable_mt_ontology', action='store_true')

    parser.add_argument('--train_with_test_ratio', type=float, default=-1)
    parser.add_argument('--train_with_test_ratio_hist_size', type=int, default=-1)

    parser.add_argument('--full_dataset', action='store_true')
    args = parser.parse_args()
    print(str(args))

    dir_path = "./data/realistic/"

    with open(dir_path + args.mode + '.pkl', 'rb') as f:
        split_dict = pickle.load(f)

    with open(dir_path + 'train_' + str(args.bins) + 'bin.pkl', 'rb') as f:
         all_data_by_iks = pickle.load(f)

    with open(dir_path + 'test_cand_' + args.mode + '_' + str(args.bins) + 'bin_te_cand100.pkl', 'rb') as f:
        test_cand_list = pickle.load(f)

    #AK
    with open('./data/torch_tecand_1000bin_te_cand100.pkl', 'rb') as fi:
        test_candidate_dict = pickle.load(fi)

    with open('./data/torch_trvate_1000bin.pkl', 'rb') as fi:
        data = pickle.load(fi)

    if args.full_dataset:
        with open('./data/trvate_idx.pkl', 'rb') as fi:
            split = pickle.load(fi)
        split = (split[0], split[1], split[2])

    else:
        # M+H data set (default)
        with open('./data/MHfilter_trvate_idx.pkl', 'rb') as fi:
            split = pickle.load(fi)
        with open("./data/filter_te_idx.pkl", 'rb') as f:
            filter_te_idx = pickle.load(f)
        split = (split[0], split[1], filter_te_idx)

    train_mask, val_mask, test_mask = compute_data_split(len(data), random=False, split=split)
    test_cand_list = [data[i] for i in range(len(data)) if test_mask[i]]
    #AK

    train_data = []
    test_data_list = []
    for ik in split_dict['train']:
        train_data.extend(all_data_by_iks[ik])
    for ik in split_dict['test']:
        test_data_list.extend(all_data_by_iks[ik])
    #AK
    #valid_data = []
    #for ik in split_dict['valid']:
        #valid_data.extend(all_data_by_iks[ik])


    for i, ik in enumerate(split_dict['test']):
        train_data.extend(all_data_by_iks[ik])
        if i==6000:
            break

    #assert len(test_data_list) == len(test_cand_list)
    del all_data_by_iks

    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')

    n_ms = train_data[0][4].shape[0]

    mlp_model = MLP_MT(emb_dim=args.hidden_dims, output_dim=n_ms, drop_ratio=args.drop_ratio,
                       disable_mt_lda=args.disable_mt_lda,
                       correlation_mat_rank=args.correlation_mat_rank,
                       mt_lda_weight=args.mt_lda_weight,
                       correlation_mix_residual_weight=args.mlp_correlation_mix_residual_weight).to(device)
    mlp_model.load_state_dict(
        torch.load(dir_path + 'best_model_mlp_' + args.mode + '.pt', map_location='cpu'))

    gnn_model = GNN_graphpred(num_layer=args.num_hidden_layers,
                              emb_dim=args.hidden_dims,
                              num_tasks=n_ms, JK=args.JK, drop_ratio=args.drop_ratio, graph_pooling=args.graph_pooling,
                              gnn_type="gin",
                              disable_two_step_pred=args.disable_two_step_pred,
                              disable_reverse=args.disable_reverse,
                              disable_fingerprint=args.disable_fingerprint,
                              disable_mt_fingerprint=args.disable_mt_fingerprint,
                              disable_mt_lda=args.disable_mt_lda,
                              disable_mt_ontology=args.disable_mt_ontology,
                              correlation_mat_rank=args.correlation_mat_rank,
                              mt_lda_weight=args.mt_lda_weight,
                              correlation_mix_residual_weight=args.gnn_correlation_mix_residual_weight
                              ).to(device)
    gnn_model.load_state_dict(
        torch.load(dir_path + 'best_model_gnn_' + args.mode + '.pt', map_location='cpu'))

    construct_train_dataset()
    #construct_valid_dataset()

    with open(dir_path + 'ens_train_' + args.mode + '_' + str(args.bins) + '.pkl', 'rb') as fi:
        train_data_fl = pickle.load(fi)
    train_fl_loader = DataLoader(train_data_fl, batch_size=args.batch_size, shuffle=True, collate_fn=batch_data_list_fl)

    #with open(dir_path + 'ens_valid_' + args.mode + '_' + str(args.bins) + '.pkl', 'rb') as fi:
        #valid_data_fl = pickle.load(fi)
    #valid_fl_loader = DataLoader(valid_data_fl, batch_size=args.batch_size, shuffle=True, collate_fn=batch_data_list_fl)

    ensemble_model = torch.nn.Sequential(
            torch.nn.Linear(1000, args.ensemble_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(args.drop_ratio),
            torch.nn.Linear(args.ensemble_hidden_dim, 1),
            torch.nn.Sigmoid(),
        ).to(device)

    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    ensemble_model_identifier = "ens_{mode}_{bins:d}_lr{lr:.6f}_l2{l2:.6f}_drop{drop:.1f}_hidden{ensemble_hidden_dim:d}".format(
        mode=args.mode,
        bins=args.bins,
        lr=args.lr,
        l2=args.l2norm,
        drop=args.drop_ratio,
        ensemble_hidden_dim=args.ensemble_hidden_dim,
    ).replace('.', 'sep')

    model_time = str(int(round(time.time() * 1000)))
    file2write = open('./results/realistic/' +ensemble_model_identifier +model_time+ '.txt', 'w')
    train(args.epochs)
    file2write.close()

    test_data_rank, test_data_loss = compute_rank()
    test_data_rank = np.array(test_data_rank)
    test_data_loss = np.array(test_data_loss)
    np.savez('./results/prediction_' + args.ens_model_file_suffix, test_data_rank=test_data_rank,
             test_data_loss=test_data_loss)

    print("Average rank %.3f +- %.3f" % (test_data_rank.mean(), test_data_rank.std()))
    for i in range(1, 21):
        print("Rank at %d %.3f" % (i, (test_data_rank <= i).sum() / float(test_data_rank.shape[0])))

