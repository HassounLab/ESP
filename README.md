# Ensembled Spectral Prediction for Metabolite Annotation (ESP)

### Xinmeng Li, Yan Zhou Chen, Apurva Kalia, Hao Zhu, Li-Ping Liu, Soha Hassoun
#### Department of Computer Science, Tufts University

A key challenge in metabolomics is annotating spectra measured from a biological sample with chemical identities. We improve on prior neural network-based annotation approaches, namely MLP-based [1] and GNN-based [2] approaches. We propose a novel ensemble model to take advantage of both MLP and GNN models. First, the MLP and GNN are enhanced by: 1) multi-tasking on additional data (spectral topic labels obtained using LDA (Latent Dirichlet Allocation) [3], and 2) attention mechanism to capture dependencies among spectra peaks. Next, we create an Ensembled Spectral Prediction (ESP) model that is trained on ranking tasks to generate the average weighted MLP and GNN spectral predictions. Our results, measured in average rank and Rank@K for the test spectra, show remarkable performance gain over existing neural network approaches.

As our aim is to fundamentally evaluate deep learning models, we created two baseline methods that can be easily implemented and replicated (with our code, or others) to do comparisons: the MLP and GNN models, per Equations 1-10.  We have shown improvements with ESP over the MLP model (implementation of NEIMS model (Wei et al., 2019) with a generalized dataset ESI/LC-MS but not EI/GC-MS data in NEIMS), in terms of a 23.7% increase in average rank performance on the full NIST candidate set. We also show 37.2% improvement in average rank over the baseline GNN model, initially presented by our group (Hao et al., 2020), which was the first to use GNNs in mass spectra annotation. The MLP and GNN are the simplest possible baseline models for comparing ML techniques - they are easily implemented and suited for re-training and evaluation when comparing to other techniques and other datasets. The published model and published datasets are for CANOPUS data.

In accordance with NIST license regulations, we are unable to publish the NIST-20 data alongside our models trained on NIST-20. Models pretrained on CANOPUS are located in /pretrained_models. CANOPUS dataset can be accessed from https://github.com/samgoldman97/mist.

The pretrained MLP, GNN, and ESP models on the CANOPUS dataset are `/pretrained_models/best_model_mlp_can.pt`, `/pretrained_models/best_model_gnn_can.pt`, and `/pretrained_models/ESP_can.pt`, respectfully.

## Requirements

### Server/Laptop Requirements
- CPU: Intel Core i5 or higher
- RAM: 8GB or higher
- GPU (optional): NVIDIA CUDA-compatible

### Software Requirements
Key packages
- python=3.8
- dgl=0.6.1
- pytorch=1.7.1
- rdkit
- pytorch-geometric=1.7.2
- numpy
- scikit-learn
- scipy

### Installation 
1. Clone the repository: git clone https://github.com/HassounLab/ESP.git
2. Install the required packages under conda (optional, can install only key requirements mentioned above):
- Install model environment, to run `train.py`
```
conda env create -f env.yml
```
and install data generation environment, to run `data_trvate.py` and `data_tecand.py`
```
conda env create -f env_d.yml
```

## Usage

### Data download

Two files under data/final_5 are too large to be part of this git repository. These are mol_dict.pkl and pos_train.csv. These datasets are kept on zonodo.org and the path is given in the files of the same name under data/final_5. Please download these files and replace the files in the git repository with the downloaded files. pos_train.csv is the CANOPUS dataset in csv format. mol_dict.pkl is a dictionary mapping InChiKeys to rdkit mol objects. You can also create mol_dict.pkl yourself if so required.

### Data Preparation

#### `data_trvate.py`
This script generates full dataset for training, validation, and test data. The generated file is `torch_trvate_`+str(int(1000/hp['resolution']))+`bin.pkl`.

To run:
```
python data_trvate.py
```

#### `data_tecand.py`
This script generates files for the test candidate torchgeometric. The generated file is named `torch_tecand_+str(int(1000/hp['resolution']))+bin_te_cand+str(hp['cand_size'])+_torchgeometric.pkl`.

To run the script, use the following command:

```
python data_tecand.py
```

### Model training
#### `train.py`
This script is used for training and restoring saved GNN and MLP models.

#### `ens_train_realistic.py`
This script is used for training and restoring saved ESP models.

To run the saved GNN model on a test set, set `--te_cand_dataset_suffix` to the desired test set and `--model_file_suffix` to the model which you wish to use.

For example, if your test set is `torch_tecand_1000bin_te_cand100` and your pretrained model is `best_model_gnn_pd.pt`, you would use the following command:

```
python train.py --cuda 1 --model gnn --model_file_suffix gnn_pd --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --te_cand_dataset_suffix torch_tecand_1000bin_te_cand100
```


To run the saved GNN model on a test set, set `--te_cand_dataset_suffix` to the desired test set and `--model_file_suffix` to the model which you wish to use.
 
For example, if your test set is `torch_tecand_1000bin_te_cand100` and your pretrained model is `best_model_mlp_pd.pt`, you would use the following command:

```
python train.py --cuda 1 --model mlp --model_file_suffix mlp_pd --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --te_cand_dataset_suffix torch_tecand_1000bin_te_cand100
```

To run a saved ESP model on a test set, set `--te_cand_dataset_suffix` to the desired test set,  `--ens_model_file_suffix` to the ESP which you wish to use, and `--mlp_model_file_suffix` and ` --gnn_model_file_suffix` to the pretrained MLP and GNN models, respectfully. 

For example, to run `ESP.pt` on `torch_tecand_1000bin_te_cand100` with MLP model `best_model_mlp_pd.pt` and GNN model `best_model_gnn_pd.pt`, use the following command:

```
python ens_train.py --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --mlp_model_file_suffix mlp_pd --gnn_model_file_suffix gnn_pd --ens_model_file_suffix ESP --cuda 1 --te_cand_dataset_suffix torch_tecand_1000bin_te_cand100
```

### Training new models
To train a new GNN or MLP model, set `--te_cand_dataset_suffix` to an empty string or don't call this argument. This will generate a file with parameters named `best_model_' + args.model_file_suffix + '.pt`.

For example, to train a new GNN model, use the following command:

```
python train.py --cuda 1 --model gnn  --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --model_file_suffix gnn_XX
```

Before you train a new ESP model, you must have pretrained MLP and GNN models (see above instructions). To train a new ESP model, set `--te_cand_dataset_suffix` to an empty string or don't call this argument. `--ens_model_file_suffix` should start with `ESP`. This will generate a file with parameters named `args.ens_model_file_suffix + '.pt`:

```
python ens_train.py --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --mlp_model_file_suffix mlp_pd --gnn_model_file_suffix gnn_pd --ens_model_file_suffix ESP --cuda 1
```

## Demo below shows results of the NIST20 test data with 100 candidates.
### To run the pretrained MLP

```
python train.py --cuda 0 --model gnn --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --disable_mt_ontology  --correlation_mat_rank 100 --model_file_suffix gnn_can --full_dataset
```

### Expected output

```
Namespace(cuda=0, model_file_suffix='mlp_can', lr=0.0005, l2norm=0.0, drop_ratio=0.3, batch_size=128, epochs=50, hidden_dims=1024, num_hidden_layers=3, JK='last', graph_pooling='mean', model='mlp', disable_mt_lda=False, correlation_mat_rank=100, mt_lda_weight=0.01, correlation_mix_residual_weight=0.7, disable_two_step_pred=True, disable_reverse=False, disable_fingerprint=True, disable_mt_fingerprint=True, disable_mt_ontology=True, full_dataset=True)

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 819/819 [09:14<00:00,  1.48it/s]
Average rank 339.350 +- 1264.715
Rank at 1 0.230
Rank at 2 0.310
Rank at 3 0.374
Rank at 4 0.413
Rank at 5 0.436
Rank at 6 0.459
Rank at 7 0.474
Rank at 8 0.493
Rank at 9 0.509
Rank at 10 0.520
Rank at 11 0.534
Rank at 12 0.540
Rank at 13 0.560
Rank at 14 0.569
Rank at 15 0.571
Rank at 16 0.584
Rank at 17 0.586
Rank at 18 0.590
Rank at 19 0.601
Rank at 20 0.609

```

### To run the pretrained GNN
```
 python train.py --cuda 0 --model gnn --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --disable_mt_ontology  --correlation_mat_rank 100 --model_file_suffix gnn_can --full_dataset
```

### Expected output
```
Namespace(cuda=0, model_file_suffix='gnn_can', lr=0.0005, l2norm=0.0, drop_ratio=0.3, batch_size=128, epochs=50, hidden_dims=1024, num_hidden_layers=3, JK='last', graph_pooling='mean', model='gnn', disable_mt_lda=False, correlation_mat_rank=100, mt_lda_weight=0.01, correlation_mix_residual_weight=0.7, disable_two_step_pred=True, disable_reverse=False, disable_fingerprint=True, disable_mt_fingerprint=True, disable_mt_ontology=True, full_dataset=True)
Average rank 241.753 +- 939.827
Rank at 1 0.115
Rank at 2 0.209
Rank at 3 0.265
Rank at 4 0.308
Rank at 5 0.332
Rank at 6 0.369
Rank at 7 0.400
Rank at 8 0.421
Rank at 9 0.443
Rank at 10 0.464
Rank at 11 0.479
Rank at 12 0.490
Rank at 13 0.507
Rank at 14 0.519
Rank at 15 0.527
Rank at 16 0.536
Rank at 17 0.545
Rank at 18 0.551
Rank at 19 0.565
Rank at 20 0.573

```

### To run the pretrained ESP
python ens_train_canopus.py --cuda 0 --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology   --correlation_mat_rank 100  --full_dataset --mode 'canopus'

### Expected output
```
Namespace(cuda=0, mlp_model_file_suffix='mlp_fp', gnn_model_file_suffix='gnn_fp', ens_model_file_suffix='ens_fp', lr=0.001, l2norm=0.0, drop_ratio=0.3, batch_size=128, epochs=100, bins=1000, mode='canopus', hidden_dims=1024, num_hidden_layers=3, JK='last', graph_pooling='mean', disable_mt_lda=False, correlation_mat_rank=100, ensemble_hidden_dim=256, mt_lda_weight=0.01, mlp_correlation_mix_residual_weight=0.8, gnn_correlation_mix_residual_weight=0.7, disable_two_step_pred=True, disable_reverse=False, disable_fingerprint=True, disable_mt_fingerprint=True, disable_mt_ontology=True, train_with_test_ratio=-1, train_with_test_ratio_hist_size=-1, full_dataset=True)
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 819/819 [00:15<00:00, 52.74it/s]
Average rank 279.557 +- 1170.300
Rank at 1 0.187
Rank at 2 0.277
Rank at 3 0.328
Rank at 4 0.369
Rank at 5 0.398
Rank at 6 0.416
Rank at 7 0.442
Rank at 8 0.459
Rank at 9 0.479
Rank at 10 0.490
Rank at 11 0.503
Rank at 12 0.515
Rank at 13 0.529
Rank at 14 0.537
Rank at 15 0.542
Rank at 16 0.556
Rank at 17 0.567
Rank at 18 0.578
Rank at 19 0.586
Rank at 20 0.592

```

#### References
[1] Wei, J.N., Belanger, D., Adams, R.P. and Sculley, D., 2019. Rapid prediction of electron–ionization mass spectrometry using neural networks. ACS central science, 5(4), pp.700-708.

[2] Zhu, H., Liu, L. and Hassoun, S., 2020. Using Graph Neural Networks for Mass Spectrometry Prediction. arXiv preprint arXiv:2010.04661.

[3] van Der Hooft, J.J.J., Wandy, J., Barrett, M.P., Burgess, K.E. and Rogers, S., 2016. Topic modeling for untargeted substructure exploration in metabolomics. Proceedings of the National Academy of Sciences, 113(48), pp.13738-13743.


#### Contact
Soha.Hassoun@tufts.edu
