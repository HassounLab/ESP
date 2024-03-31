# Ensembled Spectral Prediction for Metabolite Annotation (ESP)

### Xinmeng Li, Yan Zhou Chen, Apurva Kalia, Hao Zhu, Li-Ping Liu, Soha Hassoun
#### Department of Computer Science, Tufts University

A key challenge in metabolomics is annotating spectra measured from a biological sample with chemical identities. We improve on prior neural network-based annotation approaches, namely MLP-based [1] and GNN-based [2] approaches. We propose a novel ensemble model to take advantage of both MLP and GNN models. First, the MLP and GNN are enhanced by: 1) multi-tasking on additional data (spectral topic labels obtained using LDA (Latent Dirichlet Allocation) [3], and 2) attention mechanism to capture dependencies among spectra peaks. Next, we create an Ensembled Spectral Prediction (ESP) model that is trained on ranking tasks to generate the average weighted MLP and GNN spectral predictions. Our results, measured in average rank and Rank@K for the test spectra, show remarkable performance gain over existing neural network approaches.

As our aim is to fundamentally evaluate deep learning models, we created two baseline methods that can be easily implemented and replicated (with our code, or others) to do comparisons: the MLP and GNN models, per Equations 1-10.  We have shown improvements with ESP over the MLP model (implementation of NEIMS model (Wei et al., 2019) with a generalized dataset ESI/LC-MS but not EI/GC-MS data in NEIMS), in terms of a 23.7% increase in average rank performance on the full NIST candidate set. We also show 37.2% improvement in average rank over the baseline GNN model, initially presented by our group (Hao et al., 2020), which was the first to use GNNs in mass spectra annotation. The MLP and GNN are the simplest possible baseline models for comparing ML techniques - they are easily implemented and suited for re-training and evaluation when comparing to other techniques and other datasets.

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

#### `ens_train.py`
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
python train.py --model mlp --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --model_file_suffix mlp_pd --te_cand_dataset_suffix torch_tecand_1000bin_te_cand100 --cuda 1
```

### Expected output

```
Namespace(cuda=1, model_file_suffix='mlp_pd', lr=0.0005, l2norm=0.0, drop_ratio=0.3, batch_size=128, epochs=50, hidden_dims=1024, num_hidden_layers=3, JK='last', graph_pooling='mean', model='mlp', disable_mt_lda=False, correlation_mat_rank=100, mt_lda_weight=0.01, correlation_mix_residual_weight=0.7, disable_two_step_pred=True, disable_reverse=False, disable_fingerprint=True, disable_mt_fingerprint=True, disable_mt_ontology=True, full_dataset=False, te_cand_dataset_suffix='torch_tecand_1000bin_te_cand100')
100%|███████████████████████████████████████| 8151/8151 [03:52<00:00, 35.09it/s]
Average rank 7.269 +- 23.395
Rank at 1 0.592
Rank at 2 0.705
Rank at 3 0.756
Rank at 4 0.789
Rank at 5 0.815
Rank at 6 0.833
Rank at 7 0.848
Rank at 8 0.859
Rank at 9 0.868
Rank at 10 0.878
Rank at 11 0.887
Rank at 12 0.892
Rank at 13 0.898
Rank at 14 0.904
Rank at 15 0.907
Rank at 16 0.910
Rank at 17 0.914
Rank at 18 0.917
Rank at 19 0.921
Rank at 20 0.923
```

### To run the pretrained GNN
```
python train.py --model gnn --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --model_file_suffix gnn_pd --te_cand_dataset_suffix torch_tecand_1000bin_te_cand100 --cuda 1
```

### Expected output
```
Namespace(cuda=1, model_file_suffix='gnn_pd', lr=0.0005, l2norm=0.0, drop_ratio=0.3, batch_size=128, epochs=50, hidden_dims=1024, num_hidden_layers=3, JK='last', graph_pooling='mean', model='gnn', disable_mt_lda=False, correlation_mat_rank=100, mt_lda_weight=0.01, correlation_mix_residual_weight=0.7, disable_two_step_pred=True, disable_reverse=False, disable_fingerprint=True, disable_mt_fingerprint=True, disable_mt_ontology=True, full_dataset=False, te_cand_dataset_suffix='torch_tecand_1000bin_te_cand100')
100%|███████████████████████████████████████| 8151/8151 [05:09<00:00, 26.35it/s]
Average rank 7.819 +- 25.453
Rank at 1 0.506
Rank at 2 0.652
Rank at 3 0.729
Rank at 4 0.767
Rank at 5 0.793
Rank at 6 0.815
Rank at 7 0.834
Rank at 8 0.848
Rank at 9 0.861
Rank at 10 0.871
Rank at 11 0.881
Rank at 12 0.888
Rank at 13 0.896
Rank at 14 0.901
Rank at 15 0.907
Rank at 16 0.910
Rank at 17 0.914
Rank at 18 0.919
Rank at 19 0.921
Rank at 20 0.924
```

### To run the pretrained ESP
python ens_train.py --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --mlp_model_file_suffix mlp_pd --gnn_model_file_suffix gnn_pd --ens_model_file_suffix ESP --te_cand_dataset_suffix torch_tecand_1000bin_te_cand100

### Expected output
```
Namespace(cuda=0, mlp_model_file_suffix='mlp_pd', gnn_model_file_suffix='gnn_pd', ens_model_file_suffix='ESP', lr=0.001, l2norm=0.0, drop_ratio=0.3, batch_size=128, epochs=100, hidden_dims=1024, num_hidden_layers=3, JK='last', graph_pooling='mean', disable_mt_lda=False, correlation_mat_rank=100, ensemble_hidden_dim=256, mt_lda_weight=0.01, mlp_correlation_mix_residual_weight=0.8, gnn_correlation_mix_residual_weight=0.7, disable_two_step_pred=True, disable_reverse=False, disable_fingerprint=True, disable_mt_fingerprint=True, disable_mt_ontology=True, train_with_test_ratio=-1, train_with_test_ratio_hist_size=-1, full_dataset=False, te_cand_dataset_suffix='torch_tecand_1000bin_te_cand100')
100%|███████████████████████████████████████| 8151/8151 [05:48<00:00, 23.38it/s]
Average rank 5.501 +- 19.434
Rank at 1 0.620
Rank at 2 0.745
Rank at 3 0.799
Rank at 4 0.832
Rank at 5 0.854
Rank at 6 0.870
Rank at 7 0.882
Rank at 8 0.892
Rank at 9 0.904
Rank at 10 0.912
Rank at 11 0.916
Rank at 12 0.921
Rank at 13 0.925
Rank at 14 0.929
Rank at 15 0.932
Rank at 16 0.936
Rank at 17 0.939
Rank at 18 0.941
Rank at 19 0.944
Rank at 20 0.947
```

#### References
[1] Wei, J.N., Belanger, D., Adams, R.P. and Sculley, D., 2019. Rapid prediction of electron–ionization mass spectrometry using neural networks. ACS central science, 5(4), pp.700-708.

[2] Zhu, H., Liu, L. and Hassoun, S., 2020. Using Graph Neural Networks for Mass Spectrometry Prediction. arXiv preprint arXiv:2010.04661.

[3] van Der Hooft, J.J.J., Wandy, J., Barrett, M.P., Burgess, K.E. and Rogers, S., 2016. Topic modeling for untargeted substructure exploration in metabolomics. Proceedings of the National Academy of Sciences, 113(48), pp.13738-13743.


#### Contact
Soha.Hassoun@tufts.edu
