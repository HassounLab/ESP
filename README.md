# Ensembled Spectral Prediction for Metabolite Annotation (ESP)

### Xinmeng Li, Hao Zhu, Li-Ping Liu, Soha Hassoun
#### Department of Computer Science, Tufts University

A key challenge in metabolomics is annotating spectra measured from a biological sample with chemical identities. We improve on prior neural network-based annotation approaches, namely MLP-based [1] and GNN-based [2] approaches. We propose a novel ensemble model to take advantage of both MLP and GNN models. First, the MLP and GNN are enhanced by: 1) multi-tasking on additional data (spectral topic labels obtained using LDA (Latent Dirichlet Allocation) [3], and 2) attention mechanism to capture dependencies among spectra peaks. Next, we create an Ensembled Spectral Prediction (ESP) model that is trained on ranking tasks to generate the average weighted MLP and GNN spectral predictions. Our results, measured in average rank and Rank@K for the test spectra, show remarkable performance gain over existing neural network approaches.


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

To run the saved GNN model by loading the file `best_model_gnn_e.pt`, use the following command:

```
python train.py --cuda 5 --model gnn --model_file_suffix gnn_e --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100
```

To run the saved MLP model by loading the file `best_model_mlp_e.pt`, use the following command:

```
python train.py --cuda 5 --model mlp --model_file_suffix mlp_e --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100
```

### Training new models
To train a new GNN or MLP model, remember to un-comment train(epoch=args.epochs). This will generate a file with `parameters named best_model_ + args.model_file_suffix + .pt`.

For example, to train a new GNN model, use the following command:

```
python train.py --cuda 5 --model gnn --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --model_file_suffix gnn_XX
```

## Demo below shows results on 100 samples in the test data.
### To run the pretrained MLP

```
python train.py --cuda 5 --model mlp --model_file_suffix mlp_e --disable_two_step_pred --disable_fingerprint --disable
```

### Expected output

```
Namespace(JK='last', batch_size=128, correlation_mat_rank=100, correlation_mix_residual_weight=0.7, cuda=5, disable_fingerprint=True, disable_mt_fingerprint=True, disable_mt_lda=False, disable_mt_ontology=True, disable_reverse=False, disable_two_step_pred=True, drop_ratio=0.3, epochs=50, full_dataset=False, graph_pooling='mean', hidden_dims=1024, l2norm=0.0, lr=0.0005, model='mlp', model_file_suffix='mlp_e', mt_lda_weight=0.01, num_hidden_layers=3)
100%|█████████████████████████████████████████| 100/100 [00:05<00:00, 17.97it/s]
Average rank 6.180 +- 14.371
Rank at 1 0.560
Rank at 2 0.710
Rank at 3 0.760
Rank at 4 0.770
Rank at 5 0.790
Rank at 6 0.810
Rank at 7 0.820
Rank at 8 0.830
Rank at 9 0.860
Rank at 10 0.870
Rank at 11 0.870
Rank at 12 0.880
Rank at 13 0.900
Rank at 14 0.910
Rank at 15 0.920
Rank at 16 0.920
Rank at 17 0.920
Rank at 18 0.920
Rank at 19 0.920
Rank at 20 0.920
```

### To run the pretrained GNN
```
python train.py --model gnn --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --model_file_suffix gnn_e
```

### Expected output
```
Namespace(JK='last', batch_size=128, correlation_mat_rank=100, correlation_mix_residual_weight=0.7, cuda=0, disable_fingerprint=True, disable_mt_fingerprint=True, disable_mt_lda=False, disable_mt_ontology=True, disable_reverse=False, disable_two_step_pred=True, drop_ratio=0.3, epochs=50, full_dataset=False, graph_pooling='mean', hidden_dims=1024, l2norm=0.0, lr=0.0005, model='gnn', model_file_suffix='gnn_e', mt_lda_weight=0.01, num_hidden_layers=3)
100%|█████████████████████████████████████████| 100/100 [01:50<00:00,  1.10s/it]
Average rank 4.490 +- 8.877
Rank at 1 0.500
Rank at 2 0.610
Rank at 3 0.670
Rank at 4 0.730
Rank at 5 0.750
Rank at 6 0.790
Rank at 7 0.860
Rank at 8 0.900
Rank at 9 0.930
Rank at 10 0.950
Rank at 11 0.950
Rank at 12 0.950
Rank at 13 0.950
Rank at 14 0.970
Rank at 15 0.970
Rank at 16 0.970
Rank at 17 0.970
Rank at 18 0.970
Rank at 19 0.970
Rank at 20 0.970
```

### To run the pretrained ESP
python ens_train.py --disable_two_step_pred --disable_fingerprint --disable_mt_fingerprint --disable_mt_ontology --correlation_mat_rank 100 --mlp_model_file_suffix mlp_e --gnn_model_file_suffix gnn_e

### Expected output
```
Namespace(JK='last', batch_size=128, correlation_mat_rank=100, cuda=0, disable_fingerprint=True, disable_mt_fingerprint=True, disable_mt_lda=False, disable_mt_ontology=True, disable_reverse=False, disable_two_step_pred=True, drop_ratio=0.3, ensemble_hidden_dim=256, epochs=50, full_dataset=False, gnn_correlation_mix_residual_weight=0.7, gnn_model_file_suffix='gnn_e', graph_pooling='mean', hidden_dims=1024, l2norm=0.0, lr=0.001, mlp_correlation_mix_residual_weight=0.8, mlp_model_file_suffix='mlp_e', mt_lda_weight=0.01, num_hidden_layers=3, train_with_test_ratio=-1, train_with_test_ratio_hist_size=-1)
cpu
100%|█████████████████████████████████████████| 100/100 [01:49<00:00,  1.10s/it]
Average rank 3.740 +- 6.435
Rank at 1 0.520
Rank at 2 0.630
Rank at 3 0.700
Rank at 4 0.820
Rank at 5 0.860
Rank at 6 0.910
Rank at 7 0.910
Rank at 8 0.930
Rank at 9 0.950
Rank at 10 0.950
Rank at 11 0.950
Rank at 12 0.950
Rank at 13 0.950
Rank at 14 0.950
Rank at 15 0.950
Rank at 16 0.950
Rank at 17 0.950
Rank at 18 0.950
Rank at 19 0.960
Rank at 20 0.960
```

#### References
[1] Wei, J.N., Belanger, D., Adams, R.P. and Sculley, D., 2019. Rapid prediction of electron–ionization mass spectrometry using neural networks. ACS central science, 5(4), pp.700-708.

[2] Zhu, H., Liu, L. and Hassoun, S., 2020. Using Graph Neural Networks for Mass Spectrometry Prediction. arXiv preprint arXiv:2010.04661.

[3] van Der Hooft, J.J.J., Wandy, J., Barrett, M.P., Burgess, K.E. and Rogers, S., 2016. Topic modeling for untargeted substructure exploration in metabolomics. Proceedings of the National Academy of Sciences, 113(48), pp.13738-13743.


#### Contact
Soha.Hassoun@tufts.edu
