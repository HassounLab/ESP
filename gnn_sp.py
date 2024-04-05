import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, GINEConv
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 27 # including the extra mask tokens
num_instrument_type = 7 # including the extra mask tokens
num_fp_size = 4096 # including the extra mask tokens
num_ontology_size = 5861
num_lda_size = 100
num_bond_type = 5  # including aromatic and self-loop edge, and extra masked tokens


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)

        self.conv = GINEConv(nn=self.mlp)

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding1(edge_attr)

        x = self.conv(x, edge_index=edge_index, edge_attr=edge_embeddings)

        return x


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", disable_fingerprint=False):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Sequential(
            torch.nn.Linear(num_atom_type, emb_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(emb_dim, emb_dim)
        )

        self.x_embedding2 = torch.nn.Sequential(
            torch.nn.Linear(num_instrument_type, emb_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(emb_dim, emb_dim)
        )

        self.disable_fingerprint = disable_fingerprint
        if not self.disable_fingerprint:
            self.concat_emb_mlp = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim * 2, emb_dim),
            )
            self.x_embedding3 = torch.nn.Sequential(
                torch.nn.Linear(num_fp_size, emb_dim),
                torch.nn.Dropout(drop_ratio),
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(drop_ratio),
                torch.nn.Linear(emb_dim, emb_dim),
            )

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        x, edge_index, edge_attr, instrument, fp = argv[0], argv[1], argv[2], argv[3], argv[4]

        x1 = self.x_embedding1(x[:, :num_atom_type])
        #x2 = self.x_embedding2(instrument)
        #x = x1 + x2
        x = x1 #Canopus has no instrument conditions

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        if not self.disable_fingerprint:
            concat_emb = torch.cat([node_representation, self.x_embedding3(fp)], dim=-1)
            node_representation = self.concat_emb_mlp(concat_emb)

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin",
                 disable_two_step_pred=False,
                 disable_reverse=False,
                 disable_fingerprint=False,
                 disable_mt_fingerprint=False,
                 disable_mt_lda=False,
                 disable_mt_ontology=False,
                 correlation_mat_rank=5, correlation_type=5,
                 mt_lda_weight=0.01, correlation_mix_residual_weight=0.3):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type, disable_fingerprint=disable_fingerprint)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        gnn_output_emb_size = self.mult * (self.num_layer + 1) * self.emb_dim if self.JK == "concat" else self.mult * self.emb_dim

        self.disable_two_step_pred = disable_two_step_pred
        if not self.disable_two_step_pred:
            self.graph_binary_linear = torch.nn.Sequential(torch.nn.Linear(gnn_output_emb_size, self.num_tasks), torch.nn.Sigmoid())

        self.disable_reverse = disable_reverse
        if not self.disable_reverse:
            self.graph_pred_linear_reverse = torch.nn.Sequential(torch.nn.Linear(gnn_output_emb_size, self.num_tasks))
            self.gate = torch.nn.Sequential(torch.nn.Linear(gnn_output_emb_size, self.num_tasks), torch.nn.Sigmoid())

        self.disable_mt_fingerprint = disable_mt_fingerprint
        if not self.disable_mt_fingerprint:
            self.graph_pred_mt_fp = torch.nn.Sequential(torch.nn.Linear(gnn_output_emb_size, num_fp_size), torch.nn.Sigmoid())

        self.disable_mt_lda = disable_mt_lda
        if not self.disable_mt_lda:
            self.graph_pred_mt_lda = torch.nn.Sequential(torch.nn.Linear(gnn_output_emb_size, num_lda_size), torch.nn.Softmax(dim=-1))
            self.mt_lda_weight = mt_lda_weight

        self.disable_mt_ontology = disable_mt_ontology
        if not self.disable_mt_ontology:
            self.graph_pred_mt_ontology = torch.nn.Sequential(torch.nn.Linear(gnn_output_emb_size, num_ontology_size), torch.nn.Sigmoid())

        self.correlation_mat_rank = correlation_mat_rank
        if self.correlation_mat_rank > 0:
            self.correlation_mat = torch.nn.Parameter(torch.randn([correlation_type, self.num_tasks, self.correlation_mat_rank]), requires_grad=True)
            self.correlation_belong = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, correlation_type),
                torch.nn.Softmax(dim=-1)
            )
            self.correlation_type = correlation_type
            self.correlation_mix_residual_weight = correlation_mix_residual_weight

        self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(gnn_output_emb_size, self.num_tasks))

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv, return_logits=False):
        x, edge_index, edge_attr, batch, instrument, fp, shift = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]

        node_representation = self.gnn(x, edge_index, edge_attr, instrument[batch], fp[batch])

        pred_logit = self.pool(node_representation, batch)

        # multi task
        loss = 0.0
        if self.training:
            if not self.disable_mt_fingerprint:
                pred_mt_fp = self.graph_pred_mt_fp(pred_logit)
                loss_mt_fp = F.binary_cross_entropy(pred_mt_fp, fp)
                loss = loss + 0.1 * loss_mt_fp

            if not self.disable_mt_lda:
                if len(argv) >= 8:
                    lda_feature = argv[7]
                    pred_mt_lda = self.graph_pred_mt_lda(pred_logit)
                    loss_mt_lda = -(lda_feature * torch.log(pred_mt_lda + 1e-9)).sum(-1).mean()
                    loss = loss + self.mt_lda_weight * loss_mt_lda

            if not self.disable_mt_ontology:
                if len(argv) >= 9:
                    mt_ontology_feature = argv[8]
                    pred_mt_ontology = self.graph_pred_mt_ontology(pred_logit)
                    loss_mt_ontology = F.binary_cross_entropy(pred_mt_ontology, mt_ontology_feature, reduction='none')
                    mt_ontology_mask = mt_ontology_feature.sum(-1, keepdims=True)
                    mt_ontology_mask = torch.clamp_max_(mt_ontology_mask, max=1.0)
                    loss_mt_ontology = (loss_mt_ontology * mt_ontology_mask).mean()
                    loss = loss + 0.1 * loss_mt_ontology

        pred_val = self.graph_pred_linear(pred_logit)

        if not self.disable_reverse:
            pred_val_reverse = torch.flip(self.graph_pred_linear_reverse(pred_logit), dims=[1])
            for i in range(len(shift)):
                pred_val_reverse[i, :] = pred_val_reverse[i, :].roll(shift[i].item())
                pred_val_reverse[i, shift[i]:] = 0
            gate = self.gate(pred_logit)
            pred_val = gate * pred_val + (1 - gate) * pred_val_reverse

        if not self.disable_two_step_pred:
            pred_binary = self.graph_binary_linear(pred_logit)
            pred_val = pred_binary * pred_val

        pred_val = F.softplus(pred_val)

        if self.correlation_mat_rank > 0:
            y_belong = self.correlation_belong(pred_logit).unsqueeze(-1)
            y = pred_val.reshape([1, -1, self.num_tasks])
            y = y @ self.correlation_mat @ self.correlation_mat.transpose(-1, -2)
            y = y.transpose(0, 1)
            y = (y * y_belong).sum(-2)
            y = F.softplus(y)
            pred_val = (1.0 - self.correlation_mix_residual_weight) * y + self.correlation_mix_residual_weight * pred_val

        if return_logits:
            return pred_val, loss, pred_logit
        else:
            return pred_val, loss


if __name__ == "__main__":
    pass
