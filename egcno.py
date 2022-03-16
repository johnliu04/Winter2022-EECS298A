import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
import data
import error_metrics
import train_model
import eval_model


class RecurrentGCN_EGCNO(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN_EGCNO, self).__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear = torch.nn.Linear(node_features, node_features)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


def test_egcno(time_steps, criterion, plot):
    dataset = data.load_data(time_steps=time_steps)
    split_idx = int(len(dataset.features) * 0.8)
    train_dataset, test_dataset = dataset.features[:split_idx], dataset.features[split_idx:]

    feature_dim = int(len(dataset.features[0][0]) * len(dataset.features[0][0][0]))
    node_num = int(len(dataset.features[0]))
    static_edge_index = torch.LongTensor(dataset.edge_index)
    static_edge_weight = torch.FloatTensor(dataset.edge_weight)

    dynamic_node_features_train = torch.FloatTensor(train_dataset).view(-1, node_num, feature_dim)
    dynamic_node_features_test = torch.FloatTensor(test_dataset).view(-1, node_num, feature_dim)
    print("Number of train buckets: ", len(set(dynamic_node_features_train)))
    print("Number of test buckets: ", len(set(dynamic_node_features_test)))

    model = RecurrentGCN_EGCNO(feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_model.train_single_shot(model, optimizer, dynamic_node_features_train, static_edge_index, static_edge_weight, criterion, 10)

    pred, lab = eval_model.eval_single_shot(model, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                            criterion, 18)

    if plot:
        error_metrics.plot_preditions(18, 0, pred, lab)


if __name__ == '__main__':
    test_egcno(12, "mse", True)