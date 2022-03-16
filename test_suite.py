import torch
import data
import error_metrics
import  train_model
import eval_model
from dcrnn import RecurrentGCN_DCRNN
from egcno import RecurrentGCN_EGCNO
from egcnh import RecurrentGCN_EGCNH
from gclstm import RecurrentGCN_GCLSTM


TIME_STEPS = 3
CRITERION = "mse"
SENSOR = 122
SENSOR2 = 69
SENSOR3 = 18


if __name__ == '__main__':

    dataset = data.load_data(time_steps=TIME_STEPS)
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

    model_dcrnn = RecurrentGCN_DCRNN(feature_dim)
    model_egcno = RecurrentGCN_EGCNO(feature_dim)
    model_egcnh = RecurrentGCN_EGCNH(node_num, feature_dim)
    model_gclstm = RecurrentGCN_GCLSTM(feature_dim)

    optimizer_dcrnn = torch.optim.Adam(model_dcrnn.parameters(), lr=0.01)
    optimizer_egcno = torch.optim.Adam(model_egcno.parameters(), lr=0.01)
    optimizer_egcnh = torch.optim.Adam(model_egcnh.parameters(), lr=0.01)
    optimizer_gclstm = torch.optim.Adam(model_gclstm.parameters(), lr=0.01)

    mse_list = train_model.train_single_shot(model_dcrnn, optimizer_dcrnn, dynamic_node_features_train, static_edge_index,
                                  static_edge_weight, CRITERION, 200)
    pred, lab = eval_model.eval_single_shot(model_dcrnn, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR)
    eval_model.eval_single_shot(model_dcrnn, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR2)
    eval_model.eval_single_shot(model_dcrnn, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR3)

    mse_list2 = train_model.train_single_shot(model_egcno, optimizer_egcno, dynamic_node_features_train, static_edge_index,
                                  static_edge_weight, CRITERION, 200)
    pred2, lab2 = eval_model.eval_single_shot(model_egcno, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR)
    eval_model.eval_single_shot(model_egcno, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR2)
    eval_model.eval_single_shot(model_egcno, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR3)

    mse_list3 = train_model.train_single_shot(model_egcnh, optimizer_egcnh, dynamic_node_features_train, static_edge_index,
                                  static_edge_weight, CRITERION, 200)
    pred3, lab3 = eval_model.eval_single_shot(model_egcnh, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR)
    eval_model.eval_single_shot(model_egcnh, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR2)
    eval_model.eval_single_shot(model_egcnh, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR3)

    mse_list4 = train_model.train_autoregressive(model_gclstm, optimizer_gclstm, dynamic_node_features_train, static_edge_index,
                                  static_edge_weight, CRITERION, 200)
    pred4, lab4 = eval_model.eval_autoregressive(model_gclstm, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                CRITERION, SENSOR)
    eval_model.eval_autoregressive(model_gclstm, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                   CRITERION, SENSOR2)
    eval_model.eval_autoregressive(model_gclstm, dynamic_node_features_train, static_edge_index, static_edge_weight,
                                   CRITERION, SENSOR3)

    # error_metrics.plot_convergence(mse_list, mse_list2, mse_list3, mse_list4)
    # error_metrics.plot_preditions_all(SENSOR, 0, pred, pred2, pred3, pred4, lab)




