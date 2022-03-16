import torch
import error_metrics


def eval_single_shot(model, dynamic_node_features_test, static_edge_index, static_edge_weight, error_metric, sensor):
    snapshot_loss = error_metrics.get_loss_step(error_metric)
    final_loss = error_metrics.get_loss_final(error_metric)

    model.eval()
    loss = 0
    step = 0
    horizon = 864
    predictions = []
    labels = []
    for snapshot in range(dynamic_node_features_test.shape[0] - 1):
        y_hat = model(dynamic_node_features_test[snapshot], static_edge_index, static_edge_weight)
        # loss = loss + torch.mean((y_hat - dynamic_node_features_test[snapshot + 1]) ** 2)
        loss = loss + snapshot_loss(y_hat, dynamic_node_features_test[snapshot + 1])
        labels.append(dynamic_node_features_test[snapshot + 1])
        predictions.append(y_hat)
        step += 1
        if step > horizon:
            break
    # loss = loss / (step + 1)
    loss = final_loss(loss, step)
    loss = loss.item()
    print("Test {}: {:.4f}".format(error_metric, loss))
    error_metrics.loss_for_sensor(sensor, predictions, labels, error_metric)

    return predictions, labels


def eval_autoregressive(model, dynamic_node_features_test, static_edge_index, static_edge_weight, error_metric, sensor):
    snapshot_loss = error_metrics.get_loss_step(error_metric)
    final_loss = error_metrics.get_loss_final(error_metric)

    model.eval()
    loss = 0
    step = 0
    horizon = 864

    predictions = []
    labels = []
    h, c = None, None

    for snapshot in range(dynamic_node_features_test.shape[0] - 1):
        y_hat, h, c = model(dynamic_node_features_test[snapshot], static_edge_index, static_edge_weight, h, c)
        # loss = loss + torch.mean((y_hat - dynamic_node_features_test[snapshot + 1]) ** 2)
        loss = loss + snapshot_loss(y_hat, dynamic_node_features_test[snapshot + 1])
        labels.append(dynamic_node_features_test[snapshot + 1])
        predictions.append(y_hat)
        step += 1
        if step > horizon:
            break
    # loss = loss / (step + 1)
    loss = final_loss(loss, step)
    loss = loss.item()
    print("Test {}: {:.4f}".format(error_metric, loss))
    error_metrics.loss_for_sensor(sensor, predictions, labels, error_metric)

    return predictions, labels
