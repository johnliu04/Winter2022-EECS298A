import error_metrics


def train_single_shot(model, optimizer, dynamic_node_features_train, static_edge_index, static_edge_weight, error_metric, epochs):
    snapshot_loss = error_metrics.get_loss_step(error_metric)
    final_loss = error_metrics.get_loss_final(error_metric)

    mse_list = list()

    model.train()
    print("Running training...")
    for epoch in range(epochs):
        loss = 0
        step = 0
        for snapshot in range(dynamic_node_features_train.shape[0] - 1):
            y_hat = model(dynamic_node_features_train[snapshot], static_edge_index, static_edge_weight)
            # loss = loss + torch.mean((y_hat - dynamic_node_features_train[snapshot + 1]) ** 2)
            loss = loss + snapshot_loss(y_hat, dynamic_node_features_train[snapshot + 1])
            step += 1
            if step > 2000:
                break
        # loss = loss / (step + 1)
        loss = final_loss(loss, step)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print("Epoch {} train {}: {:.4f}".format(epoch, error_metric, loss.item()))

        mse_list.append(loss.item())

    return mse_list


def train_autoregressive(model, optimizer, dynamic_node_features_train, static_edge_index, static_edge_weight, error_metric, epochs):
    snapshot_loss = error_metrics.get_loss_step(error_metric)
    final_loss = error_metrics.get_loss_final(error_metric)

    mse_list = list()

    model.train()
    print("Running training...")
    for epoch in range(epochs):
        loss = 0
        step = 0
        h, c = None, None
        for snapshot in range(dynamic_node_features_train.shape[0] - 1):
            y_hat, h, c = model(dynamic_node_features_train[snapshot], static_edge_index, static_edge_weight, h, c)
            # loss = loss + torch.mean((y_hat - dynamic_node_features_train[snapshot + 1]) ** 2)
            loss = loss + snapshot_loss(y_hat, dynamic_node_features_train[snapshot + 1])
            step += 1
            if step > 2000:
                break
        # loss = loss / (step + 1)
        loss = final_loss(loss, step)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print("Epoch {} train {}: {:.4f}".format(epoch, error_metric, loss.item()))

        mse_list.append(loss.item())

    return mse_list