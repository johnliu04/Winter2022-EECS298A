import math

import seaborn as sns
import numpy as np
import torch
import data
import matplotlib.pyplot as plt


def get_loss_step(error_metric):
    if error_metric == "mse":
        loss_step = loss_mse_step
    elif error_metric == "rmse":
        loss_step = loss_rmse_step
    elif error_metric == "mae":
        loss_step = loss_mae_step
    else:
        loss_step = loss_mse_step
    return loss_step


def get_loss_final(error_metric):
    if error_metric == "mse":
        loss_final = loss_mse_final
    elif error_metric == "rmse":
        loss_final = loss_rmse_final
    elif error_metric == "mae":
        loss_final = loss_mae_final
    else:
        loss_final = loss_mse_final
    return loss_final


def loss_mse_step(y_hat, y):
    return torch.mean((y_hat - y) ** 2)


def loss_mse_final(loss, step):
    return loss / (step + 1)


def loss_rmse_step(y_hat, y):
    return torch.mean((y_hat - y) ** 2)


def loss_rmse_final(loss, step):
    return torch.sqrt(loss / (step + 1))


def loss_mae_step(y_hat, y):
    return torch.mean(torch.abs(y - y_hat))


def loss_mae_final(loss, step):
    return loss / (step + 1)


def loss_for_sensor(sensor, predictions, labels, error_metric):
    snapshot_loss = get_loss_step_sensor(error_metric)
    final_loss = get_loss_final_sensor(error_metric)

    preds = np.asarray([(pred[sensor][0].detach().cpu().numpy()) for pred in predictions])
    labs = np.asarray([(label[sensor][0].cpu().numpy()) for label in labels])

    loss = 0
    step = 0
    for i in range(864):
        loss = loss + snapshot_loss(preds[i], labs[i])
        step += 1
    loss = final_loss(loss, step)
    print("Test {} for sensor {}: {:.4f}".format(error_metric, sensor, loss))


def plot_preditions(sensor, timestep, predictions, labels):
    preds = np.asarray([data.denormalize(pred[sensor][timestep].detach().cpu().numpy()) for pred in predictions])
    labs = np.asarray([data.denormalize(label[sensor][timestep].cpu().numpy()) for label in labels])
    print("Data points:,", preds.shape)

    plt.figure(figsize=(30, 10))
    sns.lineplot(data=preds, label="pred")
    sns.lineplot(data=labs, label="true")
    plt.show()


def plot_preditions_all(sensor, timestep, predictions1, predictions2, predictions3, predictions4, labels):
    preds1 = np.asarray([data.denormalize(pred[sensor][timestep].detach().cpu().numpy()) for pred in predictions1])
    preds2 = np.asarray([data.denormalize(pred[sensor][timestep].detach().cpu().numpy()) for pred in predictions2])
    preds3 = np.asarray([data.denormalize(pred[sensor][timestep].detach().cpu().numpy()) for pred in predictions3])
    preds4 = np.asarray([data.denormalize(pred[sensor][timestep].detach().cpu().numpy()) for pred in predictions4])
    labs = np.asarray([data.denormalize(label[sensor][timestep].cpu().numpy()) for label in labels])
    print("Data points:,", preds1.shape)

    plt.figure(figsize=(30, 10))
    sns.lineplot(data=preds1, label="dcrnn")
    sns.lineplot(data=preds2, label="egcno")
    sns.lineplot(data=preds3, label="egcnh")
    sns.lineplot(data=preds4, label="gclstm")
    sns.lineplot(data=labs, label="true")
    plt.show()


def plot_convergence(mse_list, mse_list2, mse_list3, mse_list4):
    sns.lineplot(data=mse_list, label="dcrnn")
    sns.lineplot(data=mse_list2, label="egcno")
    sns.lineplot(data=mse_list3, label="egcnh")
    sns.lineplot(data=mse_list4, label="gclstm")
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.show()

def get_loss_step_sensor(error_metric):
    if error_metric == "mse":
        loss_step = loss_mse_step_sensor
    elif error_metric == "rmse":
        loss_step = loss_rmse_step_sensor
    elif error_metric == "mae":
        loss_step = loss_mae_step_sensor
    else:
        loss_step = loss_mse_step_sensor
    return loss_step


def get_loss_final_sensor(error_metric):
    if error_metric == "mse":
        loss_final = loss_mse_final_sensor
    elif error_metric == "rmse":
        loss_final = loss_rmse_final_sensor
    elif error_metric == "mae":
        loss_final = loss_mae_final_sensor
    else:
        loss_final = loss_mse_final_sensor
    return loss_final


def loss_mse_step_sensor(y_hat, y):
    return ((y_hat - y) ** 2)


def loss_mse_final_sensor(loss, step):
    return loss / (step + 1)


def loss_rmse_step_sensor(y_hat, y):
    return ((y_hat - y) ** 2)


def loss_rmse_final_sensor(loss, step):
    return math.sqrt(loss / (step + 1))


def loss_mae_step_sensor(y_hat, y):
    return (abs(y - y_hat))


def loss_mae_final_sensor(loss, step):
    return loss / (step + 1)