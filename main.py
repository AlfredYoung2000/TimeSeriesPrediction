import os
import random
import argparse
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch

from models import DNN, RNN, LSTM, GRU, RecursiveLSTM, AttentionLSTM, CNN
from utils import make_dirs, load_data, plot_full, data_loader, get_lr_scheduler
from utils import mean_percentage_error, mean_absolute_percentage_error, plot_pred_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(_config):
    random.seed(_config.seed)
    np.random.seed(_config.seed)
    torch.manual_seed(_config.seed)
    torch.cuda.manual_seed(_config.seed)
    paths = [_config.weights_path, _config.plots_path]
    for path in paths:
        make_dirs(path)
    data = load_data(_config.which_data)[[_config.feature]]
    data = data.copy()
    if _config.plot_full:
        plot_full(_config.plots_path, data, _config.feature)
    scaler = MinMaxScaler()
    data[_config.feature] = scaler.fit_transform(data)
    train_loader, val_loader, test_loader = \
        data_loader(data, _config.seq_length, _config.train_split, _config.test_split, _config.batch_size)
    train_losses, val_losses = list(), list()
    val_maes, val_mses, val_rmses, val_mapes, val_mpes, val_r2s = list(), list(), list(), list(), list(), list()
    test_maes, test_mses, test_rmses, test_mapes, test_mpes, test_r2s = list(), list(), list(), list(), list(), list()
    best_val_loss = 100
    best_val_improv = 0
    if _config.network == 'dnn':
        model = DNN(_config.seq_length,
                    _config.hidden_size,
                    _config.output_size).to(device)
    elif _config.network == 'cnn':
        model = CNN(_config.seq_length,
                    _config.batch_size).to(device)
    elif _config.network == 'rnn':
        model = RNN(_config.input_size,
                    _config.hidden_size,
                    _config.num_layers,
                    _config.output_size).to(device)
    elif _config.network == 'lstm':
        model = LSTM(_config.input_size,
                     _config.hidden_size,
                     _config.num_layers,
                     _config.output_size,
                     _config.bidirectional).to(device)
    elif _config.network == 'gru':
        model = GRU(_config.input_size,
                    _config.hidden_size,
                    _config.num_layers,
                    _config.output_size).to(device)
    elif _config.network == 'recursive':
        model = RecursiveLSTM(_config.input_size,
                              _config.hidden_size,
                              _config.num_layers,
                              _config.output_size).to(device)
    elif _config.network == 'attention':
        model = AttentionLSTM(_config.input_size,
                              _config.key,
                              _config.query,
                              _config.value,
                              _config.hidden_size,
                              _config.num_layers,
                              _config.output_size,
                              _config.bidirectional).to(device)
    else:
        raise NotImplementedError

    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=_config.lr, betas=(0.5, 0.999))
    optim_scheduler = get_lr_scheduler(_config.lr_scheduler, optim)
    if _config.mode == 'train':
        print("开始训练{}周期为{}".format(model.__class__.__name__, _config.num_epochs))
        for epoch in range(_config.num_epochs):
            for i, (data, label) in enumerate(train_loader):
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)
                pred = model(data)
                train_loss = criterion(pred, label)
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                train_losses.append(train_loss.item())
            if (epoch + 1) % _config.print_every == 0:
                print("周期：[{}/{}]".format(epoch + 1, _config.num_epochs))
                print("训练损失：{:.4f}".format(np.average(train_losses)))
            optim_scheduler.step()
            with torch.no_grad():
                for i, (data, label) in enumerate(val_loader):
                    data = data.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.float32)
                    pred_val = model(data)
                    val_loss = criterion(pred_val, label)
                    val_mae = mean_absolute_error(label.cpu(), pred_val.cpu())
                    val_mse = mean_squared_error(label.cpu(), pred_val.cpu(), squared=True)
                    val_rmse = mean_squared_error(label.cpu(), pred_val.cpu(), squared=False)
                    val_mpe = mean_percentage_error(label.cpu(), pred_val.cpu())
                    val_mape = mean_absolute_percentage_error(label.cpu(), pred_val.cpu())
                    val_r2 = r2_score(label.cpu(), pred_val.cpu())
                    val_losses.append(val_loss.item())
                    val_maes.append(val_mae.item())
                    val_mses.append(val_mse.item())
                    val_rmses.append(val_rmse.item())
                    val_mpes.append(val_mpe.item())
                    val_mapes.append(val_mape.item())
                    val_r2s.append(val_r2.item())
            if (epoch + 1) % _config.print_every == 0:
                print("损失：\t{:.4f}".format(np.average(val_losses)))
                print("MAE：\t{:.4f}".format(np.average(val_maes)))
                print("MSE：\t{:.4f}".format(np.average(val_mses)))
                print("RMSE：\t{:.4f}".format(np.average(val_rmses)))
                print("MPE：\t{:.4f}".format(np.average(val_mpes)))
                print("MAPE：\t{:.4f}".format(np.average(val_mapes)))
                print("R^2：\t{:.4f}".format(np.average(val_r2s)))
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)
                    torch.save(model.state_dict(),
                               os.path.join(_config.weights_path, 'Best_{}.pkl'.format(model.__class__.__name__)))
                    print("最优模型已保存\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += 1
                    print("最优验证劣于{}\n".format(best_val_improv))

    elif _config.mode == 'test':
        model.load_state_dict(
            torch.load(os.path.join(_config.weights_path, 'Best_{}.pkl'.format(model.__class__.__name__))))
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)
                pred_test = model(data)
                pred_test = pred_test.data.cpu().numpy()
                label = label.data.cpu().numpy().reshape(-1, 1)
                pred_test = scaler.inverse_transform(pred_test)
                label = scaler.inverse_transform(label)
                test_mae = mean_absolute_error(label, pred_test)
                test_mse = mean_squared_error(label, pred_test, squared=True)
                test_rmse = mean_squared_error(label, pred_test, squared=False)
                test_mpe = mean_percentage_error(label, pred_test)
                test_mape = mean_absolute_percentage_error(label, pred_test)
                test_r2 = r2_score(label, pred_test)
                test_maes.append(test_mae.item())
                test_mses.append(test_mse.item())
                test_rmses.append(test_rmse.item())
                test_mpes.append(test_mpe.item())
                test_mapes.append(test_mape.item())
                test_r2s.append(test_r2.item())

            print("测试{}".format(model.__class__.__name__))
            print("测试\tMAE：\t{:.4f}".format(np.average(test_maes)))
            print("测试\tMSE：\t{:.4f}".format(np.average(test_mses)))
            print("测试\tRMSE：\t{:.4f}".format(np.average(test_rmses)))
            print("测试\tMPE：\t{:.4f}".format(np.average(test_mpes)))
            print("测试\tMAPE：\t{:.4f}".format(np.average(test_mapes)))
            print("测试\tR^2：\t{:.4f}".format(np.average(test_r2s)))
            plot_pred_test(pred_test, label, _config.plots_path, _config.feature, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--feature', type=str, default='Appliances', help='extract which feature for prediction')
    parser.add_argument('--seq_length', type=int, default=5, help='window size')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--network', type=str, default='dnn',
                        choices=['dnn', 'cnn', 'rnn', 'lstm', 'gru', 'recursive', 'attention'])
    parser.add_argument('--input_size', type=int, default=1, help='input_size')
    parser.add_argument('--hidden_size', type=int, default=10, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--output_size', type=int, default=1, help='output_size')
    parser.add_argument('--bidirectional', type=bool, default=False, help='use bidirectional or not')
    parser.add_argument('--key', type=int, default=8, help='key')
    parser.add_argument('--query', type=int, default=8, help='query')
    parser.add_argument('--value', type=int, default=8, help='value')
    parser.add_argument('--which_data', type=str, default='./data/energydata_complete.csv', help='which data to use')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--plots_path', type=str, default='./results/plots/', help='plots path')
    parser.add_argument('--train_split', type=float, default=0.8, help='train_split')
    parser.add_argument('--test_split', type=float, default=0.5, help='test_split')
    parser.add_argument('--num_epochs', type=int, default=200, help='total epoch')
    parser.add_argument('--print_every', type=int, default=10, help='print statistics for every default epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler',
                        choices=['step', 'plateau', 'cosine'])
    parser.add_argument('--plot_full', type=bool, default=False, help='plot full graph or not')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()
    torch.cuda.empty_cache()
    main(config)
