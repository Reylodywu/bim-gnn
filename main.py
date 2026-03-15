import logging
import time
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping, opt
from model import models

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# import nni

# 设置和配置一些环境变量，以确保实验的稳定性和可重复性
def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)  # 指定GPU设备

    # 设置伪随机数生成器的种子，确保在不同运行之间获得相同的随机数
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  # 确保运行时不会对卷积算法进行优化，提高稳定性
    torch.backends.cudnn.deterministic = True  # 启用确定性的CUDA操作，以确保相同输入下的结果相同
    # torch.use_deterministic_algorithms(True)


# 解析命令行参数，配置模型的各种超参数，以及确定是否使用CUDA
def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=32, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='full_model_10_devc', choices=['full_model_(1、3、5、10)_devc',"single_1_devc"],)
    parser.add_argument('--target_type', type=str, default='temperature',choices=['temperature', 'co_ppm', 'soot_ugm3'],help='预测目标类型：温度/CO/烟尘')
    parser.add_argument('--lstm_hidden_size', type=int, default=128,help='LSTM隐藏层大小')
    parser.add_argument('--n_his', type=int, default=12)  # 历史时间步数
    parser.add_argument('--n_pred', type=int, default=3,help='the number of time interval for predcition, default as 3')  # 预测时间步数
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)  # 时间卷积核大小
    parser.add_argument('--stblock_num', type=int, default=2)  # 时空卷积块的数量
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])  # 激活函数类型
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])  # 空间卷积核大小
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv',
                        choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap',
                        choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.3)  # dropout率
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='nadamw', choices=['adamw', 'nadamw', 'lion'],
                        help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--model_type', type=str, default='improved',choices=['origin','improved'])

    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='测试集高斯噪声标准差 (例如 1.0 代表 1度/1ppm 的标准差)')
    parser.add_argument('--sparsity_rate', type=float, default=0.0, help='测试集数据缺失率 (0.0 到 1.0)')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
        torch.cuda.empty_cache()  # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect()  # Clean cache

    # Ko: 模型将使用多少有效的时间步进行预测
    # 每经过1个时空卷积块，时间步减少(args.Kt - 1) * 2
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: 定义模型中时空卷积块的结构和输出层的通道数
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])  # 输入层
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])  # 每个时空卷积块，有64个输入通道、16个瓶颈通道和64个输出通道
    if Ko == 0:
        blocks.append([128])  # Ko=0的输出层，模型的最终输出是一个单一的预测值，因为模型不能使用足够的历史时间步进行预测
    elif Ko > 0:
        blocks.append([128, 128])  # Ko>0的输出层，要充分利用保留下来的历史信息并允许更复杂的输出
    blocks.append([1])  # 最终输出层

    return args, device, blocks

# 数据扰动
def apply_perturbation(data, noise_std=0.0, sparsity_rate=0.0):
    """
    向真实数据中注入传感器噪声和数据稀疏性
    :param data: 原始 numpy 数组
    :param noise_std: 高斯噪声标准差（模拟传感器测量误差，如温度漂移）
    :param sparsity_rate: 数据丢失率 0~1 之间（模拟传感器故障或通信丢包）
    :return: 扰动后的数据
    """
    perturbed_data = data.copy()

    # 1. 添加传感器噪声 (高斯噪声)
    if noise_std > 0:
        # 生成与数据同形状的高斯噪声
        noise = np.random.normal(loc=0.0, scale=noise_std, size=perturbed_data.shape)
        perturbed_data += noise

    # 2. 添加数据稀疏性 (随机丢包)
    if sparsity_rate > 0:
        # 生成掩码 (True表示保留，False表示丢失)
        mask = np.random.rand(*perturbed_data.shape) > sparsity_rate

        # 对于时间序列的丢失，简单的做法是将其置为前一时刻的值（前向填充），
        # 这里用 pandas 快速实现前向填充，更符合真实传感器的"信号保持"特性
        df = pd.DataFrame(np.where(mask, perturbed_data, np.nan))
        perturbed_data = df.fillna(method='ffill').fillna(0).values  # 如果第一行也是nan则补0

    return perturbed_data

# 数据准备
def data_preparate(args, device):
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'temperature.csv')).shape[0]

    val_and_test_rate = 0.15
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    print('len_train:', len_train, 'len_val:', len_val, 'len_test:', len_test)

    # 获取原始数据
    train, val, test, vel_all = dataloader.load_data(args.dataset, args.target_type, len_train, len_val)
    if args.noise_std > 0 or args.sparsity_rate > 0:
        print(f"注意: 正在进行鲁棒性测试! Noise STD: {args.noise_std}, Sparsity: {args.sparsity_rate}")
        train = apply_perturbation(train, noise_std=args.noise_std, sparsity_rate=args.sparsity_rate)
        # test = apply_perturbation(test, noise_std=args.noise_std, sparsity_rate=args.sparsity_rate)
    # 保存原始数据用于迭代预测
    original_train = train.copy()
    original_val = val.copy()
    original_test = test.copy()
    original_all = vel_all.copy()

    # 标准化
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)
    vel_all = zscore.transform(vel_all)

    # 数据转换
    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)
    x_all, y_all = dataloader.data_transform(vel_all, args.n_his, args.n_pred, device)
    # print("x_test.shape:", x_test.shape, "y_test.shape:", y_test.shape)

    # 创建数据迭代器
    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    all_data = utils.data.TensorDataset(x_all, y_all)
    all_iter = utils.data.DataLoader(dataset=all_data, batch_size=args.batch_size, shuffle=False)

    # 返回原始数据，用于迭代预测
    raw_data_dict = {
        'train': original_train,
        'val': original_val,
        'test': original_test,
        'all': original_all
    }

    return n_vertex, zscore, train_iter, val_iter, test_iter, all_iter, raw_data_dict


# 准备神经网络模型
def prepare_model(args, blocks, n_vertex, device):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(delta=0.0,
                                     patience=args.patience,
                                     verbose=True,
                                     path=f"{args.model_type}_{args.target_type}_{args.dataset}_{args.n_his}_{args.n_pred}.pt")
    # 选择使用的模型
    if args.graph_conv_type == 'cheb_graph_conv':
        if args.model_type == 'origin':
            print("Using original STGCN model with ChebGraphConv")
            # 使用 ChebGraphConv 的原始 STGCN 模型
            model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
        else:
            # 使用改进的 ChebGraphConv 的 STGCN 模型
            model = models.STGCN_LSTM_Fusion_Vectorized(
                args=args,
                blocks=blocks,
                n_vertex=n_vertex,
                lstm_hidden_dim=args.lstm_hidden_size,  # LSTM隐藏层大小
                lstm_layers=2
            ).to(device)

        # model = models.STGCN_LSTM_Temporal_Advanced(
        #     args=args,
        #     blocks=blocks,
        #     n_vertex=n_vertex,
        #     lstm_hidden_dim=args.lstm_hidden_size,  # LSTM隐藏层大小
        #     lstm_layers=2
        # ).to(device)

    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    # 选择优化器
    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate,
                                decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler


def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter):
    train_losses = []  # 记录每个epoch的训练损失
    epochs = []  # 记录epoch数

    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            # y_pred = model(x).view(len(x), -1)
            y_pred = model(x).reshape(len(x), -1)
            y = y.reshape(len(x), -1)
            # print("y_pred.shape:", y_pred.shape, "y.shape:", y.shape)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        train_loss = l_sum / n
        train_losses.append(train_loss)
        epochs.append(epoch + 1)

        scheduler.step()
        val_loss = val(model, val_iter, loss)

        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'. \
              format(epoch + 1, optimizer.param_groups[0]['lr'], train_loss, val_loss, gpu_mem_alloc))

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            break

    # # 绘制训练损失曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


# 在不更新模型参数的情况下查看模型在验证数据上的表现
@torch.no_grad()
def val(model, val_iter, loss):
    model.eval()

    l_sum, n = 0.0, 0
    for x, y in val_iter:
        # y_pred = model(x).view(len(x), -1)
        y_pred = model(x).reshape(len(x), -1)
        y = y.reshape(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)


@torch.no_grad()
def _test(zscore, loss, model, test_iter, args):
    # model.load_state_dict(torch.load("STGCN_" + args.dataset + str(args.n_pred) +".pt"))
    # 使用 f-string (推荐，更清晰)
    print(args.model_type)
    model.load_state_dict(torch.load(f"{args.model_type}_{args.target_type}_{args.dataset}_{args.n_his}_{args.n_pred}.pt"))
    model.eval()

    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)

    print(
        f'Dataset {args.dataset:s} | Test loss {test_MSE:.4f} | MAE {test_MAE:.4f} |WMAPE {test_WMAPE:.4f} | RMSE {test_RMSE:.4f} ')
    return test_MAE, test_WMAPE, test_RMSE

# def test_iterative_forecast_improved(args, model, device, test_data, zscore, start_pos,n_total_pred):
#     """
#     改进的迭代预测：真正使用预测值进行迭代
#     """
#     import matplotlib.pyplot as plt
#
#     # 设置字体
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['mathtext.fontset'] = 'stix'
#
#     # 转换为numpy数组
#     if hasattr(test_data, 'values'):
#         test_data = test_data.values
#
#     # 标准化
#     test_data_norm = zscore.transform(test_data)
#     n_vertex = test_data.shape[1]
#
#     # 参数
#     n_his = args.n_his
#     n_pred = args.n_pred
#     n_total_pred = n_total_pred  # 总共要预测的步数
#
#     print(f"迭代预测参数:")
#     print(f"  历史窗口: {n_his}")
#     print(f"  每次预测: {n_pred}")
#     print(f"  总预测步数: {n_total_pred}")
#     print(f"  起始位置: {start_pos}")
#
#     # 初始化
#     current_window = test_data_norm[start_pos:start_pos + n_his].copy()
#     all_predictions = []
#
#     # 真正的迭代预测
#     model.eval()
#     with torch.no_grad():
#         n_iterations = (n_total_pred + n_pred - 1) // n_pred
#
#         for i in range(n_iterations):
#             # 准备输入
#             x_input = current_window.reshape(1, 1, n_his, n_vertex)
#             x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
#
#             # 预测
#             output = model(x_tensor)
#
#             # 处理输出维度
#             if output.shape[-1] == 1:
#                 output = output.squeeze(-1)
#             if output.ndim == 4 and output.shape[1] == 1:
#                 output = output.squeeze(1)
#
#             pred = output.cpu().numpy().squeeze()  # 可能是 (n_pred, n_vertex) 或 (n_vertex,)
#
#             # 关键修复：确保pred是2D数组
#             if pred.ndim == 1:  # 当n_pred=1时，pred是(n_vertex,)
#                 pred = pred.reshape(1, -1)  # 转为(1, n_vertex)
#
#             print(f"第{i + 1}次迭代 - pred.shape: {pred.shape}")  # 调试信息
#
#             # 确定这次迭代要取多少步
#             remaining_steps = n_total_pred - i * n_pred
#             steps_to_take = min(n_pred, remaining_steps)
#             current_pred = pred[:steps_to_take]
#
#             all_predictions.append(current_pred)
#
#             # # 更新窗口：移除前n_pred步，添加新预测的n_pred步
#             # if remaining_steps > n_pred:
#             #     current_window = np.vstack([
#             #         current_window[n_pred:],  # 移除前n_pred步
#             #         pred[:n_pred]  # 添加预测的n_pred步
#             #     ])
#             if i < n_iterations - 1:  # ✅ 修正条件
#                 current_window = np.vstack([
#                     current_window[n_pred:],  # 移除前n_pred步
#                     pred[:n_pred]  # 添加预测的n_pred步
#                 ])
#     # 合并所有预测
#     predictions = np.vstack(all_predictions)  # [n_total_pred, n_vertex]
#
#     # 获取真实值
#     true_values = test_data_norm[start_pos + n_his:start_pos + n_his + n_total_pred]
#
#     # 反标准化
#     predictions_orig = zscore.inverse_transform(predictions)
#     true_values_orig = zscore.inverse_transform(true_values)
#
#     # 计算每步误差（沿用你的方法）
#     mae_list = []
#     wmape_list = []
#     rmse_list = []
#
#     epsilon = 1e-10
#     for t in range(len(predictions_orig)):
#         mae = np.mean(np.abs(predictions_orig[t] - true_values_orig[t]))
#         wmape = np.mean(np.abs(true_values_orig[t] - predictions_orig[t]) /
#                         (np.abs(true_values_orig[t]) + epsilon)) * 100
#         rmse = np.sqrt(np.mean((predictions_orig[t] - true_values_orig[t]) ** 2))
#
#         mae_list.append(mae)
#         wmape_list.append(wmape)
#         rmse_list.append(rmse)
#
#     # 打印结果
#     overall_mae = np.mean(mae_list)
#     overall_wmape = np.mean(wmape_list)
#     overall_rmse = np.mean(rmse_list)
#
#     print(f"\nIterative Prediction Results (starting from position {start_pos},with{args.n_his}_{args.n_pred}):")
#     print(f"Overall MAE: {overall_mae:.4f}")
#     print(f"Overall WMAPE: {overall_wmape:.4f}%")
#     print(f"Overall RMSE: {overall_rmse:.4f}")
#
#     # 可视化
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
#     time_steps = np.arange(len(mae_list))
#
#     # MAE
#     ax1.plot(time_steps, mae_list, 'b-o', label='MAE', markersize=4)
#     ax1.axhline(y=overall_mae, color='b', linestyle='--', alpha=0.7,
#                 label=f'Overall MAE: {overall_mae:.4f}')
#     ax1.set_title('MAE over Time')
#     ax1.set_xlabel('Time Step')
#     ax1.set_ylabel('MAE')
#     ax1.grid(True)
#     ax1.legend()
#
#     # WMAPE
#     ax2.plot(time_steps, wmape_list, 'r-o', label='WMAPE', markersize=4)
#     ax2.axhline(y=overall_wmape, color='r', linestyle='--', alpha=0.7,
#                 label=f'Overall WMAPE: {overall_wmape:.4f}%')
#     ax2.set_title('WMAPE over Time')
#     ax2.set_xlabel('Time Step')
#     ax2.set_ylabel('WMAPE (%)')
#     ax2.grid(True)
#     ax2.legend()
#
#     # RMSE
#     ax3.plot(time_steps, rmse_list, 'g-o', label='RMSE', markersize=4)
#     ax3.axhline(y=overall_rmse, color='g', linestyle='--', alpha=0.7,
#                 label=f'Overall RMSE: {overall_rmse:.4f}')
#     ax3.set_title('RMSE over Time')
#     ax3.set_xlabel('Time Step')
#     ax3.set_ylabel('RMSE')
#     ax3.grid(True)
#     ax3.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#     # 保存图片
#     fig.savefig(f"iterative_{args.model_type}_{args.dataset}_{n_total_pred}.png")
#
#     return predictions_orig, {'mae': mae_list, 'wmape': wmape_list, 'rmse': rmse_list}

# def test_iterative_forecast_improved(args, model, device, test_data, zscore, start_pos, n_total_pred):
#     """
#     改进的迭代预测：真正使用预测值进行迭代
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # 设置字体
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['mathtext.fontset'] = 'stix'
#     plt.rcParams['font.size'] = 12
#
#     # 转换为numpy数组
#     if hasattr(test_data, 'values'):
#         test_data = test_data.values
#
#     # 标准化
#     test_data_norm = zscore.transform(test_data)
#     n_vertex = test_data.shape[1]
#
#     # 参数
#     n_his = args.n_his
#     n_pred = args.n_pred
#     n_total_pred = n_total_pred  # 总共要预测的步数
#
#     print(f"迭代预测参数:")
#     print(f"  历史窗口: {n_his}")
#     print(f"  每次预测: {n_pred}")
#     print(f"  总预测步数: {n_total_pred}")
#     print(f"  起始位置: {start_pos}")
#
#     # 初始化
#     current_window = test_data_norm[start_pos:start_pos + n_his].copy()
#     all_predictions = []
#
#     # 真正的迭代预测
#     model.eval()
#     with torch.no_grad():
#         n_iterations = (n_total_pred + n_pred - 1) // n_pred
#
#         for i in range(n_iterations):
#             # 准备输入
#             x_input = current_window.reshape(1, 1, n_his, n_vertex)
#             x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
#
#             # 预测
#             output = model(x_tensor)
#
#             # 处理输出维度
#             if output.shape[-1] == 1:
#                 output = output.squeeze(-1)
#             if output.ndim == 4 and output.shape[1] == 1:
#                 output = output.squeeze(1)
#
#             pred = output.cpu().numpy().squeeze()  # 可能是 (n_pred, n_vertex) 或 (n_vertex,)
#
#             # 关键修复：确保pred是2D数组
#             if pred.ndim == 1:  # 当n_pred=1时，pred是(n_vertex,)
#                 pred = pred.reshape(1, -1)  # 转为(1, n_vertex)
#
#             print(f"第{i + 1}次迭代 - pred.shape: {pred.shape}")  # 调试信息
#
#             # 确定这次迭代要取多少步
#             remaining_steps = n_total_pred - i * n_pred
#             steps_to_take = min(n_pred, remaining_steps)
#             current_pred = pred[:steps_to_take]
#
#             all_predictions.append(current_pred)
#
#             if i < n_iterations - 1:  # ✅ 修正条件
#                 current_window = np.vstack([
#                     current_window[n_pred:],  # 移除前n_pred步
#                     pred[:n_pred]  # 添加预测的n_pred步
#                 ])
#
#     # 合并所有预测
#     predictions = np.vstack(all_predictions)  # [n_total_pred, n_vertex]
#
#     # 获取真实值
#     true_values = test_data_norm[start_pos + n_his:start_pos + n_his + n_total_pred]
#
#     # 反标准化
#     predictions_orig = zscore.inverse_transform(predictions)
#     true_values_orig = zscore.inverse_transform(true_values)
#
#     # 计算每步每个节点的误差
#     epsilon = 1e-10
#     all_errors = []  # 存储所有误差，用于箱线图
#     mae_list = []
#     wmape_list = []
#     rmse_list = []
#
#     for t in range(len(predictions_orig)):
#         # 计算每个节点在时间步t的MAPE
#         node_mapes = np.abs(true_values_orig[t] - predictions_orig[t]) / (np.abs(true_values_orig[t]) + epsilon) * 100
#         all_errors.append(node_mapes)
#
#         # 计算整体指标
#         mae = np.mean(np.abs(predictions_orig[t] - true_values_orig[t]))
#         wmape = np.mean(node_mapes)
#         rmse = np.sqrt(np.mean((predictions_orig[t] - true_values_orig[t]) ** 2))
#
#         mae_list.append(mae)
#         wmape_list.append(wmape)
#         rmse_list.append(rmse)
#
#     # 转换为numpy数组便于处理
#     all_errors = np.array(all_errors)  # [n_total_pred, n_vertex]
#
#     # 打印结果
#     overall_mae = np.mean(mae_list)
#     overall_wmape = np.mean(wmape_list)
#     overall_rmse = np.mean(rmse_list)
#
#     print(f"\nIterative Prediction Results (starting from position {start_pos}, with {args.n_his}_{args.n_pred}):")
#     print(f"Overall MAE: {overall_mae:.4f}")
#     print(f"Overall WMAPE: {overall_wmape:.4f}%")
#     print(f"Overall RMSE: {overall_rmse:.4f}")
#
#     # 创建箱线图可视化
#
#     create_boxplot_visualization(all_errors, overall_wmape, args, n_total_pred)
#
#     return predictions_orig, {'mae': mae_list, 'wmape': wmape_list, 'rmse': rmse_list}
#
# def create_boxplot_visualization(all_errors, overall_wmape, args, n_total_pred):
#     """
#     创建完整显示的箱线图可视化 - 修正标签和说明
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import math
#
#     # 设置字体和显示参数
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['mathtext.fontset'] = 'stix'
#     plt.rcParams['font.size'] = 12
#
#     # 创建图形
#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#
#     # 准备箱线图数据
#     n_steps = min(12, n_total_pred)
#     boxplot_data = []
#
#     for t in range(n_steps):
#         if t < len(all_errors):
#             boxplot_data.append(all_errors[t])
#         else:
#             boxplot_data.append(all_errors[-1])
#
#     # 动态计算y轴范围
#     all_max_values = []
#     for errors in boxplot_data:
#         q75 = np.percentile(errors, 75)
#         iqr = np.percentile(errors, 75) - np.percentile(errors, 25)
#         upper_whisker = q75 + 1.5 * iqr
#         actual_max = np.max(errors[errors <= upper_whisker]) if len(
#             errors[errors <= upper_whisker]) > 0 else upper_whisker
#         all_max_values.append(max(upper_whisker, actual_max))
#
#     max_y_value = max(all_max_values)
#     y_limit = max(7, math.ceil(max_y_value * 1.1))
#
#     # 创建箱线图
#     box_props = dict(facecolor='lightgray', color='black', linewidth=1.5)
#     whisker_props = dict(color='black', linewidth=1.5)
#     cap_props = dict(color='black', linewidth=1.5)
#     median_props = dict(color='blue', linewidth=2.5)
#
#     bp = ax.boxplot(boxplot_data,
#                     positions=range(1, n_steps + 1),
#                     patch_artist=True,
#                     boxprops=box_props,
#                     whiskerprops=whisker_props,
#                     capprops=cap_props,
#                     medianprops=median_props,
#                     showfliers=False,
#                     widths=0.6)
#
#     # 设置y轴 - 修正标签为WMAPE
#     ax.set_ylim(0, y_limit)
#     y_ticks = range(0, y_limit + 1)
#     ax.set_yticks(y_ticks)
#     ax.set_ylabel('Prediction error (WMAPE)', fontsize=14, fontweight='bold')  # 修正为WMAPE
#
#     # 设置x轴
#     ax.set_xlim(0.5, n_steps + 0.5)
#     ax.set_xticks(range(1, n_steps + 1))
#     ax.set_xlabel('Future steps', fontsize=14, fontweight='bold')
#
#     # 添加网格
#     ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
#     ax.set_axisbelow(True)
#
#     # 计算中位数WMAPE
#     median_wmape = np.median([np.median(errors) for errors in boxplot_data])
#
#     # 修正：添加带说明的标注
#     # 整体WMAPE标注 - 添加说明文字
#     ax.text(0.95, 0.95, f'Overall WMAPE\n{overall_wmape:.2f}%',
#             transform=ax.transAxes,
#             fontsize=14,
#             color='red',
#             fontweight='bold',
#             ha='right', va='top',
#             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
#                       edgecolor='red', alpha=0.9))
#
#     # 中位数WMAPE标注 - 添加说明文字
#     ax.text(0.95, 0.05, f'Median WMAPE\n{median_wmape:.2f}%',
#             transform=ax.transAxes,
#             fontsize=14,
#             color='red',
#             fontweight='bold',
#             ha='right', va='bottom',
#             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
#                       edgecolor='red', alpha=0.9))
#
#     # 添加图例
#     from matplotlib.patches import Patch
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Patch(facecolor='lightgray', edgecolor='black', label='25%-75%'),
#         Line2D([0], [0], color='black', linewidth=1.5, label='1.5IQR'),
#         Line2D([0], [0], color='blue', linewidth=2.5, label='Median')
#     ]
#     legend = ax.legend(handles=legend_elements, loc='upper left',
#                        frameon=True, facecolor='white', edgecolor='black',
#                        fontsize=12, framealpha=0.9)
#
#     # 设置标题
#     model_name = getattr(args, 'model_type', 'improved')
#     dataset_name = getattr(args, 'dataset', 'full_model_10_devc')
#     title = f'({model_name})-{dataset_name}'
#     ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
#
#     # 确保完整显示
#     plt.tight_layout()
#     plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.1)
#
#     # 保存图片
#     filename = f"boxplot_{model_name}_{dataset_name}_{n_total_pred}.png"
#     plt.savefig(filename, dpi=300, bbox_inches='tight',
#                 facecolor='white', edgecolor='none')
#
#     plt.show()
#
#     print(f"箱线图已保存为: {filename}")
#     print(f"Overall WMAPE: {overall_wmape:.2f}%")
#     print(f"Median WMAPE: {median_wmape:.2f}%")
#
#     return fig, ax

def test_iterative_forecast_improved(args, model, device, test_data, zscore, start_pos, n_total_pred):
    """
    改进的迭代预测：真正使用预测值进行迭代，并生成多指标箱线图
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 12

    # 转换为numpy数组
    if hasattr(test_data, 'values'):
        test_data = test_data.values

    # 标准化
    test_data_norm = zscore.transform(test_data)
    n_vertex = test_data.shape[1]

    # 参数
    n_his = args.n_his
    n_pred = args.n_pred
    n_total_pred = n_total_pred  # 总共要预测的步数

    print(f"迭代预测参数:")
    print(f"  历史窗口: {n_his}")
    print(f"  每次预测: {n_pred}")
    print(f"  总预测步数: {n_total_pred}")
    print(f"  起始位置: {start_pos}")

    # 初始化
    current_window = test_data_norm[start_pos:start_pos + n_his].copy()
    all_predictions = []

    # 真正的迭代预测
    model.eval()
    with torch.no_grad():
        n_iterations = (n_total_pred + n_pred - 1) // n_pred

        for i in range(n_iterations):
            # 准备输入
            x_input = current_window.reshape(1, 1, n_his, n_vertex)
            x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)

            # 预测
            output = model(x_tensor)

            # 处理输出维度
            if output.shape[-1] == 1:
                output = output.squeeze(-1)
            if output.ndim == 4 and output.shape[1] == 1:
                output = output.squeeze(1)

            pred = output.cpu().numpy().squeeze()  # 可能是 (n_pred, n_vertex) 或 (n_vertex,)

            # 关键修复：确保pred是2D数组
            if pred.ndim == 1:  # 当n_pred=1时，pred是(n_vertex,)
                pred = pred.reshape(1, -1)  # 转为(1, n_vertex)

            print(f"第{i + 1}次迭代 - pred.shape: {pred.shape}")  # 调试信息

            # 确定这次迭代要取多少步
            remaining_steps = n_total_pred - i * n_pred
            steps_to_take = min(n_pred, remaining_steps)
            current_pred = pred[:steps_to_take]

            all_predictions.append(current_pred)

            if i < n_iterations - 1:  # ✅ 修正条件
                current_window = np.vstack([
                    current_window[n_pred:],  # 移除前n_pred步
                    pred[:n_pred]  # 添加预测的n_pred步
                ])

    # 合并所有预测
    predictions = np.vstack(all_predictions)  # [n_total_pred, n_vertex]

    # 获取真实值
    true_values = test_data_norm[start_pos + n_his:start_pos + n_his + n_total_pred]

    # 反标准化
    predictions_orig = zscore.inverse_transform(predictions)
    true_values_orig = zscore.inverse_transform(true_values)

    # 计算三种误差指标
    epsilon = 1e-10
    mae_errors = []  # 存储每步每个节点的MAE
    wmape_errors = []  # 存储每步每个节点的WMAPE
    rmse_errors = []  # 存储每步每个节点的RMSE

    mae_list = []  # 存储每步的平均MAE
    wmape_list = []  # 存储每步的平均WMAPE
    rmse_list = []  # 存储每步的平均RMSE

    for t in range(len(predictions_orig)):
        pred_t = predictions_orig[t]
        true_t = true_values_orig[t]

        # 计算每个节点的误差
        mae_nodes = np.abs(pred_t - true_t)
        wmape_nodes = np.abs(pred_t - true_t) / (np.abs(true_t) + epsilon) * 100
        rmse_nodes = (pred_t - true_t) ** 2  # 先平方，后面开根号

        mae_errors.append(mae_nodes)
        wmape_errors.append(wmape_nodes)
        rmse_errors.append(np.sqrt(rmse_nodes))  # 对每个节点开根号

        # 计算该时间步的平均指标
        mae_list.append(np.mean(mae_nodes))
        wmape_list.append(np.mean(wmape_nodes))
        rmse_list.append(np.sqrt(np.mean(rmse_nodes)))

    # 计算整体指标
    overall_mae = np.mean(mae_list)
    overall_wmape = np.mean(wmape_list)
    overall_rmse = np.mean(rmse_list)

    # 打印结果
    print(f"\nIterative Prediction Results (starting from position {start_pos}, with {args.n_his}_{args.n_pred}):")
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall WMAPE: {overall_wmape:.4f}%")
    print(f"Overall RMSE: {overall_rmse:.4f}")

    # 创建三指标箱线图可视化
    def create_multi_metric_boxplots():
        # 创建三个子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 定义三种误差数据和相关参数
        error_data = [mae_errors, wmape_errors, rmse_errors]
        error_names = ['MAE', 'WMAPE', 'RMSE']
        error_units = ['', '(%)', '']
        overall_values = [overall_mae, overall_wmape, overall_rmse]
        colors = ['green', 'red', 'blue']

        for idx, (errors, name, unit, overall_val, color) in enumerate(zip(
                error_data, error_names, error_units, overall_values, colors)):

            ax = axes[idx]
            n_steps = len(errors)

            # 动态计算y轴范围
            all_max_values = []
            for error_list in errors:
                if len(error_list) > 0:
                    q75 = np.percentile(error_list, 75)
                    iqr = np.percentile(error_list, 75) - np.percentile(error_list, 25)
                    upper_whisker = q75 + 1.5 * iqr
                    actual_max = np.max(error_list[error_list <= upper_whisker]) if len(
                        error_list[error_list <= upper_whisker]) > 0 else upper_whisker
                    all_max_values.append(max(upper_whisker, actual_max))

            max_y_value = max(all_max_values) if all_max_values else 1
            if name == 'WMAPE':
                y_limit = max(7, math.ceil(max_y_value * 1.1))
            else:
                y_limit = math.ceil(max_y_value * 1.2)

            # 创建箱线图
            box_props = dict(facecolor='lightgray', color='black', linewidth=1.5)
            whisker_props = dict(color='black', linewidth=1.5)
            cap_props = dict(color='black', linewidth=1.5)
            median_props = dict(color=color, linewidth=2.5)

            bp = ax.boxplot(errors,
                            positions=range(1, n_steps + 1),
                            patch_artist=True,
                            boxprops=box_props,
                            whiskerprops=whisker_props,
                            capprops=cap_props,
                            medianprops=median_props,
                            showfliers=False,
                            widths=0.6)

            # 设置y轴
            ax.set_ylim(0, y_limit)
            ax.set_ylabel(f'Prediction error ({name}){unit}', fontsize=12, fontweight='bold')

            # 设置x轴
            ax.set_xlim(0.5, n_steps + 0.5)
            ax.set_xticks(range(1, n_steps + 1))
            ax.set_xlabel('Future steps', fontsize=12, fontweight='bold')

            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            ax.set_axisbelow(True)

            # 计算中位数
            median_val = np.median([np.median(error_list) for error_list in errors])

            # 添加整体指标标注
            if name == 'WMAPE':
                ax.text(0.95, 0.95, f'Overall {name}\n{overall_val:.2f}%',
                        transform=ax.transAxes,
                        fontsize=12,
                        color=color,
                        fontweight='bold',
                        ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=color, alpha=0.9))

                ax.text(0.95, 0.85, f'Median {name}\n{median_val:.2f}%',
                        transform=ax.transAxes,
                        fontsize=12,
                        color=color,
                        fontweight='bold',
                        ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=color, alpha=0.9))
            else:
                ax.text(0.95, 0.95, f'Overall {name}\n{overall_val:.3f}',
                        transform=ax.transAxes,
                        fontsize=12,
                        color=color,
                        fontweight='bold',
                        ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=color, alpha=0.9))

                ax.text(0.95, 0.85, f'Median {name}\n{median_val:.3f}',
                        transform=ax.transAxes,
                        fontsize=12,
                        color=color,
                        fontweight='bold',
                        ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=color, alpha=0.9))

            # 添加图例 (只在第一个子图添加)
            if idx == 0:
                from matplotlib.patches import Patch
                from matplotlib.lines import Line2D
                legend_elements = [
                    Patch(facecolor='lightgray', edgecolor='black', label='25%-75%'),
                    Line2D([0], [0], color='black', linewidth=1.5, label='1.5IQR'),
                    Line2D([0], [0], color=color, linewidth=2.5, label='Median')
                ]
                ax.legend(handles=legend_elements, loc='upper left',
                          frameon=True, facecolor='white', edgecolor='black',
                          fontsize=10, framealpha=0.9)

            # 设置子图标题
            ax.set_title(f'{name} Error Distribution', fontsize=14, fontweight='bold')

        # 设置总标题
        model_name = getattr(args, 'model_type', 'improved')
        dataset_name = getattr(args, 'dataset', 'full_model_10_devc')
        fig.suptitle(f'({model_name})-{dataset_name} - Multi-Metric Analysis',
                     fontsize=16, fontweight='bold')

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # 保存图片
        filename = f"multi_metric_boxplot_{model_name}_{dataset_name}_{n_total_pred}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        plt.show()

        print(f"多指标箱线图已保存为: {filename}")

    # 调用箱线图生成函数
    create_multi_metric_boxplots()

    return predictions_orig, {
        'mae': mae_list,
        'wmape': wmape_list,
        'rmse': rmse_list,
        'mae_errors': mae_errors,
        'wmape_errors': wmape_errors,
        'rmse_errors': rmse_errors
    }

def run_single_model(model_type):
    """运行单个模型并返回结果"""
    args, device, blocks = get_parameters()
    args.model_type = model_type

    n_vertex, zscore, train_iter, val_iter, test_iter, all_iter, raw_data = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex, device)

    # 如果需要训练，取消注释下面这行
    # train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)

    _test(zscore, loss, model, train_iter, args)
    predictions, errors = test_iterative_forecast_improved(
        args, model, device, raw_data['train'], zscore, start_pos=100, n_total_pred=12)

    return predictions, errors


def plot_model_comparison(results):
    """
    绘制多个模型的比较图
    results: dict, 格式为 {'model_name': {'predictions': ..., 'errors': ...}}
    """
    # 创建模型名称映射
    model_name_mapping = {
        'Origin Model': 'STGCN-only',
        'Improved Model': 'LSTM-STGCN'
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    colors = ['blue', 'red', 'green', 'orange']

    for idx, (model_name, model_results) in enumerate(results.items()):
        errors = model_results['errors']
        color = colors[idx % len(colors)]

        # 获取显示用的标签名称
        display_name = model_name_mapping.get(model_name, model_name)

        # 提取数据 - errors是字典，包含'mae', 'wmape', 'rmse'键
        mae_values = errors['mae']
        wmape_values = errors['wmape']
        rmse_values = errors['rmse']

        # 计算总体指标
        overall_mae = np.mean(mae_values)
        overall_wmape = np.mean(wmape_values)
        overall_rmse = np.mean(rmse_values)

        time_steps = range(len(mae_values))

        # MAE over Time
        axes[0].plot(time_steps, mae_values, 'o-', color=color, label=f'{display_name}', linewidth=2)
        axes[0].axhline(y=overall_mae, color=color, linestyle='--', alpha=0.7,
                        label=f'{display_name} Overall: {overall_mae:.4f}')

        # WMAPE over Time
        axes[1].plot(time_steps, wmape_values, 'o-', color=color, label=f'{display_name}', linewidth=2)
        axes[1].axhline(y=overall_wmape, color=color, linestyle='--', alpha=0.7,
                        label=f'{display_name} Overall: {overall_wmape:.4f}%')

        # RMSE over Time
        axes[2].plot(time_steps, rmse_values, 'o-', color=color, label=f'{display_name}', linewidth=2)
        axes[2].axhline(y=overall_rmse, color=color, linestyle='--', alpha=0.7,
                        label=f'{display_name} Overall: {overall_rmse:.4f}')

    # 设置图表属性
    axes[0].set_title('MAE over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('MAE')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title('WMAPE over Time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('WMAPE (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title('RMSE over Time', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('RMSE')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# # 在主函数中调用
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=UserWarning)
#
#     # 运行两个模型
#     results = {}
#
#     print("Running Origin Model...")
#     pred_origin, err_origin = run_single_model('origin')
#     results['Origin Model'] = {'predictions': pred_origin, 'errors': err_origin}
#
#     print("Running Improved Model...")
#     pred_improved, err_improved = run_single_model('improved')
#     results['Improved Model'] = {'predictions': pred_improved, 'errors': err_improved}
#
#     # 绘制比较图
#     plot_model_comparison(results)


# if __name__ == "__main__":
#     # Logging
#     # logger = logging.getLogger('stgcn')
#     # logging.basicConfig(filename='stgcn.log', level=logging.INFO)
#     logging.basicConfig(level=logging.INFO)
#
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=UserWarning)
#
#     args, device, blocks = get_parameters()
#     # 修改调用
#     n_vertex, zscore, train_iter, val_iter, test_iter, all_iter, raw_data = data_preparate(args, device)
#     loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex, device)
#     train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)
#     _test(zscore, loss, model, test_iter, args)
#
#     # predictions, errors = test_iterative_forecast_improved(args, model, device, raw_data['train'], zscore, start_pos=100,n_total_pred=12)
#
#     print("迭代预测测试完成!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()

    # 设定测试参数
    noise_levels = [0.0, 1.0, 2.0, 5.0]
    sparsity_levels = [0.0, 0.1, 0.3]

    # 用于存储结果的列表
    results_summary = []

    print("\n========= 开始训练集(Train) 【单步预测】 鲁棒性测试 =========")
    for noise in noise_levels:
        for sparsity in sparsity_levels:
            args.noise_std = noise
            args.sparsity_rate = sparsity

            # 重新制备带扰动的数据
            # (确保 data_preparate 中已经按上一条回答加入了对 train 注入扰动的逻辑)
            n_vertex, zscore, train_iter, val_iter, test_iter, all_iter, raw_data = data_preparate(args, device)
            loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex, device)

            print(f"\n[运行中] 测试条件: 噪声 Std={noise}, 缺失率={sparsity}")

            # 评估【单步预测】误差（传入 train_iter），并接收返回的指标
            test_MAE, test_WMAPE, test_RMSE = _test(zscore, loss, model, train_iter, args)

            # 存入汇总列表
            results_summary.append({
                'noise': noise,
                'sparsity': sparsity,
                'mae': test_MAE,
                'wmape': test_WMAPE,
                'rmse': test_RMSE
            })

    # ================== 自动生成 Markdown 表格 ==================
    print("\n\n" + "=" * 50)
    print("✅ 实验完成！以下为您生成的 【单步预测】 Markdown 格式结果表格：")
    print("=" * 50 + "\n")

    print("| 噪声标准差 (Noise Std) | 丢失率 (Sparsity) | MAE | WMAPE (%) | RMSE |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    for res in results_summary:
        print(
            f"| {res['noise']:.1f} | {res['sparsity']:.1f} | {res['mae']:.4f} | {res['wmape']:.4f} | {res['rmse']:.4f} |")

    print("\n" + "=" * 50)



