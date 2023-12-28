import argparse
import torch
from pytools import choose_cuda
import torch_npu
'''
导入新的数据集/模型需要处理的步骤

0、修改模型
0、模型加载

1、设置数据集和模型参数SET,添加命令行
2、模型名称=模型+超参
2.5、微调模型名称

3、进行数据预处理

4、debug各模型和数据集匹配
'''


class Set(object):
    '''
    模型和其他都要用到的设置
    '''

    def __init__(self, config):
        self.config = config
        # 选运行设备
        self.device = torch.device(self.config.cuda if torch_npu.npu.is_available() else "cpu")
        torch_npu.npu.set_device(self.device)#必须设置
        # 数据集路径
        if self.config.data_path == 'None':
            if self.config.mode == 'Api':
                self.input_path_data = './Dataset/{}/Input/'.format(self.config.dataset)
            else:
                self.input_path_data = './Dataset/{}/'.format(self.config.dataset)
        else:
            self.input_path_data = self.config.data_path

        ## 构建对应模型的输入、输出及日志路径
        if self.config.debug_size:
            self.output_path_model = "./result/model/"  # 存放最佳模型
            self.output_path_log = "./result/log/"  # 存放训练日志
            self.output_path_record = "./result/record/"  # 存放测试记录
        else:
            self.output_path_model = "./result/model/{}/".format(self.config.model)  # 存放最佳模型
            self.output_path_log = "./result/log/{}/".format(self.config.model)  # 存放训练日志
            if self.config.mode == 'Api':
                self.output_path_record = "./result/record/{}/Api/".format(self.config.model)  # 存放测试记录
            else:
                self.output_path_record = "./result/record/{}/".format(self.config.model)  # 存放测试记录

        # *******1、设置数据集和模型参数SET
        self.split_classes = [1]  # 用于按类别分层划分数据集,列表形式
        if self.config.dataset == 'radar':
            self.num_classes = 7
            self.length = 16252
            self.num_channel = 1
            self.split_classes = [self.num_classes]
            self.file_name_data = 'time_series_16252_7.npy'
            self.file_name_label = 'time_label_16252_7.npy'
            if self.config.model == 'avgnet':
                self.max_nodes = 500
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
            if self.config.model == 'dtsgnet':
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
                self.num_segment = int((self.length - self.config.segment_length) / self.config.unfold_step + 1)
        elif self.config.dataset in ['L2', 'L3', 'Tr_L2', 'L3_1angle_10_30']:
            self.file_name_data = 'Pw_curve.npy'
            self.file_name_label = 'Pw_label.npy'
            self.num_classes = 10
            self.length = 28
            self.num_channel = 1
            self.split_classes = [self.num_classes, 10]
            if self.config.model == 'avgnet':
                self.max_nodes = 28
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
            elif self.config.model == 'dtsgnet':
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
                self.num_segment = int((self.length - self.config.segment_length) / self.config.unfold_step + 1)
            else:
                self.max_nodes = 28
        elif self.config.dataset in ['SW', 'Tr_s', 'Tr_s2', 'S2_1angle_10_30', 'S3_1angle_10_30']:
            self.num_classes = 10
            self.length = 31
            self.num_channel = 1
            self.split_classes = [self.num_classes, 10]
            self.file_name_data = 'Pw_curve.npy'
            self.file_name_label = 'Pw_label.npy'
            if self.config.model == 'avgnet':
                self.max_nodes = 31
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
            if self.config.model == 'dtsgnet':
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
                self.num_segment = int((self.length - self.config.segment_length) / self.config.unfold_step + 1)
        elif self.config.dataset == 'KW':
            self.num_classes = 10
            self.length = 11
            self.num_channel = 1
            self.file_name_data = 'Pw_curve.npy'
            self.file_name_label = 'Pw_label.npy'
            if self.config.model == 'avgnet':
                self.max_nodes = 11
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
            if self.config.model == 'dtsgnet':
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
                self.num_segment = int((self.length - self.config.segment_length) / self.config.unfold_step + 1)
        elif self.config.dataset == 'Kw_f':
            self.num_classes = 10
            self.length = 2996
            self.num_channel = 1
            self.split_classes = [self.num_classes, 10]
            self.file_name_data = 'Kw_f_curve.npy'
            self.file_name_label = 'Kw_f_label.npy'
            if self.config.model == 'avgnet':
                self.max_nodes = 300
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
            if self.config.model == 'dtsgnet':
                self.num_filter = self.config.w  # self.num_filter决定了对角线的条数
                self.num_segment = int((self.length - self.config.segment_length) / self.config.unfold_step + 1)
        else:
            print('ERROR')

        # ********2、数据集的名称
        if self.config.dataset in ['radar', 'L2', 'L3', 'SW', 'KW', 'Tr_s', 'Tr_s2', 'Tr_L2', 'Kw_f', 'L3_1angle_10_30',
                                   'S2_1angle_10_30', 'S3_1angle_10_30']:
            self.dataset_name = "dataset={}_len={}_".format(self.config.dataset, self.length)
        else:
            self.dataset_name = "dataset={}_".format(self.config.dataset)

        # *******3、模型名称 = 模型 + 超参
        '''
        模型名字=模型+加不同的超参设置
        '''
        if self.config.model == 'avgnet':
            self.model_name = '{}_lr={}_nodes={}_w={}_' \
                .format(self.config.model, self.config.lr,
                        self.max_nodes, self.config.w)
        elif self.config.model == 'dtsgnet':
            self.model_name = '{}_lr={}_w={}_segment_length={}_unfold_step={}_' \
                .format(self.config.model, self.config.lr,
                        self.config.w,
                        self.config.segment_length, self.config.unfold_step)
        elif self.config.model == 'ESAVGNET':
            self.model_name = '{}_lr={}_nodes={}_w={}_' \
                .format(self.config.model, self.config.lr,
                        self.config.window, self.config.w)
        else:
            self.model_name = '{}_lr={}_'.format(self.config.model, self.config.lr)

        # *******2.5、微调模型名称
        '''
        不同数据集最后的全连接参数数量不同，所以需要完全相同的数据集
        '''
        self.tune_model = f'dataset={self.config.dataset}_len={2000}_' + self.model_name

        ## 保存文件的名字
        if self.config.debug_size:
            self.file_name_result = "debug_"  # file_name指定所有保存的文件名字
            self.file_name_compare = "debug_"
        else:
            self.file_name_result = self.dataset_name + self.model_name  # file_name指定所有保存的文件名字
            self.file_name_compare = self.dataset_name  # 汇总不同模型的结果


def arg_parse():
    parser = argparse.ArgumentParser(description='Signal_diagonal_matrix_CNN arguments.')
    parser.add_argument('--cuda', dest='cuda', type=str, help='cpu/cuda:0-7', )  # cuda不带：+数字代表使用默认值,
    parser.add_argument('-d', '--dataset', dest='dataset', type=str,
                        help='Dataset: 128, 3040, 3w, Hop, Radar')  # 用来选择使用的数据集（多模型需要修改代码）和嵌入文件名
    parser.add_argument('--model', dest='model', type=str,
                        help='Model: CNN1D, CNN2D, LSTM, GRU, MCLDNN, SigNet, AvgNet, DTSGNet')  # 用来选择使用的模型（多模型需要修改代码）和嵌入文件名
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--num-workers', dest='num_workers', type=int, help='Number of workers to load data.')
    parser.add_argument('--mode', dest='mode', type=str, choices=['Train', 'Test', 'Tune', 'Api'],
                        help='Train or test', )
    parser.add_argument('--debug-size', dest='debug_size', type=int, help='0代表不debug',
                        default=0)  # 程序调试模式，少量样本试跑程序，完整程序最好在终端开screen
    parser.add_argument('--is-early-stop', dest='is_early_stop', type=bool, default=False, help='True/False')  # 提前结束开关位
    parser.add_argument('--data-path', dest='data_path', type=str, default='None')

    parser.set_defaults(mode="Train", debug_size=0, is_early_stop=False, cuda='npu:6',  # 常置于debug模式
                        dataset="S2_1angle_10_30", batch_size=128,
                        model="avgnet",
                        lr=0.001, num_epochs=50, num_workers=8)
    # ---------对应模型相关参数--------
    # GNN
    parser.add_argument('--w', help='The local visible window <= (windows-2)', type=int, default=11)
    # dtsgnet
    parser.add_argument('--segment_length', help='Number of segment nodes (Sliding window size)', type=int, default=100)
    parser.add_argument('--unfold_step', help='The step parameter in the Unfold(),===segment_length', type=int,
                        default=100)
    parser.add_argument('--GNN', help='GNN model, GCN GraphSAGE', type=str, default='GraphSAGE')
    parser.add_argument('--num_layers', help='Number of GNN layers', type=int, default=6)
    parser.add_argument('--RNN', help='RNN model, GRU LSTM', type=str, default='LSTM')
    parser.add_argument('--hidden_channels', help='Size of each hidden layer output sample.', type=int, default=64)
    # ESAVGNET
    parser.add_argument('--window', dest='window', type=int, default=13, help='3,5,7,9,...')
    return parser.parse_args()
