import os
import time
import csv
import copy
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from thop import profile
import scipy.io as scio

from model import lstm,avgnet
from pytools import *
from args import *
import sys
import subprocess
from apex import amp

sys.path.extend(['/usr/local/Ascend/ascend-toolkit/latest/python/site-packages',
                 '/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe'])


class Multi_Model_Dataset(Set):
    def __init__(self, config):
        super().__init__(config)

    def data_preprocess(self,label_test=None):
        '''
        争取把不同数据集通过本函数统一格式输出,[样本数，特征的维度，特征/通道123]
        :return:
        '''
        if self.config.mode=='Api':
            data = list(scio.loadmat(self.input_path_data + 'Pw' + '.mat').values())[3]
            data = np.reshape(data, (-1, self.length))# 【B，L】
        else:
            data = np.load(self.input_path_data + self.file_name_data)  # 【B，L】
            label = np.load(self.input_path_data + self.file_name_label)  # 【B，1】
        data = np.reshape(data, [data.shape[0], -1, 1])  # 【B，L,F】
        print('data.shape:', data.shape)

        if self.config.debug_size:
            if label_test is None:
                self.split_classes=[1] #取消分层划分
                data, _, label, _ = train_test_split(data, label, train_size=self.config.debug_size,
                                                 random_state=1)  # 整数代表绝对值，舍掉test部分即可实现小量化
            else:
                data_train, _,label_train,_=train_test_split(data_train, label_train, train_size=self.config.debug_size,
                                                     random_state=1)  # 整数代表绝对值，舍掉test部分即可实现小量化
                data_test, _,label_test,_=train_test_split(data_test, label_test, train_size=self.config.debug_size//3,
                                                     random_state=1)  # 整数代表绝对值，舍掉test部分即可实现小量化
        #个别方法预处理
        if self.config.model == 'avgnet':
            sampling_interval = int(self.length / self.max_nodes)
            sample = [a for a in range(0, self.length, sampling_interval)]
            # data = data[:, sample,: ]
        elif self.config.model == 'dtsgnet':
            # DTSG将提前分段
            data = torch.from_numpy(data).type(torch.FloatTensor)  # PYG输入为[N, Length, Channel]形式,
            data = data.unfold(dimension=1, size=self.config.segment_length,
                               step=self.config.unfold_step)  # [N,seg,Length,Channel]
            print('data',data.shape)
            # data = data.permute(0, 1, 3, 2) #dtsg应该是要修正一下维度了
            data = data.numpy()

        #划分数据集
        if self.config.mode == 'Api':
            data_test = data
            label_test = np.empty([data.shape[0]])
            data_train = np.empty([1, data.shape[1], data.shape[2]])
            label_train = np.empty(1)
            # print('data_train.shape:', data_train.shape)
            # print('label_train.shape:', label_train.shape)
        elif label_test is not None:
            print('No split')
        else:
            data_train, data_test, label_train, label_test = split(data, label, test_size=0.25, random_state=2,
                                                                   split_classes=self.split_classes,
                                                                   dataset_sizes=len(label))

            # np.save(self.input_path_data + 'Input/' + 'Pw_curve_test.npy', data_test)  # 保存完后需要移动到Input/
            # np.save(self.input_path_data + 'Input/' + 'Pw_label_test.npy', label_test)
            # print('data_test:', data_test)
            # print('label_test:', label_test)
        return data_train, data_test, label_train, label_test

    def modelloader(self):
        '''
        把不同的模型统一格式输出
        :return:
        '''
        if self.config.model == 'lstm':
            model = lstm.lstm2(self.config).to(self.device)
        elif self.config.model == 'avgnet':
            model = avgnet.Net(self.config).to(self.device)
        elif self.config.model == 'CNN1D':
            model = CNN1D.ResNet1D(self.config).to(self.device)
        elif self.config.model == 'CNN2D':
            model = CNN2D.CNN2D(self.config).to(self.device)
        elif self.config.model == 'MCLDNN':
            model = MCLDNN.MCLDNN(self.config).to(self.device)
        elif self.config.model == 'dtsgnet':
            model = dtsgnet.ModelCSharedAdj(self.config).to(self.device)
        elif self.config.model == 'gru':
            model = gru.gru2(self.config).to(self.device)
        elif self.config.model == 'ESAVGNET':
            model = ESAVGNET.Net(self.config).to(self.device)
        return model


class Save_Data(Set):
    def __init__(self, config):
        super().__init__(config)
        # **********保存相关***********

        if not os.path.exists(self.output_path_record):
            os.makedirs(self.output_path_record)
        if not os.path.exists(self.output_path_model):
            os.makedirs(self.output_path_model)
        if not os.path.exists(self.output_path_log):
            os.makedirs(self.output_path_log)

        if self.config.mode == 'Train':
            with open(self.output_path_log + self.file_name_result + 'epoch_log.csv', 'w',
                      newline='') as t:  # 创建表头，只保留最新一次的训练记录
                writer_train = csv.writer(t)
                # writer_train.writerow(args_information)
                writer_train.writerow(['epoch', 'phase', 'epoch_loss', 'epoch_acc', 'best_acc'])

        if not os.path.exists(
                "./result/record/" + self.file_name_compare + 'total_record.csv'):  # 创建表头，保留模型所有的测试记录，同一模型不覆盖
            with open("./result/record/" + self.file_name_compare + 'total_record.csv', 'w', newline='') as t1:
                writer_train1 = csv.writer(t1)
                writer_train1.writerow(
                    ['model', 'epoch_acc', 'total_precision', 'total_recall', 'total_f1score',
                     'flops', 'params'])

    # 训练阶段用
    def save_train_epoch_log(self, epoch, phase, epoch_loss, epoch_acc, best_acc):
        # 保存  当前一轮  的训练集精度、测试集精度和最高测试集精度
        with open(self.output_path_log + self.file_name_result + 'epoch_log.csv', 'a',
                  newline='') as t1:
            writer_train1 = csv.writer(t1)
            writer_train1.writerow([epoch, phase, epoch_loss, epoch_acc, best_acc])

    def save_train_total_log(self, flops, params, time_elapsed, best_status):
        ####  记录整体表现：训练时间、最好精度。 ###
        with open(self.output_path_log + self.file_name_result + 'total_log.csv', 'a', newline='') as t1:
            writer_train1 = csv.writer(t1)
            # writer_train1.writerow(args_information)
            writer_train1.writerow(['flops:', flops, 'params', params])
            writer_train1.writerow(['Training Time:', '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)])
            writer_train1.writerow(best_status)

    def save_model(self, best_model_wts):
        torch.save(best_model_wts, self.output_path_model + self.file_name_result + 'bestmodel.pth')

    # 测试阶段用
    def save_test_total_record(self, epoch_acc, total_precision, total_recall, total_f1score, flops, params):
        #########  记录整体表现：训练时间、最好精度。 ######
        with open("./result/record/" + self.file_name_compare + 'total_record.csv', 'a', newline='') as t1:
            writer_train1 = csv.writer(t1)
            writer_train1.writerow(
                [self.model_name, epoch_acc.item(), total_precision.item(), total_recall.item(), total_f1score.item(),
                 flops / (1000 ** 3), params / (1000 ** 2)])

    def save_test_pred_record(self, labels_outputs):
        '''
        第一列是标签，其他列代表神经网络的输出
        :param labels_outputs:
        :return:
        '''
        # label_names = ["label", "0", "1", "2"]   后续分析不需要标题
        labels_outputs = torch.cat(labels_outputs, dim=0).tolist()  # 转多个tensor到标注数字
        df_pred = pd.DataFrame(data=labels_outputs)
        df_pred.to_csv(self.output_path_record + self.file_name_result + 'pred_record.csv', encoding='utf-8', index=False)

    def save_api_pred_record(self,preds):
        with open(self.output_path_record + self.file_name_result + 'pred_record.csv', 'a',
                  newline='') as t1:
            writer_train1 = csv.writer(t1)
            preds=preds.tolist()
            writer_train1.writerow(preds)

class Main(Save_Data, Multi_Model_Dataset):
    def __init__(self, config):
        '''
            接收命令行全全局参数
            保存路径
            训练的相关设置
        :param config:
        '''
        super(Main, self).__init__(config)
        print(
            f'************** dataset:{self.config.dataset} model:{self.config.model} 调试：{self.config.debug_size}****************')
        print(self.config)
        print('device:', self.device)
        # 加载模型，加载数据loader
        self.model = self.modelloader()
        self.batch_signal, self.dataset_sizes = self.dataloader()

        # 交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.lr)
        # 学习率衰减
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.6)
        # EarlyStopping
        self.early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.00001, store_path=self.output_path_model)
        #amp
        # self.model,self.optimizer=amp.initialize(self.model, self.optimizer, opt_level='O1')
    def train(self):
        since = time.time()  # 开始训练的时间
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        #### 一轮训练和测试开始 #####
        for epoch in range(self.config.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.config.num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    # scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corrects = 0
                signal_num = 0
                # Iterate over data.
                pbar = tqdm(self.batch_signal[phase])
                for inputs, labels in pbar:
                    # print(type(inputs))
                    inputs = inputs.to(self.device).float()  # (batch_size, 2, 128)，没有float会出参数类型不一致的问题
                    labels = labels.to(self.device)  # (batch_size, )
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    ##### 一次迭代，进行数据实时显示  #####
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    signal_num += inputs.size(0)
                    # 在进度条的右边实时显示数据集类型、loss值和精度
                    epoch_loss = running_loss / signal_num
                    epoch_acc = running_corrects.double() / signal_num
                    pbar.set_postfix({'Set': '{}'.format(phase),
                                      'Loss': '{:.4f}'.format(epoch_loss),
                                      'Acc': '{:.4f}'.format(epoch_acc)})
                    # print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), end=' ')

                ##### 一轮迭代，进行数据统计 #####
                # 显示该轮的loss和精度
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # 调整学习率或者提前结束测试
                if phase == 'train':
                    self.scheduler.step()
                else:
                    self.early_stopping(epoch_loss, self.model)
                # 复制模型参数 准备保存工作
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_status = [self.file_name_result, best_acc]

                # # 保存  当前一轮  的训练集精度、测试集精度和最高测试集精度
                self.save_train_epoch_log(epoch, phase, epoch_loss, epoch_acc, best_acc)

            # 保存测试精度  最高时的  模型参数
            self.save_model(best_model_wts)
            print('Best test Acc: {:4f}'.format(best_acc))
            print()
            ### 收敛则中断 ####
            if self.early_stopping.early_stop and self.config.is_early_stop:
                print('Early stopping')
                break
        ###############################

        time_elapsed = time.time() - since
        flops, params = profile(self.model, inputs=(inputs,), verbose=True)
        ####  记录整体表现：训练时间、最好精度。 ###
        self.save_train_total_log(flops, params, time_elapsed, best_status)

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))

        # # load best model weights
        del inputs
        del labels
        # self.model.load_state_dict(best_model_wts, strict=False)
        self.test()
        return self.model

    def test(self):
        self.model.load_state_dict(torch.load(self.output_path_model + self.file_name_result + 'bestmodel.pth'), strict=False)

        since = time.time()  # 开始测试的时间
        labels_outputs = []
        for epoch in range(1):
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['test']:
                self.model.eval()  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                signal_num = 0
                # Iterate over data.
                pbar = tqdm(self.batch_signal[phase])  # batch_signal是字典
                # test_acc = torchmetrics.Accuracy()
                metric_collection = torchmetrics.MetricCollection([
                    torchmetrics.Accuracy(average='macro', num_classes=self.num_classes, task='multiclass'),
                    torchmetrics.Recall(average='macro', num_classes=self.num_classes, task='multiclass'),
                    torchmetrics.Precision(average='macro', num_classes=self.num_classes, task='multiclass'),
                    torchmetrics.F1Score(average='macro', num_classes=self.num_classes, task='multiclass'),
                ]).to(self.device)

                for inputs, labels in pbar:
                    inputs = inputs.to(self.device).float()  # (batch_size, 2, 128)
                    labels = labels.to(self.device)  # (batch_size, )

                    with torch.set_grad_enabled(phase == 'train'):  # 每过一次网络保存一次梯度，这个内存消耗就非常大了
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                    labels_outputs.append(
                        torch.cat((labels.reshape(labels.shape[0], 1), F.softmax(outputs, dim=1)),
                                  dim=1))  # 拼接的张量必须维度相同
                    ##### 一次迭代，进行数据实时显示  #####
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    signal_num += inputs.size(0)

                    metric_collection(outputs, labels)
                    # 在进度条的右边实时显示数据集类型、loss值和精度
                    epoch_loss = running_loss / signal_num
                    epoch_acc = running_corrects.double() / signal_num
                    pbar.set_postfix({'Set': '{}'.format(phase),
                                      'Loss': '{:.4f}'.format(epoch_loss),
                                      'Acc': '{:.4f}'.format(epoch_acc)})
                ###################
                metric = metric_collection.compute()
                # 显示该轮的loss和精度
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print('-' * 10)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                print("评价指标:", metric)

        ####################
        time_elapsed = time.time() - since
        print('-' * 10)
        print('Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        flops, params = profile(self.model, inputs=(inputs,), verbose=True)
        print('-' * 10)
        print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
        print("---|---|---")
        print("%s | %.2f | %.2f" % (self.config.model, params / (1000 ** 2), flops / (1000 ** 3)))
        # 记录预测结果和标签,用于后续指标分析
        self.save_test_pred_record(labels_outputs)
        #########  记录整体表现：训练时间、最好精度。 ######
        self.save_test_total_record(metric['MulticlassAccuracy'], metric['MulticlassPrecision'],
                                    metric['MulticlassRecall'], metric['MulticlassF1Score'],
                                    flops, params)
        return self.model

    def tune(self):
        print('开始微调，微调模型名称：{}'.format(self.tune_model))
        self.model.load_state_dict(torch.load(self.output_path_model + self.file_name_result + 'bestmodel.pth'), strict=False)
        self.train()

    def api(self):
        self.model.load_state_dict(torch.load(self.output_path_model + self.file_name_result + 'bestmodel.pth'), strict=False)

        since = time.time()  # 开始测试的时间
        labels_outputs = []
        for epoch in range(1):
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['test']:
                self.model.eval()  # Set model to evaluate mode
                # Iterate over data.
                pbar = tqdm(self.batch_signal[phase])  # batch_signal是字典


                for inputs, labels in pbar:
                    inputs = inputs.to(self.device).float()  # (batch_size, 2, 128)
                    labels = labels.to(self.device)  # (batch_size, )

                    with torch.set_grad_enabled(phase == 'train'):  # 每过一次网络保存一次梯度，这个内存消耗就非常大了
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        # loss = self.criterion(outputs, labels)

                    labels_outputs.append(
                        torch.cat((labels.reshape(labels.shape[0], 1), outputs), dim=1))  # 拼接的张量必须维度相同
                print('preds：', preds.tolist())


        ####################
        time_elapsed = time.time() - since
        print('-' * 10)
        print('Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # 记录预测结果和标签,用于后续指标分析
        self.save_api_pred_record(preds)
        #########  记录整体表现：训练时间、最好精度。 ######
        return self.model

    def dataloader(self):

        data_train, data_test, label_train, label_test = self.data_preprocess()

        # 训练集和测试集的数量
        dataset_sizes = {'train': len(label_train), 'test': len(label_test)}
        # 变张量
        data_train = torch.from_numpy(data_train)
        data_test = torch.from_numpy(data_test)
        label_train = torch.from_numpy(label_train)
        label_test = torch.from_numpy(label_test)
        # 放入数据库
        dataset_train = torch.utils.data.TensorDataset(data_train, label_train)
        dataset_test = torch.utils.data.TensorDataset(data_test, label_test)

        # 数据库——数据加载器
        dataloader = {'train': torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.config.batch_size,
                                                           shuffle=True, num_workers=self.config.num_workers),
                      'test': torch.utils.data.DataLoader(dataset=dataset_test, batch_size=self.config.batch_size,
                                                          shuffle=False, num_workers=self.config.num_workers)}

        return dataloader, dataset_sizes

    def run(self):
        if self.config.mode == 'Train':
            # 训练模型
            self.train()
        elif self.config.mode == 'Test':
            # 导入预训练模型
            # self.model.load_state_dict(
            #     torch.load(self.output_path_model + self.file_name + 'bestmodel.pth'))
            self.test()
        elif self.config.mode == 'Tune':
            self.tune()
        elif self.config.mode == 'Api':
            self.api()


if __name__ == '__main__':
    arg = arg_parse()
    main = Main(arg)
    main.run()
