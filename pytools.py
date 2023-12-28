import torch
import subprocess
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split


def choose_cuda():
    # 使用 nvidia-smi 命令获取 GPU 使用情况和内存信息
    gpu_info = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
        universal_newlines=True)

    # 按行分割使用情况和内存信息
    gpu_usage = gpu_info.strip().split('\n')

    # 构建 GPU 使用率和内存的元组列表
    usage_memory_tuples = [(int(line.split(',')[0]), int(line.split(',')[1]), int(line.split(',')[2])) for line in
                           gpu_usage]

    # 按使用率由低到高排序
    sorted_by_usage = sorted(usage_memory_tuples, key=lambda x: x[1])

    # 取使用率前三个
    top_three_by_usage = sorted_by_usage[:3]

    # 按内存由低到高排序
    sorted_top_three = sorted(top_three_by_usage, key=lambda x: x[2])

    # 输出排序后的 GPU 信息
    # print("按照使用率由低到高排序，使用率前三个按照内存由低到高排序的 GPU：")
    # for index, usage, memory in sorted_top_three:
    #     print(f"GPU {index}: 使用率={usage}%, 内存={memory} MB")
    print(f"GPU {sorted_top_three[0][0]}: 使用率={sorted_top_three[0][1]}%, 内存={sorted_top_three[0][2]} MB")
    cuda = 'cuda:' + str(sorted_top_three[0][0])
    print(cuda)
    return cuda


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, store_path='./model_saved'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.store_path = store_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        # 这里会存储迄今最优模型的参数
        # torch.save({
        #     'loss': val_loss,
        #     'model_state_dict': model.state_dict(),
        # }, self.store_path + f'/checkpoint_{int(time.time())}.pth')
        torch.save(model, self.store_path + f'/checkpoint_best.pth')
        self.val_loss_min = val_loss


def split(data, label, test_size, random_state, split_classes, dataset_sizes):
    '''
        主要计算，以多大的间隔分层dataset_sizes / value
        num_classes:[classes,[other]]
    '''

    value = reduce(lambda x, y: x * y, split_classes)

    array = np.arange(value)  # 生成从 0 到 10 的数组，每个元素都不同

    # 创建一个形状为  的数组，每一行都填充不同的值
    array = np.tile(array, (int(dataset_sizes / value), 1)).T
    array = np.concatenate(array)
    print(array)
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=test_size,
                                                                      random_state=random_state, stratify=array)
    return data_train, data_test, label_train, label_test
