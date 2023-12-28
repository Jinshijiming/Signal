
import torch
import torch.nn as nn

from args import Set


class lstm2(nn.Module,Set):
    def __init__(self,config):
        nn.Module.__init__(self)
        Set.__init__(self,config)

        self.lstm1 = nn.LSTM(
            input_size=self.num_channel,
            hidden_size=128,
            num_layers=1,
            bias=False,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            bias=False,
            batch_first=True
        )

        self.fc = nn.Linear(self.length*64, self.num_classes)


    def forward(self, x):
        '''
        x, _ = self.lstm1(x.transpose(2,1))  [batch_size, length, 1]
        '''
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = torch.reshape(x, [x.shape[0],-1])
        x = self.fc(x)

        return x

# data = torch.randn(20,2,512)
# model = lstm2()
# print(model(data).shape)

# from torchinfo import summary
# model = lstm2().cuda()
# summary(model, input_size=(128, 2, 128))

