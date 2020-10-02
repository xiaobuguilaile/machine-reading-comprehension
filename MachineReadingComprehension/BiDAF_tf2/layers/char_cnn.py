# -*-coding:utf-8 -*-

'''
@File       : char_cnn.py
@Author     : HW Shen
@Date       : 2020/10/2
@Desc       :
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

OPTIONS = {
    'kernel_sizes': [7, 7, 3, 3, 3, 3],
    'pool_sizes': [3, 3, -1, -1, -1, 3],
    'n_kernels': 256
}


class CharCNN(nn.Module):

    def __init__(self, output_dim, in_feature_dim, hidden_size, max_length=1014, pool_size=3, drop_out=0.5):

        super(CharCNN, self).__init__()

        model_options = OPTIONS
        print(model_options)
        self.kernel_sizes = model_options['kernel_sizes']
        self.pool_sizes = model_options['pool_sizes']
        self.n_filters = model_options['n_kernels']
        self.pool_size = pool_size
        self.output_dim = int((max_length - 96) / 27)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=in_feature_dim if i == 0 else self.n_filters,
                                    out_channels=self.n_filters,
                                    kernel_size=self.kernel_sizes[i]),
                          nn.ReLU(),
                          ) for i in range(len(self.kernel_sizes))])

        self.linear1 = nn.Linear(self.n_filters * self.output_dim, hidden_size)
        self.drop_out1 = nn.Dropout(drop_out)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.drop_out2 = nn.Dropout(drop_out)

        self.linear3 = nn.Linear(hidden_size, output_dim)

    def forward(self, inputs):

        bs = inputs.shape[0]
        output = inputs
        output = output.permute(0, 2, 1)
        for i, layer in enumerate(self.conv_layers):
            output = layer(output)
            if i not in [2, 3, 4]:
                output = F.max_pool1d(output, kernel_size=self.pool_size, stride=self.pool_size)

        output = output.permute(0, 2, 1)
        output = output.contiguous().view(bs, -1)
        output = self.linear1(output)
        output = torch.relu(output)
        output = self.drop_out1(output)
        output = self.linear2(output)
        output = torch.relu(output)
        output = self.drop_out2(output)

        output = self.linear3(output)

        return output


if __name__ == '__main__':
    cnn = CharCNN(output_dim=2, in_feature_dim=70, hidden_size=1024)
    temp = torch.randn(size=(8, 1014, 70))
    print(cnn)

    output = cnn(temp)
    print(output)
