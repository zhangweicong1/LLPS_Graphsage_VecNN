import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VecNN(nn.Module):
    def __init__(self, rna_embed_len=740, graphsage_embedding=1):
        super(VecNN, self).__init__()
        # 根据输入向量的长度调整卷积层的大小
        if graphsage_embedding == 1:
            if rna_embed_len == 740:
                self.rna_conv_len = 8 * ((rna_embed_len - 2) // 2)

        if graphsage_embedding == 0:
            if rna_embed_len == 640:
                self.rna_conv_len = 8 * ((rna_embed_len - 2) // 2)

        self.conv1d_rna = nn.Conv1d(1, 8, kernel_size=3) # 添加一维卷积层
        # self.relu1 = nn.ReLU()
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.rna_layer = nn.Sequential(
            nn.ReLU(),
            # nn.Linear(5104, 2048),
            # nn.Linear(512, 2048),
            # nn.Linear(2952, 2048),
            # nn.Linear(648, 2048),
            nn.Linear(self.rna_conv_len, 1024),
            nn.ReLU(),
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, rnas_input):
        rnas_input = self.conv1d_rna(rnas_input.unsqueeze(1))
        rnas_input = self.pool(rnas_input)
        rnas_input = self.flatten(rnas_input)
        # print(rnas_input)
        #print(rnas_input.shape)

        rnas_output = self.rna_layer(rnas_input)
        output = self.output_layer(rnas_output)

        return output
