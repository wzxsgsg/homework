import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
import numpy as np

# 包含了一个隐含层的全联机神经网络
class NeuralNet(nn.Module):
    # 输入数据的维度，中间层的节点数，输出数据的维度
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.bert = BertModel.from_pretrained('../homework/bert-pytorch/bert-base-uncased/bert-base-uncased/')
        self.embed = nn.Embedding(48, 1)
        self.l1 = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.tanh = nn.Tanh()
        self.l3 = nn.Linear(hidden_size_2, num_classes)

    def findSolution(self, data):
        def F(x, y):
            return 3 * (1 - x) ** 2 * torch.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * torch.exp(
                -x ** 2 - y ** 2) - 1 / 3 ** torch.exp(-(x + 1) ** 2 - y ** 2)

        pred = []
        for i in range(len(data)):
            pred.append(F(data[i][0], data[i][1]))
        pred = torch.tensor(pred)
        pred = (pred - torch.min(pred)) + 1e-3
        max_index = torch.argmax(pred)
        return data[max_index]

    def forward(self, data):
        data = self.embed(data)
        # _, data = self.bert(data)
        out = self.l1(data.squeeze(2))
        out = self.relu(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)
        out = self.findSolution(out)
        return out