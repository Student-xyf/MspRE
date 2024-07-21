import torch
import torch.nn as nn


class SimAM_Module(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM_Module, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "SimAM_Module"

    def forward(self, x):
        # 输入张量 x 的维度为 [batch_size, seq_length, hidden_size]
        b, seq_len, hidden_size = x.size()

        # 计算 x 的平均值
        x_mean = x.mean(dim=1, keepdim=True)

        # 计算 x 与平均值的差的平方
        x_minus_mu_square = (x - x_mean).pow(2)

        # 计算分母部分
        denominator = (4 * (x_minus_mu_square.sum(dim=1, keepdim=True) / (seq_len - 1) + self.e_lambda))

        # 计算 y
        y = x_minus_mu_square / denominator + 0.5

        # 使用 Sigmoid 激活函数
        y = self.activation(y)

        # 对输入张量进行注意力加权
        weighted_x = x * y

        return weighted_x
