import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from .utils import makedirpath

__all__ = ['EncoderHier', 'Encoder', 'PositionClassifier'] #层次自编码器？，小自编码器？，位置分类器（自监督学习增强信息提取能力）


class Encoder(nn.Module):
    def __init__(self, K, D=64, bias=True): # 构造函数，用来初始化一些卷积层和几个属性
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, 2, 0, bias=bias) # in_channels, out_channels,kernel_size,stride,padding
        self.conv2 = nn.Conv2d(64, 64, 5, 2, 0, bias=bias) # 输出尺寸除不尽的时候，卷积和池化都是向下取整
        self.conv3 = nn.Conv2d(64, 128, 5, 2, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, D, 5, 1, 0, bias=bias)

        self.K = K
        self.D = D
        self.bias = bias

    def forward(self, x):
        h = self.conv1(x) # x尺寸为1024时候，h为510*510*64的特征图
        h = F.leaky_relu(h, 0.1) # 0.1是leaky_relu的斜率

        h = self.conv2(h) # 输出为253*253*64
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h) # 125*125*128

        if self.K == 64: # 如果该实例的K为64，则用leaky_relu激活上一层输出
            h = F.leaky_relu(h, 0.1)
            h = self.conv4(h) # 输出121*121*D，D默认为64

        h = torch.tanh(h) # 最后一层用tanh激活

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath) # 创建一个'ckpts/{name}/形式的目录，如果已经存在也可
        torch.save(self.state_dict(), fpath) # 在上边创建的目录中存放训练好的权重，这种方式只保存模型参数，不保存网络结构

    def load(self, name): #下载模型参数
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath)) #torch.load用于加载torch.save保存的对象，然后用load_state_dict传给当前网络
        # load_state_dict是父类nn.Module中的方法
    @staticmethod # 静态方法，不需要self和cls参数，就是一个放在类里边的普通函数，在类里边用self调用，类外边也可以直接用类名.方法名调用
    def fpath_from_name(name):
        return f'ckpts/{name}/encoder_nohier.pkl' # 返回一个已经训练好并保存的模型的参数文件的地址，pkl是二进制文件，和pth一样


def forward_hier(x, emb_small, K):
    K_2 = K // 2 # k应该是输入patch的尺寸
    n = x.size(0) # n是通道数，比如3
    x1 = x[..., :K_2, :K_2] #x1234分别是左上，左下，右上，右下
    x2 = x[..., :K_2, K_2:]
    x3 = x[..., K_2:, :K_2]
    x4 = x[..., K_2:, K_2:]
    xx = torch.cat([x1, x2, x3, x4], dim=0) #x是64*64尺寸图像，x1234是32*32，这里应该是默认张量第0维是通道维，这里拼接之后还是32*32，只不过通道数变多了
    hh = emb_small(xx)

    h1 = hh[:n] #用小的自编码器对拼接后的图像处理之后不要改变通道数，然后再把输出分成4个patch对应输出
    h2 = hh[n: 2 * n]
    h3 = hh[2 * n: 3 * n]
    h4 = hh[3 * n:]

    h12 = torch.cat([h1, h2], dim=3)
    h34 = torch.cat([h3, h4], dim=3)
    h = torch.cat([h12, h34], dim=2)
    return h # 返回的通道跟刚开始的x一样，只是已经经过了小的自编码器的编码，


class EncoderDeep(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 0, bias=bias)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0, bias=bias)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 0, bias=bias)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 0, bias=bias)
        self.conv8 = nn.Conv2d(32, D, 3, 1, 0, bias=bias)

        self.K = K
        self.D = D

    def forward(self, x): # 输入x为32*32时候，
        h = self.conv1(x) # 15*15*32
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h) # 13*13*64
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h) # 11*11*128
        h = F.leaky_relu(h, 0.1)

        h = self.conv4(h) # 9*9*128
        h = F.leaky_relu(h, 0.1)

        h = self.conv5(h) # 7*7*64
        h = F.leaky_relu(h, 0.1)

        h = self.conv6(h) # 5*5*32
        h = F.leaky_relu(h, 0.1)

        h = self.conv7(h) # 3*3*32
        h = F.leaky_relu(h, 0.1)

        h = self.conv8(h) # 1*1*64 变成了一个64维的向量
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/encdeep.pkl'


class EncoderHier(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        if K > 64:
            self.enc = EncoderHier(K // 2, D, bias=bias)

        elif K == 64:
            self.enc = EncoderDeep(K // 2, D, bias=bias)

        else:
            raise ValueError()

        self.conv1 = nn.Conv2d(D, 128, 2, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(128, D, 1, 1, 0, bias=bias)

        self.K = K
        self.D = D

    def forward(self, x):
        h = forward_hier(x, self.enc, K=self.K) #得到的h是已经经过小的自编码器处理过的特征图

        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/enchier.pkl'


################


xent = nn.CrossEntropyLoss()


class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class PositionClassifier(nn.Module):
    def __init__(self, K, D, class_num=8):
        super().__init__()
        self.D = D

        self.fc1 = nn.Linear(D, 128)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(128, 128)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc3 = NormalizedLinear(128, class_num)

        self.K = K

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, name):
        return f'ckpts/{name}/position_classifier_K{self.K}.pkl'

    @staticmethod
    def infer(c, enc, batch):
        x1s, x2s, ys = batch

        h1 = enc(x1s)
        h2 = enc(x2s)

        logits = c(h1, h2)
        loss = xent(logits, ys)
        return loss

    def forward(self, h1, h2):
        h1 = h1.view(-1, self.D)
        h2 = h2.view(-1, self.D)

        h = h1 - h2

        h = self.fc1(h)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.act2(h)

        h = self.fc3(h)
        return h

