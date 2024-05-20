import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Apc(nn.Module):
    def __init__(self):
        super(Apc, self).__init__()
        self.conv1 = Conv2d(3, 6, 2, 1, padding=0)
#padding可以设为"valid"——不填充 或 "same"——自动全0填充，步长为1的时候，输出和输入图片尺寸一致

    def forward(self, x):
        x = self.conv1(x)
        return x


# print(dataloader)
writer = SummaryWriter("../../Dataset/nn/logs")
apc = Apc()
step = 0
for data in dataloader:
    imgs, targets = data
    output = apc(imgs)
    print(output.shape)
    output = torch.reshape(output, (-1,3,31,31))
    #print(output.shape)
    writer.add_images("input", imgs, step)
    writer.add_images("Conv2d", output, global_step=step)
    step = step + 1
