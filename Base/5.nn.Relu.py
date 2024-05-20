import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Apc(nn.Module):
    def __init__(self):
        super(Apc, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    # ReLU——小于0计0，大于等于零计自身值
    # Sigmoid——小于0计0，大于等于0计比自身小的数
    def forward(self, input):
        output = self.sigmoid(input)
        return output


apc = Apc()
step = 0
writer = SummaryWriter("../../Dataset/nn/Relu")
for data in dataloader:
    imgs, targets = data
    writer.add_images("imgs", imgs, global_step=step)
    output = apc(imgs)
    writer.add_images("Sigmoid", output, step)
    step = step + 1
