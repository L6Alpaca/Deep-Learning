import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Apc(nn.Module):
    def __init__(self):
        super(Apc, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)
apc = Apc()
writer = SummaryWriter("../CNN/logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = apc(imgs)
    writer.add_images("input", imgs, global_step=step)
    writer.add_images("Maxpool", output, global_step=step)
    step = step + 1

print(dataloader)
#tensorboard打开SummaryWriter
#tensorboard --logdir "Deep Learning/CNN/logs"
