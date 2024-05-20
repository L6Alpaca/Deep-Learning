import matplotlib.pyplot as plt
import torch.cuda
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as ds
from torch.utils.data import DataLoader
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_data = ds.MNIST("MNIST", True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

#查看train_loader 属性，为(1,28,28)
# 设置生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.main(x)
        img = img.view(-1, 28, 28, 1)
        return img


#设置判别器，中间用LeakyRelu()好
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = img.view(-1, 28 * 28)
        x = self.main(x)
        return x


device = 'cuda' #设置device


# generator绘画
def gen_img_plot(model, epoch, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())   # squeeze——用于删除shape中的单维度
                                                                        # detach()——原本这个张量是和其他部分连在一起参与计算和梯度传播的，
                                                                        # 用了detach()后，它就变得独立了，和之前的计算过程没啥关系了，也不会再受梯度计算的影响。
    fig = plt.figure(figsize=(4, 4))
    num = str(epoch)
    plt.title("Epoch:"+num,fontsize = 10)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2)
        plt.axis("off")
    plt.show()


gen = Generator().to(device)
dis = Discriminator().to(device)
d_optimizer = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0001)
loss = nn.BCELoss()

test_input = torch.randn(16, 100, device=device)

# 循环训练
D_epoch = []
G_epoch = []
for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(train_loader)
    for step, (img, _) in enumerate(train_loader):  #img——Tensor型，(64,1,28,28)最后一个只有32,_表示img对应y值0~9，这里只用到图像值
        img = img.to(device)
        size = img.size(0)  #取第一维的值64||32
        random_noise = torch.randn(size, 100, device=device)
        # Discriminatetor操作
        d_optimizer.zero_grad()
        real_output = dis(img)  # 判别器输入真实图片
        d_real_loss = loss(real_output,
                           torch.ones_like(real_output))
        d_real_loss.backward()
        gen_img = gen(random_noise)  # 由噪音生成图片
        fake_output = dis(gen_img.detach())  # 判别器输入生成的图片
        d_fake_loss = loss(fake_output,
                           torch.zeros_like(fake_output))
        d_fake_loss.backward()
        d_loss = d_fake_loss + d_real_loss
        d_optimizer.step()
        #对discriminator优化

        g_optimizer.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss(fake_output, torch.ones_like(fake_output))
        g_loss.backward()
        g_optimizer.step()
        #对generator优化

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_epoch.append(d_epoch_loss)
        G_epoch.append(g_epoch_loss)
        print("Epoch:", epoch)
        gen_img_plot(gen, epoch + 1, test_input)
