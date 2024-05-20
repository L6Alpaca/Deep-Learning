import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0 , 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
# 定义卷积核--weight 权重

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(kernel)

output = F.conv2d(input, kernel, stride=(1,1),padding = "same")  #stride=(x,y)表示在x，y方向上的步长
print(output)
