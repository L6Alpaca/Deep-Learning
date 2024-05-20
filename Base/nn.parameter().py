'''
nn.Parameter()是PyTorch中用于定义可学习参数（权重）的类。
可学习参数是深度学习模型中需要通过反向传播进行优化的变量。
在模型训练的过程中，模型会根据损失函数对这些参数进行调整，从而使模型能够更好地拟合训练数据。
'''
import torch
import torch.nn as nn

# 创建一个权重张量
weight = torch.randn(3, 4)  #生成的权重是随机的,服从标准正态分布的

# 将其包装为 nn.Parameter
param_weight = nn.Parameter(weight) #初始时，weight==param_weight

# 可以像普通张量一样操作这个参数
print(weight,param_weight)

'''
在实际训练过程中，权重（如 `param_weight` ）会通过优化算法（如随机梯度下降等）根据计算得到的梯度来进行修改。
在每次迭代中，优化算法会根据损失函数对权重的梯度，按照一定的规则（如沿梯度反方向并结合学习率等因素）来调整权重的值，使其逐渐趋向于最优解，从而实现模型的学习和优化。
随着训练的进行，这些权重会不断被更新和调整，以适应对数据的学习和拟合。
'''