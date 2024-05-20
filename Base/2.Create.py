import torch.nn as nn


class mynn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


Apc = mynn()
a = 1
t = Apc(a)
print(t)
