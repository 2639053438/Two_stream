import torch

x = torch.rand(13,101)
x1 = torch.rand(13,101)
y0 = torch.cat((x, x1),1)
y1 = torch.nn.Softmax(dim=0)(y0)

print(y1)