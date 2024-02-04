import torch
from flash_attention import SpatialTransformer
atten = SpatialTransformer(32, 1, 1, 40)

x = torch.ones(1,32,16,16)
c = torch.ones(1,32,40)
y = atten(x, c)
print(y.shape)
