
```python

import torch
from torch import nn

layer = nn.TransformerDecoderLayer(d_model=512, nhead=8,dim_feedforward=2048)
gpt = nn.TransformerDecoder(layer, num_layers=12) # 50M

total_params = sum(
	param.numel() for param in gpt.parameters()
)
print(total_params)

src = torch.rand((10, 1, 512)) # (L, B, D)
out = gpt.forward(tgt=src, memory=src)
print(out.shape)
print(out)
# optimal compute: 1.1B tokens
```
