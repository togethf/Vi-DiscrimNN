import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5
from thop import profile

# 1. 加载模型（可选 pretrained）
model = shufflenet_v2_x0_5(pretrained=True)

# 2. 替换分类头
model.fc = nn.Linear(model.fc.in_features, 2)

# 2. 构造输入
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 统计 FLOPs 和 Params
flops, params = profile(model, inputs=(dummy_input,), verbose=False)

print("========== ShuffleNetV2 x0.5 ==========")
print(f"Params: {params/1e6:.3f} M")
print(f"FLOPs: {flops/1e6:.3f} MFLOPs")
print(f"FLOPs: {flops/1e9:.6f} B")
