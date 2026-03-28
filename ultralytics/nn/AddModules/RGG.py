import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, GhostConv, RepConv, autopad

# 1. RGGConv: 论文 Fig. 6(b)
# 结构: Input(C) -> [Split] -> RepConv(C) -> [Split] -> GhostConv x2 (C)
#       Input(C) -------------------------------------> Concat (3C) -> 1x1 Conv -> Output(C)
#       RepConv(C) ----------------------------------->
class RGGConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # 假设 c1 = c2，因为它是作为 Bottleneck 内部使用的
        # 论文提到: RepConv + 2 * GhostConv
        self.rep_conv = RepConv(c1, c2, 3, 1) # k=3, s=1
        
        # GhostConv x 2: 用两个 GhostConv 串联
        # 第一个 GhostConv: c2 -> c2 (k=3, s=1)
        # 第二个 GhostConv: c2 -> c2 (k=3, s=1)
        self.ghost1 = GhostConv(c2, c2, 3, 1)
        self.ghost2 = GhostConv(c2, c2, 3, 1)
        
        # 最后的 1x1 卷积，将拼接后的 3倍通道 降维回 c2
        # Concat sources: [Input, RepConv_Out, GhostStack_Out] -> c1 + c2 + c2 = 3*c (if c1=c2)
        self.conv_1x1 = Conv(c1 + c2 * 2, c2, 1, 1)

    def forward(self, x):
        # 1. RepConv 分支
        x_rep = self.rep_conv(x)
        
        # 2. GhostConv 分支 (串联)
        x_ghost = self.ghost2(self.ghost1(x_rep))
        
        # 3. 拼接: Input + RepConv_Out + Ghost_Out [cite: 314]
        # 论文图6(b)显示 Input 也连到了 Concat
        cat = torch.cat([x, x_rep, x_ghost], dim=1)
        
        # 4. 融合输出
        return self.conv_1x1(cat)

# 2. RGGBottleneck: 论文 Fig. 6(c)
# 结构: Conv(1x1) -> RGGConv -> Add (Residual)
class RGGBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 第一层是普通卷积 (通常是 1x1 或 3x3，YOLOv8 Bottleneck 第一层是 3x3)
        # 但图 6(c) 显示 "Conv -> RGGConv"
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.rgg_conv = RGGConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.rgg_conv(self.cv1(x)) if self.add else self.rgg_conv(self.cv1(x))

# 3. RGGModule (替代 C2f): 论文 Fig. 6(d) / Fig. 5
# 结构几乎与 C2f 一样，只是把内部的 Bottleneck 换成了 RGGBottleneck
class RGGModule(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # 使用 RGGBottleneck
        self.m = nn.ModuleList(RGGBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))