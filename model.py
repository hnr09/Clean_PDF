"""필기 제거 + 복원용 U-Net. 출력: (복원 이미지, 필기 마스크 logit)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class EraserUNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        ch = [c, c * 2, c * 4, c * 8, c * 8]
        self.enc = nn.ModuleList([block(1, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(4)])
        self.dec = nn.ModuleList([block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], 2, 1)

    def forward(self, x):
        feats, h = [], x
        for i, e in enumerate(self.enc):
            h = e(h if i == 0 else F.max_pool2d(h, 2))
            feats.append(h)
        h = feats.pop()
        for d in self.dec:
            s = feats.pop()
            h = d(torch.cat([F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        out = self.head(h)
        clean = torch.clamp(x + out[:, :1], 0, 1)  # 잔차 방식: 입력에서 필기만 빼기/채우기
        return clean, out[:, 1:]
