import os, sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import einops






class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))



class PatchPooling(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, patch_stride=2) -> None:
        super().__init__()
        self.patch_stride = patch_stride
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
    
    def forward(self, x):
        print(f'{x.shape=}')
        patch_x = torch.cat(
            [x[..., i::self.patch_stride, j::self.patch_stride] for i in range(self.patch_stride) for j in range(self.patch_stride)], 
            dim=1
        )
        print(f'{patch_x.shape=}')
        
        return self.conv(
            # 640, 640, 3 => 320, 320, 12
            patch_x
        )


class PatchFeature(nn.Module):
    def __init__(self, patch_size=2) -> None:
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        print(f'{x.shape=}')
        patch_x = einops.rearrange(
            x, 'b c (p1 h) (p2 w) -> b (p1 p2  c) h w',
            p1=self.patch_size, p2=self.patch_size
        )
        print(f'{patch_x.shape=}')
        return patch_x


class PatchFeatureConv(nn.Module):
    def __init__(self, input_channels, output_channels, norm=True, act='relu6', patch_size=2) -> None:
        super().__init__()
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(input_channels, output_channels, kernel_size=patch_size, stride=patch_size)
        if norm:
            self.norm = nn.BatchNorm2d(output_channels, eps=1e-5)
        else:
            self.norm = None
        if act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            self.act = None

    def forward(self, x):
        print(f'{x.shape=}')
        patch_x = self.proj(x)
        print(f'[proj]  {patch_x.shape=}  {patch_x.min()=} / {patch_x.max()=}')
        if self.norm is not None:
            patch_x = self.norm(patch_x)
            print(f'[norm]  {patch_x.shape=}  {patch_x.min()=} / {patch_x.max()=}')
        if self.act is not None:
            patch_x = self.act(patch_x)
            print(f'[act]  {patch_x.shape=}  {patch_x.min()=} / {patch_x.max()=}')
        print(f'{patch_x.shape=}')
        return patch_x



if __name__ == '__main__':
    
    import numpy as np
    import torch

    a = np.linspace(0, 99, 100).reshape((1, 10, 10))
    aa = np.array([a,a])
    aaa = torch.tensor(aa, dtype=torch.float)
    print(f'{aaa.shape=}')


    pp = PatchPooling(1, 3, 1, 1, patch_stride=2)
    bbb = pp(aaa)
    print(f'{bbb.shape}')
    print(f'='*80 + '\n')
    
    a = np.linspace(0, 99, 100).reshape((1, 10, 10))
    aa = np.array([a,a])
    aaa = torch.tensor(aa, dtype=torch.float)
    print(f'{aaa.shape=}')
    patch_size = 5
    
    pf = PatchFeature(patch_size=patch_size)
    bb = pf(aaa)
    print(f'{bb.shape=}')
    print(f'='*80 + '\n')
    
    
    a = np.linspace(0, 99, 100).reshape((1, 10, 10))
    aa = np.array([a,a])
    aaa = torch.tensor(aa, dtype=torch.float)
    print(f'{aaa.shape=}')
    patch_size = 2
    
    pf = PatchFeatureConv(input_channels=1, output_channels=10, norm=True, act='relu6', patch_size=patch_size)
    bb = pf(aaa)
    print(f'{bb.shape=}')
    print(f'='*80 + '\n')
    