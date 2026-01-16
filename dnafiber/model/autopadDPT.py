import torch.nn as nn


class AutoPad(nn.Module):
    def __init__(self, module, divisible_by=14):
        super().__init__()
        self.module = module
        self.divisible_by = divisible_by

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]
        pad_h = (-height) % self.divisible_by
        pad_w = (-width) % self.divisible_by

        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        x = self.module(x)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :height, :width]

        return x

    def __repr__(self):
        return f"AutoPad({self.module.__class__.__name__})"
