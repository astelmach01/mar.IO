from torch import nn
import torch
import copy


class MarioNet(nn.Module):
    """mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, dtype=torch.float),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, dtype=torch.float),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dtype=torch.float),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(3136, 2048),
            nn.SELU(),
            nn.Linear(2048, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input_tensor, model):
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        print(input_tensor.shape)
        if model == "online":
            return self.online(input_tensor)
        elif model == "target":
            return self.target(input_tensor)
