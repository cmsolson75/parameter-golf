import torch
from torch import nn


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32)) # D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :] # 1, 1, D
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev



def main() -> None:

    x = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0]]])  # (1, 5, 1)
    breakpoint()
    smear = SmearGate(1)

    with torch.no_grad():
        smear.gate[:] = torch.tensor([0.0])  # sigmoid(0)=0.5

    out = smear(x)

    print("x.squeeze()    =", x.squeeze())
    print("out.squeeze()  =", out.squeeze())


if __name__ == "__main__":
    main()