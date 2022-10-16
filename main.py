import torch
import torch.nn as nn


def main():
    t1 = torch.randn(5, requires_grad=True)
    t1s = torch.sigmoid(t1)
    t2 = torch.randint(0, 2, (5,)).float()

    mse = nn.MSELoss()
    bce = nn.BCELoss()
    bce_logits = nn.BCEWithLogitsLoss()

    print(t1)
    print(t1s)
    print(t2)
    print(mse(t1, t2))
    print(bce(t1s, t2))
    print(bce_logits(t1, t2))


if __name__ == '__main__':
    main()
